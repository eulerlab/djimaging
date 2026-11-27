"""Per-ROI trust over time: can the trace of an ROI be trusted in a given time bin?

Trust is the fraction of the ROI mask that is on its cell (xy only, a geometric quantity): from the residual
displacement of the ROI's neighbourhood (after the applied corrections), the uncorrected fast motion of every frame and
the FOV coverage. z (defocus / plane change) cannot be measured per ROI; it is reported separately per time bin as an
approximate within-recording z change [um] from the structural channel (``dz_um``). Both are structural (gain / offset
invariant), so drug-induced brightness changes do not lower them.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy import ndimage, signal

from djimaging.utils.motion.image_utils import (
    bandpass, bin_half_stacks, bp_fill, bp_fill_stack, half_means, masked_pearson, spearman_brown, valid_region)
from djimaging.utils.motion.roi_shifts import gather_patches, local_shifts_batch, roi_ids_from_mask
from djimaging.utils.motion.zproxy import dz_from_similarity


@dataclass
class TrustResult:
    roi_ids: np.ndarray
    bin_edges: np.ndarray  # frame indices
    trust: np.ndarray  # (n_rois, n_bins) fraction of the ROI on its cell in [0, 1] (mean over the frames of the bin)
    res_vec: np.ndarray  # (n_rois, n_bins, 2) local displacement vs. field reference - applied ROI shift [px]
    res_shift: np.ndarray  # (n_rois, n_bins) |res_vec| [px]
    coverage: np.ndarray  # (n_rois, n_bins) fraction of the ROI mask inside the FOV
    fast: np.ndarray  # (n_bins,) max |fast| (uncorrected) global motion within the bin [px]
    dz_um: np.ndarray  # (n_bins,) approx. |z change| vs. the recording's mean plane [um] (structural channel; NaN)
    corr_self: np.ndarray  # (n_rois, n_bins) diagnostic, not in trust: noise-corrected patch corr vs. own mean
    match_corr: np.ndarray  # (n_rois,) noise-corrected patch corr recording mean vs. field reference
    match_shift: np.ndarray  # (n_rois, 2) local displacement recording mean vs. field reference - applied shift [px]
    reliability: np.ndarray  # (n_rois,) split-half reliability of the ROI patch in the recording mean


def roi_overlap_tables(roi_mask: np.ndarray, roi_ids: np.ndarray) -> list:
    """Per ROI the normalized autocorrelation of its mask, A(k) = |M & (M + k)| / |M| for integer k.

    For a binary mask and a bilinearly shifted copy, the overlap at a subpixel displacement d is the bilinear
    interpolation of A at d: the fraction of the ROI's pixels that are still on the cell when the cell moved by d
    (symmetric in d).
    """
    tables = []
    for rid in roi_ids:
        m = roi_mask == rid
        xs, ys = np.nonzero(m)
        if len(xs) == 0:
            tables.append(np.zeros((1, 1)))
            continue
        c = m[xs.min():xs.max() + 1, ys.min():ys.max() + 1].astype(float)
        tables.append(signal.correlate2d(c, c, mode='full') / c.sum())
    return tables


def on_cell_fraction(table: np.ndarray, disp: np.ndarray) -> np.ndarray:
    """Overlap fraction for displacements ``disp`` (..., 2) [px] from an ``roi_overlap_tables`` entry."""
    disp = np.asarray(disp, dtype=float)
    cx, cy = (table.shape[0] - 1) / 2., (table.shape[1] - 1) / 2.
    pts = disp.reshape(-1, 2)
    vals = ndimage.map_coordinates(table, [pts[:, 0] + cx, pts[:, 1] + cy], order=1, mode='constant', cval=0.)
    vals[~np.all(np.isfinite(pts), axis=1)] = 0.  # unknown position: not trusted
    return vals.reshape(disp.shape[:-1])


def trust_on_cell(roi_mask: np.ndarray, roi_ids: np.ndarray, res_vec: np.ndarray, edges: np.ndarray,
                  fast_frames: np.ndarray | None, coverage: np.ndarray) -> np.ndarray:
    """Fraction of each ROI on its cell per bin: mean over the frames of the bin, times the FOV coverage.

    The displacement of a frame is the ROI's residual of its bin (``res_vec``, measured on the bin mean) plus the
    deviation of the frame's uncorrected fast motion from the bin mean (the bin mean is already in ``res_vec``).
    """
    nb = len(edges) - 1
    frames = np.arange(int(edges[0]), int(edges[-1]))
    b_idx = np.clip(np.searchsorted(edges, frames, side='right') - 1, 0, nb - 1)
    if fast_frames is None:
        dev = np.zeros((len(frames), 2))
    else:
        fast = np.asarray(fast_frames, dtype=float)[frames]
        means = np.stack([np.bincount(b_idx, fast[:, k], minlength=nb) for k in range(2)], 1) \
            / np.maximum(np.bincount(b_idx, minlength=nb), 1)[:, None]
        dev = fast - means[b_idx]
    counts = np.maximum(np.bincount(b_idx, minlength=nb), 1)
    out = np.zeros((len(roi_ids), nb))
    for i, table in enumerate(roi_overlap_tables(roi_mask, roi_ids)):
        frac = on_cell_fraction(table, res_vec[i][b_idx] + dev)
        out[i] = np.bincount(b_idx, frac, minlength=nb) / counts
    return out * np.clip(np.nan_to_num(coverage), 0, 1)


def stim_bins(bin_starts: np.ndarray, fs: float, triggertimes) -> tuple[np.ndarray, float | None]:
    """Bins that start at or after the stimulus onset (first trigger), and the onset [s].

    Motion before the stimulus (e.g. the focus being adjusted after the recording was started) should not count in
    summaries. Without triggers, or if no bin starts after the onset, all bins are used.
    """
    bin_starts = np.asarray(bin_starts, dtype=float)
    tt = np.asarray(triggertimes if triggertimes is not None else [], dtype=float).ravel()
    tt = tt[np.isfinite(tt)]
    if tt.size == 0:
        return np.ones(len(bin_starts), bool), None
    onset = float(tt[0])
    sel = bin_starts >= np.floor(onset * fs)
    if not sel.any():
        sel = np.ones(len(bin_starts), bool)
    return sel, onset


def fast_per_bin(fast_bins: np.ndarray, bin_centers: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Maximum |fast motion| of the shift bins within each trust bin."""
    fast_abs = np.linalg.norm(fast_bins, axis=1)
    out = np.zeros(len(edges) - 1)
    for b in range(len(out)):
        sel = (bin_centers >= edges[b]) & (bin_centers < edges[b + 1])
        out[b] = fast_abs[sel].max() if sel.any() else 0.
    return out


def compute_trust(reg0: np.ndarray, ref: dict, roi_mask: np.ndarray, edges: np.ndarray,
                  fast_frames: np.ndarray | None, npixartifact: int, zself_ch1: np.ndarray | None = None,
                  roi_shifts: np.ndarray | None = None, dilate: int = 2, patch_half: int = 6,
                  min_rel: float = 0.1) -> TrustResult:
    """Per-ROI trust per time bin for traces extracted from the slowly registered stack ``reg0`` (functional channel).

    ``trust`` = fraction of the ROI mask on its cell (see ``trust_on_cell``), from

    - res_vec: local displacement of the ROI neighbourhood (band-passed patch of the bin mean) vs. the *field
      reference*, relative to the applied between-presentation shift ``roi_shifts`` -> is the mask still on the cell?
    - fast_frames: (n_frames, 2) uncorrected fast global motion of every frame (raw - slow displacement).
    - coverage: fraction of the ROI mask inside the valid FOV.

    The ROI shape matters: the same displacement moves a small ROI further off its cell than a large one.

    Reported separately (not in trust):

    - dz_um: approx. |z change| vs. the recording's own mean plane [um], from the structural-channel similarity
      ``zself_ch1`` (see ``compute_zproxy``, ``dz_from_similarity``); field level, NaN without structural channel.
    - corr_self: noise-corrected Pearson corr of the band-passed ROI patch (ROI dilated by ``dilate`` px) vs. the
      recording's own mean.
    - Across presentations (one value per ROI; brightness changes, e.g. by a drug, lower it): noise-corrected patch
      corr and local shift of the recording mean vs. the field reference.
    """
    nb = len(edges) - 1
    roi_ids = roi_ids_from_mask(roi_mask)
    n = len(roi_ids)
    roi_shifts = np.zeros((n, 2)) if roi_shifts is None else np.asarray(roi_shifts, float)
    valid0 = valid_region(roi_mask.shape, npixartifact)

    self_m, self_a, self_b = [bp_fill(x) for x in half_means(reg0)]
    ref_m = bandpass(ref['ch0'])
    ref_a, ref_b = [bandpass(x) for x in ref['ch0_halves']]

    p = roi_mask.size
    wm, wp = np.zeros((n, p)), np.zeros((n, p))
    cx, cy = np.zeros(n, int), np.zeros(n, int)
    for i, rid in enumerate(roi_ids):
        m = roi_mask == rid
        wm[i] = m.ravel()
        wp[i] = (ndimage.binary_dilation(m, iterations=dilate) & valid0).ravel()
        cx[i], cy[i] = [int(round(c)) for c in ndimage.center_of_mass(m)]

    pad, h = patch_half + 3, patch_half + 2
    ref_patches = gather_patches(np.pad(ref_m, pad)[None], cx, cy, pad, h)[0]

    r_self = spearman_brown(masked_pearson(self_a.ravel()[None], self_b.ravel(), wp)[:, 0])
    r_ref = spearman_brown(masked_pearson(ref_a.ravel()[None], ref_b.ravel(), wp)[:, 0])
    match_corr = (masked_pearson(self_m.ravel()[None], ref_m.ravel(), wp)[:, 0]
                  / np.sqrt(np.maximum(r_self, min_rel) * np.maximum(r_ref, min_rel)))
    match_shift = local_shifts_batch(ref_patches, gather_patches(np.pad(self_m, pad)[None], cx, cy, pad, h))[0] \
        - roi_shifts

    m_raw, a_raw, b_raw = bin_half_stacks(reg0, edges)
    f = (np.isfinite(m_raw) & valid0[:, :, None]).reshape(p, nb).T.astype(float)
    m_bp, a_bp, b_bp = [bp_fill_stack(x).reshape(p, nb).T for x in (m_raw, a_raw, b_raw)]
    coverage = (f @ wm.T / np.maximum(wm.sum(axis=1), 1)[None, :]).T
    c_self = masked_pearson(m_bp, self_m.ravel(), wp, f)
    r_half = masked_pearson(a_bp, b_bp, wp, f)
    img_pad = np.pad(m_bp.reshape(nb, *roi_mask.shape), ((0, 0), (pad, pad), (pad, pad)))
    res_vec = np.transpose(local_shifts_batch(ref_patches, gather_patches(img_pad, cx, cy, pad, h))
                           - roi_shifts[None], (1, 0, 2))  # (n_rois, n_bins, 2)

    with warnings.catch_warnings():  # ROIs without any valid bin yield NaN, handled below (min_rel)
        warnings.simplefilter('ignore', RuntimeWarning)
        r_bin = spearman_brown(np.nanmedian(r_half, axis=1))
    corr_self = c_self / np.sqrt(np.maximum(r_bin, min_rel) * np.maximum(r_self, min_rel))[:, None]

    edges = np.asarray(edges)
    fast = np.zeros(nb) if fast_frames is None else fast_max_per_bin(fast_frames, edges)
    trust = trust_on_cell(roi_mask, roi_ids, res_vec, edges, fast_frames, coverage)
    dz_um = np.full(nb, np.nan) if zself_ch1 is None else np.asarray(dz_from_similarity(zself_ch1, 'ch1'))
    return TrustResult(roi_ids=roi_ids, bin_edges=edges, trust=trust, res_vec=res_vec,
                       res_shift=np.linalg.norm(res_vec, axis=-1), coverage=coverage, fast=fast, dz_um=dz_um,
                       corr_self=corr_self, match_corr=match_corr, match_shift=match_shift, reliability=r_self)


def fast_max_per_bin(fast_frames: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Maximum |fast motion| of the frames within each bin [px]."""
    fa = np.linalg.norm(np.asarray(fast_frames, dtype=float), axis=1)
    return np.array([fa[int(a):int(b)].max() if b > a else 0. for a, b in zip(edges[:-1], edges[1:])])


def trust_per_repeat(trust: np.ndarray, edges: np.ndarray, fs: float, triggertimes: np.ndarray,
                     ntrigger_rep: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Minimum trust over the bins overlapping each stimulus repeat.

    Returns (n_rois, n_repeats) and the repeat start times, or (None, None) without repeats (``ntrigger_rep`` <= 1 or
    fewer triggers than one repeat).
    """
    return per_repeat(trust, edges, fs, triggertimes, ntrigger_rep, agg=np.min)


def per_repeat(x: np.ndarray, edges: np.ndarray, fs: float, triggertimes: np.ndarray, ntrigger_rep: int,
               agg=np.min) -> tuple[np.ndarray | None, np.ndarray | None]:
    """``agg`` over the bins overlapping each stimulus repeat of ``x`` (..., n_bins) -> (..., n_repeats).

    NaN bins are ignored (all NaN -> NaN). Returns (None, None) without repeats.
    """
    triggertimes = np.asarray(triggertimes, dtype=float)
    if ntrigger_rep is None or ntrigger_rep <= 1 or len(triggertimes) < ntrigger_rep:
        return None, None
    x = np.asarray(x, dtype=float)
    starts = triggertimes[::ntrigger_rep]
    rep_dur = np.median(np.diff(starts)) if len(starts) > 1 else edges[-1] / fs - starts[0]
    t_edges = np.asarray(edges) / fs
    out = np.full(x.shape[:-1] + (len(starts),), np.nan)
    nan_agg = {np.min: np.nanmin, np.max: np.nanmax}.get(agg, agg)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for r, t0 in enumerate(starts):
            sel = (t_edges[1:] > t0) & (t_edges[:-1] < t0 + rep_dur)
            if sel.any():
                out[..., r] = nan_agg(x[..., sel], axis=-1)
    return out, starts
