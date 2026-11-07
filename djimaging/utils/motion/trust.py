"""Per-ROI trust over time: can the trace of an ROI be trusted in a given time bin?

All components are structural (gain / offset invariant), so drug-induced brightness changes do not lower the trust.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from djimaging.utils.motion.image_utils import (
    bandpass, bin_half_stacks, bp_fill, bp_fill_stack, half_means, masked_pearson, soft, spearman_brown, valid_region)
from djimaging.utils.motion.roi_shifts import gather_patches, local_shifts_batch, roi_ids_from_mask


@dataclass
class TrustResult:
    roi_ids: np.ndarray
    bin_edges: np.ndarray  # frame indices
    corr_self: np.ndarray  # (n_rois, n_bins) noise-corrected patch corr vs. the recording's own mean
    res_shift: np.ndarray  # (n_rois, n_bins) |local displacement vs. field reference - applied ROI shift| [px]
    coverage: np.ndarray  # (n_rois, n_bins) fraction of the ROI mask inside the FOV
    fast: np.ndarray  # (n_bins,) max |fast| (uncorrected) global motion within the bin [px]
    zfield: np.ndarray  # (n_bins,) field-level z factor from the structural channel in [0, 1]
    trust: np.ndarray  # (n_rois, n_bins) in [0, 1]
    match_corr: np.ndarray  # (n_rois,) noise-corrected patch corr recording mean vs. field reference
    match_shift: np.ndarray  # (n_rois, 2) local displacement recording mean vs. field reference - applied shift [px]
    reliability: np.ndarray  # (n_rois,) split-half reliability of the ROI patch in the recording mean


def fast_per_bin(fast_bins: np.ndarray, bin_centers: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Maximum |fast motion| of the shift bins within each trust bin."""
    fast_abs = np.linalg.norm(fast_bins, axis=1)
    out = np.zeros(len(edges) - 1)
    for b in range(len(out)):
        sel = (bin_centers >= edges[b]) & (bin_centers < edges[b + 1])
        out[b] = fast_abs[sel].max() if sel.any() else 0.
    return out


def compute_trust(reg0: np.ndarray, ref: dict, roi_mask: np.ndarray, edges: np.ndarray, fast: np.ndarray,
                  npixartifact: int, zself_ch1: np.ndarray | None = None, roi_shifts: np.ndarray | None = None,
                  dilate: int = 2, patch_half: int = 6, corr_lo: float = 0.5, corr_hi: float = 0.85,
                  res_scale: float = 0.75, fast_scale: float = 0.75, z_lo: float = 0.75, z_hi: float = 0.95,
                  min_rel: float = 0.1) -> TrustResult:
    """Per-ROI trust per time bin for traces extracted from the slowly registered stack ``reg0`` (functional channel).

    ``trust = soft(corr_self) * gauss(res_shift) * gauss(fast) * soft(zfield) * coverage``

    - corr_self: noise-corrected Pearson corr of the band-passed ROI patch (ROI dilated by ``dilate`` px) vs. the
      recording's own mean -> defocus / shape change during the recording.
    - res_shift: local displacement of the ROI neighbourhood vs. the *field reference*, relative to the applied
      between-presentation shift ``roi_shifts`` -> is the mask still on the cell?
    - fast: uncorrected fast global motion within the bin.
    - zfield: structural-channel similarity vs. the recording's own mean (``zself_ch1``, see ``compute_zproxy``).
    - coverage: fraction of the ROI mask inside the valid FOV.

    Across presentations (one value per ROI, not part of trust because brightness changes, e.g. by a drug, lower it):
    noise-corrected patch corr and local shift of the recording mean vs. the field reference.
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
    res = np.linalg.norm(local_shifts_batch(ref_patches, gather_patches(img_pad, cx, cy, pad, h))
                         - roi_shifts[None], axis=-1).T

    r_bin = spearman_brown(np.nanmedian(r_half, axis=1))
    corr_self = c_self / np.sqrt(np.maximum(r_bin, min_rel) * np.maximum(r_self, min_rel))[:, None]
    zfield = np.ones(nb) if zself_ch1 is None else soft(zself_ch1, z_lo, z_hi)
    trust = (soft(corr_self, corr_lo, corr_hi) * np.exp(-0.5 * (res / res_scale) ** 2)
             * np.exp(-0.5 * (np.asarray(fast) / fast_scale) ** 2)[None, :] * zfield[None, :]
             * np.clip(coverage, 0, 1))
    return TrustResult(roi_ids=roi_ids, bin_edges=np.asarray(edges), corr_self=corr_self, res_shift=res,
                       coverage=coverage, fast=np.asarray(fast), zfield=zfield, trust=np.nan_to_num(trust),
                       match_corr=match_corr, match_shift=match_shift, reliability=r_self)


def trust_per_repeat(trust: np.ndarray, edges: np.ndarray, fs: float, triggertimes: np.ndarray,
                     ntrigger_rep: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Minimum trust over the bins overlapping each stimulus repeat.

    Returns (n_rois, n_repeats) and the repeat start times, or (None, None) without repeats (``ntrigger_rep`` <= 1 or
    fewer triggers than one repeat).
    """
    triggertimes = np.asarray(triggertimes, dtype=float)
    if ntrigger_rep is None or ntrigger_rep <= 1 or len(triggertimes) < ntrigger_rep:
        return None, None
    starts = triggertimes[::ntrigger_rep]
    rep_dur = np.median(np.diff(starts)) if len(starts) > 1 else edges[-1] / fs - starts[0]
    t_edges = np.asarray(edges) / fs
    out = np.full((trust.shape[0], len(starts)), np.nan)
    for r, t0 in enumerate(starts):
        sel = (t_edges[1:] > t0) & (t_edges[:-1] < t0 + rep_dur)
        if sel.any():
            out[:, r] = trust[:, sel].min(axis=1)
    return out, starts
