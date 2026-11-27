"""Relative z proxy: similarity of the imaged structure after removing xy motion.

There is usually no z-stack, so z cannot be measured in um. Instead, the band-passed structure of each time bin is
compared with the recording's own mean (within-recording change) and with the field reference (plane vs. reference),
noise-corrected with split-half reliabilities. Drops below ~1 mean the structure changed beyond xy translation
(defocus / z change / deformation). The structural channel (e.g. SR101) is activity-independent.

Validated against a z-stack (zstack/20190130): similarity 0.95 ~ 1-2 um, 0.90 ~ 2-3 um, 0.75-0.8 ~ 4 um,
0.5-0.6 ~ 6 um (order of magnitude; depends on cell size, PSF and objective).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from djimaging.utils.motion.image_utils import (
    bandpass, bin_edges, bin_half_stacks, bp_fill, bp_fill_stack, half_means, masked_pearson, pearson,
    register_stack, spearman_brown, valid_region)
from djimaging.utils.motion.registration import union_masks

# Similarity (noise-corrected corr of band-passed images at recording resolution) vs. |z offset| [um], from one z-stack
# (zstack/20190130, docs/motion_qc.md "z-stack check"). Depends on cell size, PSF and objective: order of magnitude.
Z_CALIBRATION = {
    'ch0': (np.array([1.0, 0.95, 0.90, 0.775, 0.55]), np.array([0., 1.65, 2.25, 3.75, 6.])),
    'ch1': (np.array([1.0, 0.95, 0.90, 0.775, 0.55]), np.array([0., 1.5, 2.5, 4., 5.5])),
}


def dz_from_similarity(sim, channel: str = 'ch1'):
    """Rough |z offset| [um] for a noise-corrected similarity; clipped to the calibrated range, NaN stays NaN.

    Scalar in, float out; array in, array out. No sign: a similarity says how far, not in which direction.
    """
    s, z = Z_CALIBRATION[channel]
    x = np.asarray(sim if sim is not None else np.nan, dtype=float)
    out = np.where(np.isfinite(x), np.interp(np.nan_to_num(x), s[::-1], z[::-1]), np.nan)
    return float(out) if out.ndim == 0 else out


def _zproxy_channel(stack, ref_img, ref_halves, disp_frames, edges, valid):
    reg = register_stack(stack, disp_frames)
    sm, sa, sb = [bp_fill(x).ravel() for x in half_means(reg)]
    rm = bandpass(ref_img).ravel()
    ra, rb = [bandpass(x).ravel() for x in ref_halves]
    r_self = spearman_brown(pearson(sa[valid], sb[valid]))
    r_ref = spearman_brown(pearson(ra[valid], rb[valid]))
    nb = len(edges) - 1
    M, A, B = [bp_fill_stack(x).reshape(-1, nb).T for x in bin_half_stacks(reg, edges)]
    W = valid[None, :].astype(float)
    c_self = masked_pearson(M, sm, W)[0]
    c_ref = masked_pearson(M, rm, W)[0]
    r_bin = spearman_brown(np.nanmedian(masked_pearson(A, B, W)[0]))
    return (c_self / np.sqrt(r_bin * r_self), c_ref / np.sqrt(r_bin * r_ref),
            pearson(sm[valid], rm[valid]) / np.sqrt(r_self * r_ref))


def compute_zproxy(ch0: np.ndarray, ch1: np.ndarray | None, ref: dict, disp_frames: np.ndarray, npixartifact: int,
                   art_mask: np.ndarray | None = None, bin_frames: int = 16, border: int = 4) -> dict:
    """Noise-corrected similarity per bin after removing xy motion (raw displacement).

    ``corr_d = corr(bin, target) / sqrt(R_bin * R_target)`` with reliabilities from even/odd frames (Spearman-Brown),
    ``R_bin`` pooled over the recording (median).

    Returns:
        dict with ``bin_edges`` (frames), ``self_ch0``, ``ref_ch0`` (n_bins,), ``rec_ref_ch0`` (whole recording vs.
        reference) and the same for ch1 (None without structural channel).
    """
    npix = max(int(npixartifact), int(ref['npixartifact']))
    valid_all = valid_region(ch0.shape, npix, border=border)
    edges = bin_edges(ch0.shape[2], bin_frames)
    out = dict(bin_edges=edges)
    s0, r0, rr0 = _zproxy_channel(ch0, ref['ch0'], ref['ch0_halves'], disp_frames, edges, valid_all.ravel())
    out.update(self_ch0=s0, ref_ch0=r0, rec_ref_ch0=float(rr0))
    if ch1 is not None and ref.get('ch1') is not None:
        excl1 = union_masks(art_mask, ref.get('art_mask'))
        valid1 = valid_all if excl1 is None else valid_all & ~ndimage.binary_dilation(excl1, iterations=2)
        s1, r1, rr1 = _zproxy_channel(ch1, ref['ch1'], ref['ch1_halves'], disp_frames, edges, valid1.ravel())
        out.update(self_ch1=s1, ref_ch1=r1, rec_ref_ch1=float(rr1))
    else:
        out.update(self_ch1=None, ref_ch1=None, rec_ref_ch1=None)
    return out
