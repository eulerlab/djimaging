"""Make the structural channel (e.g. SR101) usable as an activity- and stimulus-independent reference.

Without this, stimulus light leaking into the structural channel and bleed-through of the functional indicator
(e.g. OGB-1) make the structural channel change with the stimulus, which looks like z motion.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from djimaging.utils.motion.image_utils import bin_stack, valid_region


def _highpass_bins(b: np.ndarray, hp_sigma_bins: float = 5) -> np.ndarray:
    """(x, y, nb) -> (nb, x*y) high-passed time courses. Slow z drift must not leak into the estimates."""
    bt = b.reshape(-1, b.shape[2]).T.astype(float)
    return bt - ndimage.gaussian_filter1d(bt, sigma=hp_sigma_bins, axis=0, mode='nearest')


def _corr_with(bt: np.ndarray, s: np.ndarray) -> np.ndarray:
    with np.errstate(invalid='ignore', divide='ignore'):
        r = (bt * s[:, None]).sum(0) / (np.sqrt((bt ** 2).sum(0)) * np.sqrt((s ** 2).sum()))
    return np.nan_to_num(r)


def fill_artifact_lines(stack: np.ndarray, npixartifact: int) -> np.ndarray:
    """Copy of ``stack`` with the artifact lines replaced by the first valid line.

    Otherwise a band-pass filter smears the bright artifact ~10 px into the image, so that the band-passed structure
    changes with the stimulus.
    """
    out = np.array(stack, dtype=np.float32, copy=True)
    if npixartifact >= 1:
        out[:npixartifact] = out[npixartifact:npixartifact + 1]
    return out


def clean_channels(
        ch0: np.ndarray,
        ch1: np.ndarray | None,
        npixartifact: int,
        bin_frames: int = 16,
        r_thresh: float = 0.5,
        col_thresh: float = 0.5,
        dilate: int = 1,
        max_extra_lines: int = 12,
) -> tuple[np.ndarray, np.ndarray | None, dict]:
    """Prepare functional (ch0) and structural (ch1) stacks for motion estimation and motion QC (copies).

    1. Light-artifact width: lines next to the ``npixartifact`` lines whose structural-channel time course follows the
       artifact lines (pure stimulus light) are added to the excluded lines (e.g. 2018 setup-3 data: 4-6 instead of 3).
    2. Bleed-through of indicator activity into ch1: ``ch1 -= k * (ch0 - mean_t(ch0))`` with one global ``k``.
       Removes activity, keeps the static structure.
    3. Remaining ch1 pixels that correlate with the artifact are filled with the per-frame median of clean pixels.
    4. The (extended) artifact lines of both channels are replaced by the first valid line.

    All correlations use 2 s bins (``bin_frames``), high-passed, with the per-bin frame median removed.

    Returns:
        ch0, ch1 (None if not given), info dict with ``npixartifact`` (extended), ``k_bleed`` and ``art_mask`` (ch1
        pixels filled in step 3, or None).
    """
    info = dict(npixartifact=int(npixartifact), k_bleed=0., art_mask=None)
    if ch1 is None or npixartifact < 1:
        return fill_artifact_lines(ch0, npixartifact), (None if ch1 is None else fill_artifact_lines(
            ch1, npixartifact)), info

    nx, ny = ch1.shape[:2]
    ch1 = np.array(ch1, dtype=np.float32, copy=True)
    b1, _ = bin_stack(ch1, bin_frames)
    b0, _ = bin_stack(ch0, bin_frames)
    s_art = _highpass_bins(b1[:npixartifact]).mean(axis=1)
    if np.std(s_art) <= 0:
        return fill_artifact_lines(ch0, npixartifact), fill_artifact_lines(ch1, npixartifact), info

    # 1) artifact width
    r_cols = np.median(_corr_with(_highpass_bins(b1), s_art).reshape(nx, ny), axis=1)
    npix = npixartifact
    while npix < min(nx - 1, npixartifact + max_extra_lines) and r_cols[npix] > col_thresh:
        npix += 1

    # 2) bleed-through of indicator activity into ch1
    valid = valid_region((nx, ny), npix).ravel()
    d0 = _highpass_bins(b0 - np.median(b0[npix:], axis=(0, 1))[None, None, :])[:, valid]
    d1 = _highpass_bins(b1 - np.median(b1[npix:], axis=(0, 1))[None, None, :])[:, valid]
    denom = (d0 * d0).sum()
    k = max(float((d0 * d1).sum() / denom), 0.) if denom > 0 else 0.
    if k > 0:
        ch1 = (ch1 - k * (ch0 - ch0.mean(axis=2, keepdims=True))).astype(np.float32)
        b1, _ = bin_stack(ch1, bin_frames)

    # 3) remaining localized artifact pixels
    r = _corr_with(_highpass_bins(b1 - np.median(b1[npix:], axis=(0, 1))[None, None, :]), s_art).reshape(nx, ny)
    art = r > r_thresh
    art[:npix, :] = False
    if dilate and art.any():
        art = ndimage.binary_dilation(art, iterations=dilate)
        art[:npix, :] = False
    if art.any():
        good = valid_region(art.shape, npix) & ~art
        ch1[art, :] = np.median(ch1[good], axis=0)[None, :]

    info.update(npixartifact=int(npix), k_bleed=float(k), art_mask=art if art.any() else None)
    return fill_artifact_lines(ch0, npix), fill_artifact_lines(ch1, npix), info
