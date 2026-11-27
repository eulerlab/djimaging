"""Rigid subpixel registration of low-resolution two-photon recordings to a field reference.

``disp`` is the displacement of the field relative to the reference (px, (x, y)): content at reference position p is
found at p + disp in the recording. Registering a frame means shifting it by -disp.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from djimaging.utils.motion.image_utils import (
    bin_stack, fill_nan, half_means, interp_frames, pearson, reg_image, register_stack, shift_image)


def xcorr_disp(ref: np.ndarray, img: np.ndarray, max_shift: int = 8, whiten: float = 0.5,
               smooth_sigma: float = 1.0, upsample: int = 20) -> tuple[np.ndarray, float]:
    """Displacement of ``img`` relative to ``ref`` from a partially whitened FFT cross-correlation.

    Args:
        ref, img: Registration images (see ``reg_image``).
        max_shift: Search window in px; prevents jumps to spurious far-away peaks.
        whiten: Exponent of the spectral normalization (0: plain cross-correlation, 1: phase correlation). Partial
            whitening keeps the sharpness of phase correlation without amplifying noise in blurred / low-SNR images
            (full phase correlation failed under defocus).
        smooth_sigma: Gaussian smoothing of the correlation surface in px.
        upsample: Subpixel refinement with the matrix-multiply DFT around the integer peak.

    Returns:
        disp (2,) in px and the peak quality (peak height relative to the spread of the correlation surface).
    """
    from skimage.registration._phase_cross_correlation import _upsampled_dft
    R = np.fft.fft2(img) * np.conj(np.fft.fft2(ref))
    R = R / np.maximum(np.abs(R), 1e-12) ** whiten
    if smooth_sigma:
        kx = np.fft.fftfreq(ref.shape[0])[:, None]
        ky = np.fft.fftfreq(ref.shape[1])[None, :]
        R = R * np.exp(-2 * (np.pi * smooth_sigma) ** 2 * (kx ** 2 + ky ** 2))
    cc = np.fft.fftshift(np.real(np.fft.ifft2(R)))
    c0 = np.array(cc.shape) // 2
    max_shift = int(max(0, min(max_shift, c0.min() - 1)))  # the window must fit into small images
    win = cc[c0[0] - max_shift:c0[0] + max_shift + 1, c0[1] - max_shift:c0[1] + max_shift + 1]
    ix, iy = np.unravel_index(np.argmax(win), win.shape)
    peak = np.array([ix - max_shift, iy - max_shift], dtype=float)
    quality = float((win.max() - np.median(cc)) / (np.std(cc) + 1e-12))
    up_size = int(np.ceil(upsample * 1.5))
    dftshift = np.trunc(up_size / 2.0)
    ucc = np.real(_upsampled_dft(np.conj(R), up_size, upsample, dftshift - peak * upsample).conj())
    mx = np.array(np.unravel_index(np.argmax(ucc), ucc.shape), dtype=float)
    return peak + (mx - dftshift) / upsample, quality


def estimate_disp_images(ref_comb: np.ndarray, imgs_comb, **kwargs) -> np.ndarray:
    """Displacement (n, 2) of each registration image relative to the reference."""
    return np.array([xcorr_disp(ref_comb, img, **kwargs)[0] for img in imgs_comb]).reshape(-1, 2)


def disp_bins(ch0: np.ndarray, ch1: np.ndarray | None, ref_comb: np.ndarray, npixartifact: int, bin_frames: int,
              excl1: np.ndarray | None = None, **xcorr_kws) -> tuple[np.ndarray, np.ndarray]:
    """Displacement per bin of ``bin_frames`` frames and the bin centers (frames)."""
    b0, centers = bin_stack(ch0, bin_frames)
    b1 = None if ch1 is None else bin_stack(ch1, bin_frames)[0]
    combs = [reg_image(b0[:, :, i], None if b1 is None else b1[:, :, i], npixartifact, excl1=excl1)
             for i in range(b0.shape[2])]
    return estimate_disp_images(ref_comb, combs, **xcorr_kws), centers


def make_reference(ch0: np.ndarray, ch1: np.ndarray | None, npixartifact: int, art_mask: np.ndarray | None = None,
                   init_bin_frames: int = 8, n_iter: int = 3, frac: float = 0.3, shift_bin_frames: int = 4,
                   **xcorr_kws) -> dict:
    """Field reference from the reference recording (cleaned stacks, see ``clean_channels``).

    1. Iterative register-and-average of the most similar bins (robust initial template).
    2. All frames registered (shift bins) and averaged: final reference plus even/odd-frame half means (the halves give
       the reference's reliability for noise-corrected similarities).

    The reference keeps the raw coordinate frame of the recording (median displacement = 0), because ROI masks are
    drawn on the raw recording.

    Returns:
        dict with ``ch0``, ``ch1`` (or None), ``ch0_halves``, ``ch1_halves`` (tuples or None), ``comb`` (registration
        image), ``npixartifact`` and ``art_mask``.
    """
    b0, _ = bin_stack(ch0, init_bin_frames)
    b1 = None if ch1 is None else bin_stack(ch1, init_bin_frames)[0]
    n = b0.shape[2]

    def comb_of(i0, i1):
        return reg_image(i0, i1, npixartifact, excl1=art_mask)

    combs = [comb_of(b0[:, :, i], None if b1 is None else b1[:, :, i]) for i in range(n)]
    ref0 = np.median(b0, axis=2)
    ref1 = None if b1 is None else np.median(b1, axis=2)
    for _ in range(n_iter):
        ref_comb = comb_of(ref0, ref1)
        disp = estimate_disp_images(ref_comb, combs, **xcorr_kws)
        disp -= np.median(disp, axis=0)
        corrs = np.array([pearson(ref_comb.ravel(), shift_image(c, -d, cval=0).ravel()) for c, d in zip(combs, disp)])
        idx = np.argsort(corrs)[::-1][:max(3, int(frac * n))]
        with warnings.catch_warnings():  # pixels shifted out of view in all bins are NaN, filled below
            warnings.simplefilter('ignore', RuntimeWarning)
            ref0 = fill_nan(np.nanmean([shift_image(b0[:, :, i], -disp[i]) for i in idx], axis=0))
            if b1 is not None:
                ref1 = fill_nan(np.nanmean([shift_image(b1[:, :, i], -disp[i]) for i in idx], axis=0))

    disp, centers = disp_bins(ch0, ch1, comb_of(ref0, ref1), npixartifact, shift_bin_frames, excl1=art_mask,
                              **xcorr_kws)
    disp -= np.median(disp, axis=0)
    disp_f = interp_frames(centers, disp, ch0.shape[2])
    m0, a0, c0 = [fill_nan(x) for x in half_means(register_stack(ch0, disp_f))]
    if ch1 is not None:
        m1, a1, c1 = [fill_nan(x) for x in half_means(register_stack(ch1, disp_f))]
    else:
        m1, a1, c1 = None, None, None
    return dict(ch0=m0, ch1=m1, ch0_halves=(a0, c0), ch1_halves=None if m1 is None else (a1, c1),
                comb=comb_of(m0, m1), npixartifact=int(npixartifact), art_mask=art_mask)


def lowpass_disp(disp: np.ndarray, dt: float, tau_s: float | None, breaks=()) -> np.ndarray:
    """Slow displacement: median filter (robust to outliers) followed by a Gaussian (sigma = ``tau_s`` seconds).

    ``breaks``: bin indices where a new segment starts; segments are smoothed separately, so a jump at a break (e.g.
    the focus adjusted before the stimulus started) is corrected as a step instead of being smeared into the other
    segment.
    """
    cuts = sorted({int(b) for b in breaks if 0 < int(b) < len(disp)})
    if cuts:
        return np.concatenate([lowpass_disp(seg, dt, tau_s) for seg in np.split(disp, cuts, axis=0)], axis=0)
    if tau_s is None or tau_s <= 0 or len(disp) < 3:
        return disp.copy()
    k = max(3, int(round(2 * tau_s / dt)) | 1)
    med = ndimage.median_filter(disp, size=(k, 1), mode='nearest')
    return ndimage.gaussian_filter1d(med, sigma=tau_s / dt, axis=0, mode='nearest')


@dataclass
class MotionEstimate:
    """Displacement of a recording relative to the field reference."""
    bin_frames: int
    bin_centers: np.ndarray  # (n_bins,) frame index of the bin centers
    disp_bins: np.ndarray  # (n_bins, 2) raw displacement [px]
    slow_bins: np.ndarray  # (n_bins, 2) smoothed displacement (corrected)
    n_frames: int

    @property
    def fast_bins(self) -> np.ndarray:
        """Residual (n_bins, 2) motion that is not corrected."""
        return self.disp_bins - self.slow_bins

    @property
    def disp_frames(self) -> np.ndarray:
        return interp_frames(self.bin_centers, self.disp_bins, self.n_frames)

    @property
    def slow_frames(self) -> np.ndarray:
        return interp_frames(self.bin_centers, self.slow_bins, self.n_frames)


def union_masks(*masks):
    out = None
    for m in masks:
        if m is not None:
            out = np.array(m, dtype=bool) if out is None else (out | m)
    return out


def estimate_motion(ch0: np.ndarray, ch1: np.ndarray | None, ref: dict, npixartifact: int, fs: float,
                    art_mask: np.ndarray | None = None, bin_frames: int = 4, tau_slow_s: float = 10.,
                    split_s: float | None = None, **xcorr_kws) -> MotionEstimate:
    """Displacement of a (cleaned) recording relative to the field reference, per bin of ``bin_frames`` frames.

    The slow part (``tau_slow_s``) is what gets corrected; the remainder is reported as fast motion. With ``split_s``
    (e.g. the stimulus onset), the bins before and after it (by bin start) are smoothed separately.
    """
    if ref.get('ch1') is None:
        ch1 = None
    excl1 = union_masks(art_mask, ref.get('art_mask'))
    npix = max(int(npixartifact), int(ref['npixartifact']))
    ref_comb = reg_image(ref['ch0'], ref.get('ch1'), npix, excl1=excl1)
    disp, centers = disp_bins(ch0, ch1, ref_comb, npix, bin_frames, excl1=excl1, **xcorr_kws)
    breaks = () if split_s is None else [int(np.sum(centers - (bin_frames - 1) / 2. < np.floor(split_s * fs)))]
    slow = lowpass_disp(disp, bin_frames / fs, tau_slow_s, breaks=breaks)
    return MotionEstimate(bin_frames=int(bin_frames), bin_centers=centers, disp_bins=disp, slow_bins=slow,
                          n_frames=int(ch0.shape[2]))
