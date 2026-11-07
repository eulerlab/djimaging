"""Image helpers for motion estimation and motion QC.

Conventions: stacks are (x, y, t) as in ScanM files; the first ``npixartifact`` x-lines are the light artifact.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def bandpass(img: np.ndarray, s_lo: float = 0.75, s_hi: float = 4.0) -> np.ndarray:
    """Spatial band-pass (difference of Gaussians) of a 2D image or of every frame of an (x, y, n) stack."""
    sig_lo = (s_lo, s_lo) + (0,) * (img.ndim - 2)
    sig_hi = (s_hi, s_hi) + (0,) * (img.ndim - 2)
    img = np.asarray(img, dtype=np.float64)
    return ndimage.gaussian_filter(img, sig_lo) - ndimage.gaussian_filter(img, sig_hi)


def valid_region(shape: tuple, npixartifact: int, border: int = 0) -> np.ndarray:
    """Boolean (x, y) mask without the light-artifact lines (and an optional border)."""
    v = np.ones(shape[:2], dtype=bool)
    v[:npixartifact + border, :] = False
    if border > 0:
        v[-border:, :] = False
        v[:, :border] = False
        v[:, -border:] = False
    return v


def taper(shape: tuple, alpha: float = 0.25) -> np.ndarray:
    """2D Tukey window, reduces edge effects in FFT-based registration."""
    from scipy.signal.windows import tukey
    return np.outer(tukey(shape[0], alpha), tukey(shape[1], alpha))


def _norm(img: np.ndarray, valid: np.ndarray) -> np.ndarray:
    x = img - np.mean(img[valid])
    s = np.std(x[valid])
    x = x / (s if s > 0 else 1.)
    x[~valid] = 0.
    return x


def reg_image(img0: np.ndarray, img1: np.ndarray | None, npixartifact: int, excl1: np.ndarray | None = None,
              w0: float = 1., w1: float = 1.) -> np.ndarray:
    """Combined registration image: band-passed, normalized channels, summed and tapered.

    Both channels share the same displacement, so their normalized structure can be summed. Activity in the
    functional channel changes intensities, not positions; the structural channel (e.g. SR101) adds
    activity-independent landmarks. Pixels in ``excl1`` (dilated) are ignored in the structural channel.
    """
    valid = valid_region(img0.shape, npixartifact)
    comb = w0 * _norm(bandpass(img0), valid)
    if img1 is not None:
        valid1 = valid if excl1 is None else valid & ~ndimage.binary_dilation(excl1, iterations=2)
        comb = comb + w1 * _norm(bandpass(img1), valid1)
    return comb * taper(img0.shape)


def bin_stack(stack: np.ndarray, nb: int) -> tuple[np.ndarray, np.ndarray]:
    """Average consecutive frames in bins of ``nb`` frames. Returns the binned stack and the bin centers (frames)."""
    n = stack.shape[2] // nb
    binned = stack[:, :, :n * nb].reshape(stack.shape[0], stack.shape[1], n, nb).mean(axis=3)
    centers = np.arange(n) * nb + (nb - 1) / 2.
    return binned, centers


def bin_edges(n_frames: int, bin_frames: int) -> np.ndarray:
    """Frame indices of the edges of bins of ``bin_frames`` frames; the last bin absorbs the remainder."""
    nb = max(n_frames // bin_frames, 1)
    edges = np.arange(nb + 1) * bin_frames
    edges[-1] = n_frames
    return edges


def interp_frames(centers: np.ndarray, disp: np.ndarray, n_frames: int) -> np.ndarray:
    """Interpolate per-bin displacements (n_bins, 2) to every frame (n_frames, 2)."""
    frames = np.arange(n_frames)
    return np.stack([np.interp(frames, centers, disp[:, k]) for k in range(disp.shape[1])], axis=1)


def shift_image(img: np.ndarray, shift, order: int = 1, cval: float = np.nan) -> np.ndarray:
    return ndimage.shift(img, shift, order=order, mode='constant', cval=cval)


def register_stack(stack: np.ndarray, disp_frames: np.ndarray) -> np.ndarray:
    """Shift every frame by ``-disp`` (bilinear interpolation, edge values repeated).

    ``disp`` is the displacement of the field relative to the reference: content at reference position p is found
    at p + disp. Pixels whose source lies > 0.5 px outside the FOV become NaN. Equivalent to
    ``ndimage.shift(frame, -disp, order=1, mode='nearest')`` per frame, vectorized over frames that share the
    integer part of the shift.
    """
    nx, ny, _ = stack.shape
    d = np.asarray(disp_frames, dtype=np.float64)
    k = np.floor(d).astype(int)
    f = (d - k).astype(np.float32)
    out = np.empty(stack.shape, dtype=np.float32)
    pad = int(np.abs(k).max()) + 2
    for kx, ky in np.unique(k, axis=0):
        idx = np.nonzero((k[:, 0] == kx) & (k[:, 1] == ky))[0]
        sub = np.pad(stack[:, :, idx], ((pad, pad), (pad, pad), (0, 0)), mode='edge')
        x0, y0 = pad + kx, pad + ky
        a = sub[x0:x0 + nx, y0:y0 + ny]
        b = sub[x0 + 1:x0 + 1 + nx, y0:y0 + ny]
        c = sub[x0:x0 + nx, y0 + 1:y0 + 1 + ny]
        e = sub[x0 + 1:x0 + 1 + nx, y0 + 1:y0 + 1 + ny]
        fx, fy = f[idx, 0], f[idx, 1]
        out[:, :, idx] = a * ((1 - fx) * (1 - fy)) + b * (fx * (1 - fy)) + c * ((1 - fx) * fy) + e * (fx * fy)
    px = np.arange(nx)[:, None, None]
    py = np.arange(ny)[None, :, None]
    valid = (px + d[:, 0] >= -0.5) & (px + d[:, 0] <= nx - 0.5) & (py + d[:, 1] >= -0.5) & (py + d[:, 1] <= ny - 0.5)
    out[~valid] = np.nan
    return out


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64) - np.mean(a)
    b = np.asarray(b, dtype=np.float64) - np.mean(b)
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


def fill_nan(img: np.ndarray) -> np.ndarray:
    """Replace NaN pixels by the image median."""
    return np.where(np.isfinite(img), img, np.nanmedian(img))


def bp_fill(img: np.ndarray) -> np.ndarray:
    return bandpass(fill_nan(img))


def bp_fill_stack(stack: np.ndarray) -> np.ndarray:
    """``bp_fill`` for every image of an (x, y, n) stack in one call."""
    med = np.nanmedian(stack, axis=(0, 1))
    return bandpass(np.where(np.isfinite(stack), stack, med[None, None, :]))


def half_means(stack: np.ndarray, i0: int = 0, i1: int | None = None) -> tuple:
    """Mean, even-frame mean and odd-frame mean of ``stack[..., i0:i1]`` (NaN-aware)."""
    s = stack[:, :, i0:i1]
    return np.nanmean(s, axis=2), np.nanmean(s[:, :, 0::2], axis=2), np.nanmean(s[:, :, 1::2], axis=2)


def bin_half_stacks(stack: np.ndarray, edges: np.ndarray) -> tuple:
    """Per-bin mean / even-frame mean / odd-frame mean images as (x, y, n_bins) stacks (NaN-aware)."""
    nb = len(edges) - 1
    shp = stack.shape[:2] + (nb,)
    M, A, B = np.empty(shp, np.float32), np.empty(shp, np.float32), np.empty(shp, np.float32)
    for b in range(nb):
        s = stack[:, :, edges[b]:edges[b + 1]]
        M[:, :, b] = np.nanmean(s, axis=2)
        A[:, :, b] = np.nanmean(s[:, :, 0::2], axis=2)
        B[:, :, b] = np.nanmean(s[:, :, 1::2], axis=2)
    return M, A, B


def spearman_brown(r_half):
    """Reliability of the full mean from the correlation of two half means."""
    r = np.clip(r_half, 1e-3, 0.999)
    return 2 * r / (1 + r)


def masked_pearson(X: np.ndarray, Y: np.ndarray, W: np.ndarray, F: np.ndarray | None = None,
                   min_n: int = 5) -> np.ndarray:
    """Pearson correlation between images ``X[b]`` and ``Y`` (or ``Y[b]``) over the pixels of every mask ``W[i]``.

    Args:
        X: (n_images, n_pixels).
        Y: (n_pixels,) or (n_images, n_pixels).
        W: (n_masks, n_pixels) with 0/1 entries.
        F: (n_images, n_pixels) 0/1 validity per image, default all valid.
        min_n: Minimum number of pixels, NaN otherwise.

    Returns:
        (n_masks, n_images) correlations.
    """
    X = np.asarray(X, np.float64)
    Y = np.broadcast_to(np.asarray(Y, np.float64), X.shape)
    F = np.ones_like(X) if F is None else np.asarray(F, np.float64)
    Wt = np.asarray(W, np.float64).T
    Xf, Yf = X * F, Y * F
    N = F @ Wt
    Sx, Sy = Xf @ Wt, Yf @ Wt
    Sxx, Syy, Sxy = (Xf * X) @ Wt, (Yf * Y) @ Wt, (Xf * Y) @ Wt
    with np.errstate(invalid='ignore', divide='ignore'):
        cov = Sxy - Sx * Sy / N
        vx = Sxx - Sx ** 2 / N
        vy = Syy - Sy ** 2 / N
        r = cov / np.sqrt(vx * vy)
    r[(N < min_n) | ~(vx * vy > 0)] = np.nan
    return r.T


def soft(x, lo: float, hi: float):
    """Linear ramp from 0 (at ``lo``) to 1 (at ``hi``)."""
    return np.clip((np.asarray(x, dtype=float) - lo) / (hi - lo), 0, 1)
