"""Per-ROI displacements between presentations and the resulting presentation ROI weights.

After global (rigid) registration, ROIs can still be displaced relative to the field reference, e.g. after a break
between conditions (non-rigid deformation, tilt + z change). Per-ROI shifts are measured on each presentation's
registered mean image, pooled over the presentations of a condition, shrunk towards a smooth displacement field
(empirical Bayes), and applied as subpixel-shifted mask weights.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from djimaging.utils.motion.image_utils import bandpass, bp_fill, fill_nan, valid_region


def roi_ids_from_mask(roi_mask: np.ndarray) -> np.ndarray:
    """ROI ids of an Igor-format mask (negative values), sorted by absolute value."""
    ids = np.unique(roi_mask)
    ids = ids[ids < 0]
    return ids[np.argsort(np.abs(ids))].astype(int)


def roi_centroids(roi_mask: np.ndarray, roi_ids: np.ndarray | None = None) -> np.ndarray:
    roi_ids = roi_ids_from_mask(roi_mask) if roi_ids is None else roi_ids
    return np.array([ndimage.center_of_mass(roi_mask == i) for i in roi_ids]).reshape(-1, 2)


def gather_patches(img_stack_pad: np.ndarray, cx, cy, pad: int, h: int) -> np.ndarray:
    """(n_img, X + 2 pad, Y + 2 pad) padded images -> patches (n_img, n, 2h+1, 2h+1) around (cx, cy)."""
    o = np.arange(-h, h + 1)
    rows = (np.asarray(cx)[:, None] + pad + o[None, :])[:, :, None]
    cols = (np.asarray(cy)[:, None] + pad + o[None, :])[:, None, :]
    return img_stack_pad[:, rows, cols]


def local_shifts_batch(ref_patches: np.ndarray, img_patches: np.ndarray, max_shift: int = 2,
                       return_peak: bool = False):
    """Displacement of image patches relative to reference patches: integer search (Pearson correlation) within
    +-max_shift and parabolic subpixel refinement.

    Args:
        ref_patches: (n, h, w).
        img_patches: (..., n, h, w).

    Returns:
        shifts (..., n, 2) and, if ``return_peak``, the peak correlation (..., n).
    """
    m = max_shift
    n, h, w = ref_patches.shape
    r = ref_patches[:, m:h - m, m:w - m].reshape(n, -1)
    r = r - r.mean(axis=-1, keepdims=True)
    rn = np.sqrt((r * r).sum(axis=-1))
    lead = img_patches.shape[:-3]
    k = 2 * m + 1
    cc = np.empty(lead + (n, k, k))
    for ix, dx in enumerate(range(-m, m + 1)):
        for iy, dy in enumerate(range(-m, m + 1)):
            c = img_patches[..., m + dx:h - m + dx, m + dy:w - m + dy].reshape(lead + (n, -1))
            c = c - c.mean(axis=-1, keepdims=True)
            with np.errstate(invalid='ignore', divide='ignore'):
                cc[..., ix, iy] = (c * r).sum(axis=-1) / (np.sqrt((c * c).sum(axis=-1)) * rn)
    cc = np.where(np.isfinite(cc), cc, -np.inf)
    flat = cc.reshape(lead + (n, k * k))
    am = np.argmax(flat, axis=-1)
    ix, iy = am // k, am % k

    def take(dx, dy):
        jx, jy = np.clip(ix + dx, 0, k - 1), np.clip(iy + dy, 0, k - 1)
        return np.take_along_axis(flat, (jx * k + jy)[..., None], axis=-1)[..., 0]

    c0 = take(0, 0)

    def parab(cm, cp, inner):
        d = cm - 2 * c0 + cp
        with np.errstate(invalid='ignore', divide='ignore'):
            v = 0.5 * (cm - cp) / d
        return np.where(inner & (d < 0) & np.isfinite(v), v, 0.)

    sx = parab(take(-1, 0), take(1, 0), (ix > 0) & (ix < k - 1))
    sy = parab(take(0, -1), take(0, 1), (iy > 0) & (iy < k - 1))
    shifts = np.stack([ix - m + sx, iy - m + sy], axis=-1)
    return (shifts, c0) if return_peak else shifts


def estimate_roi_shifts(mean_img: np.ndarray, ref_img: np.ndarray, roi_mask: np.ndarray, patch_half: int = 6,
                        max_shift: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Per-ROI displacement of a presentation's (globally registered) mean image vs. the reference image.

    Local patch cross-correlation of band-passed images (gain invariant, so brightness changes of cells, e.g. by a
    drug, do not create shifts).

    Returns:
        shifts (n_rois, 2) [px] and quality (n_rois,) = peak correlation.
    """
    self_m = bp_fill(mean_img)
    ref_m = bandpass(ref_img)
    c = np.round(roi_centroids(roi_mask)).astype(int)
    pad = patch_half + max_shift + 1
    h = patch_half + max_shift
    rp = gather_patches(np.pad(ref_m, pad)[None], c[:, 0], c[:, 1], pad, h)[0]
    ip = gather_patches(np.pad(self_m, pad)[None], c[:, 0], c[:, 1], pad, h)[0]
    shifts, peak = local_shifts_batch(rp, ip, max_shift=max_shift, return_peak=True)
    return shifts, peak


def _quality_weight(quality, min_quality: float):
    return np.clip((np.asarray(quality, dtype=float) - min_quality) / (1 - min_quality), 0, 1)


def smooth_shift_field(centroids: np.ndarray, shifts: np.ndarray, quality: np.ndarray | None = None,
                       sigma: float = 4., robust_c: float = 0.5, n_iter: int = 3, loo: bool = False,
                       eval_points: np.ndarray | None = None, min_quality: float = 0.3) -> np.ndarray:
    """Smooth displacement field from noisy per-ROI shifts: robust Gaussian kernel regression.

    Weights = kernel(distance; sigma px) * quality weight * robust (Huber-like) weight. ``loo``: leave-one-out
    prediction at every ROI. ``sigma=np.inf`` gives the robust weighted mean shift. Evaluated at the ROI centroids or
    at ``eval_points``.
    """
    centroids = np.asarray(centroids, float)
    shifts = np.asarray(shifts, float)
    n = len(shifts)
    q = np.ones(n) if quality is None else _quality_weight(quality, min_quality)

    def kernel(a, b):
        if not np.isfinite(sigma):
            return np.ones((len(a), len(b)))
        d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
        return np.exp(-0.5 * d2 / sigma ** 2)

    k_roi = kernel(centroids, centroids)
    if loo:
        np.fill_diagonal(k_roi, 0.)
    w_rob = np.ones(n)
    fit = np.zeros_like(shifts)
    for _ in range(n_iter):
        w = k_roi * (q * w_rob)[None, :]
        fit = w @ shifts / np.maximum(w.sum(1, keepdims=True), 1e-12)
        r = np.linalg.norm(shifts - fit, axis=1)
        w_rob = np.where(r <= robust_c, 1., robust_c / np.maximum(r, 1e-12))
    if eval_points is None:
        return fit
    k = kernel(np.asarray(eval_points, float), centroids) * (q * w_rob)[None, :]
    return k @ shifts / np.maximum(k.sum(1, keepdims=True), 1e-12)


def pool_roi_shifts(centroids: np.ndarray, shifts_list: list, quality_list: list, sigma: float = 4.,
                    lam_single: float = 0.5, min_quality: float = 0.3, max_shift: int = 3,
                    inconsistent_px: float = 0.5) -> tuple[np.ndarray, dict]:
    """Between-presentation ROI shifts from the per-ROI shifts of one or several presentations.

    1. Per-ROI shifts of all given presentations are pooled (median).
    2. A robust smooth field (Gaussian kernel regression, ``sigma`` px) is fitted to the pooled shifts.
    3. The per-ROI residual is shrunk towards the field by the empirical-Bayes factor
       ``lam = 1 - noise_var / total_var``, noise from the disagreement between presentations (``lam_single`` for a
       single presentation). Smooth deformations -> small lam; reproducible per-ROI displacements -> large lam.

    Low-quality ROIs get the field value. Validated: pooled per-ROI shifts predict held-out presentations of the
    same condition to ~0.08 px (noise floor), vs. 0.34-0.39 px with global correction only.

    Returns:
        shifts (n_rois, 2) and a dict with ``lam``, ``pooled``, ``field``, ``spread`` (per-ROI disagreement),
        ``n_presentations`` and the flags ``at_limit``, ``inconsistent``, ``low_quality`` (bool arrays).
    """
    s = np.stack([np.asarray(x, float) for x in shifts_list])
    q_all = np.stack([np.asarray(x, float) for x in quality_list])
    k = len(s)
    pooled = np.median(s, axis=0) if k > 1 else s[0]
    q = q_all.mean(axis=0)
    fld = smooth_shift_field(centroids, pooled, q, sigma=sigma, min_quality=min_quality)
    resid = pooled - fld
    if k > 1:
        noise_var = np.mean(np.var(s, axis=0, ddof=1).sum(axis=1)) / k
        lam = float(np.clip(1 - noise_var / max(np.mean((resid ** 2).sum(axis=1)), 1e-12), 0, 1))
    else:
        lam = float(lam_single)
    final = fld + lam * resid
    low_q = q < min_quality
    final[low_q] = fld[low_q]
    spread = np.sqrt(np.var(s, axis=0).sum(axis=1)) if k > 1 else np.zeros(len(pooled))
    flags = dict(at_limit=np.any(np.abs(s) >= max_shift - 0.25, axis=(0, 2)), inconsistent=spread > inconsistent_px,
                 low_quality=low_q)
    return final, dict(lam=lam, pooled=pooled, field=fld, spread=spread, n_presentations=k, **flags)


def shifted_roi_weights(roi_mask: np.ndarray, roi_shifts: np.ndarray, roi_ids: np.ndarray | None = None) -> np.ndarray:
    """Subpixel-shifted ROI masks as weights (n_rois, x, y): each binary mask moved by its shift (bilinear)."""
    roi_ids = roi_ids_from_mask(roi_mask) if roi_ids is None else roi_ids
    return np.stack([ndimage.shift((roi_mask == i).astype(float), s, order=1, mode='constant')
                     for i, s in zip(roi_ids, roi_shifts)])


def _rel_and_contrast(img, m, others):
    ring = ndimage.binary_dilation(m, iterations=3) & ~ndimage.binary_dilation(m, iterations=1) & ~others
    if ring.sum() < 4:
        ring = ndimage.binary_dilation(m, iterations=3) & ~m
    bg = np.nanpercentile(img[ring], 20)
    pk = np.nanpercentile(img[m], 90)
    rel = (img - bg) / max(pk - bg, 1e-6)
    contrast = (np.nanmedian(img[m]) - bg) / max(abs(bg), 1e-6)
    return rel, contrast


def presentation_roi_weights(mean_img: np.ndarray, valid_frac: np.ndarray, ref_img: np.ndarray,
                             field_mask: np.ndarray, roi_shifts: np.ndarray, npixartifact: int,
                             rel_thresh: float = 0.2, rel_drop: float = 0.3, min_valid_frac: float = 0.95,
                             min_contrast_frac: float = 0.5) -> tuple[np.ndarray, dict]:
    """Presentation ROI weights in the frame of the (slowly) registered stack.

    1. Each ROI is moved by its between-presentation shift (subpixel -> bilinear weights).
    2. Mild reshape, never adds pixels:
       - pixels outside the FOV in more than ``1 - min_valid_frac`` of the frames are dropped;
       - *border* pixels that were part of the cell in the reference (rel >= rel_thresh) but not anymore in the
         presentation's mean image (rel < rel_thresh and lost >= rel_drop) are dropped, only if the ROI as a whole kept
         >= min_contrast_frac of its reference contrast; otherwise the ROI is flagged ``low_contrast``.

    Args:
        mean_img: Mean image of the registered presentation (functional channel).
        valid_frac: Fraction of frames in which each pixel is inside the FOV after registration.
        ref_img: Reference image (functional channel).
        field_mask: Igor-format field mask.
        roi_shifts: (n_rois, 2) shifts in the order of ``roi_ids_from_mask(field_mask)``.
        npixartifact: Light-artifact lines.

    Returns:
        weights (n_rois, x, y) and info: per ROI dict with ``n_px_ref``, ``n_px``, ``low_contrast``.
    """
    roi_ids = roi_ids_from_mask(field_mask)
    roi_shifts = np.asarray(roi_shifts, float)
    w = shifted_roi_weights(field_mask, roi_shifts, roi_ids)
    mean_img = fill_nan(mean_img)
    valid0 = valid_region(field_mask.shape, npixartifact)
    b = w >= 0.5
    all_new, all_ref = b.any(axis=0), field_mask < 0
    info = {}
    for i, rid in enumerate(roi_ids):
        m_ref = field_mask == rid
        m = b[i]
        if m.sum() == 0:
            info[int(rid)] = dict(n_px_ref=int(m_ref.sum()), n_px=0, low_contrast=True)
            continue
        rel_new, con_new = _rel_and_contrast(mean_img, m, all_new & ~m)
        rel_ref, con_ref = _rel_and_contrast(ref_img, m_ref, all_ref & ~m_ref)
        rel_ref = ndimage.shift(rel_ref, roi_shifts[i], order=1, mode='nearest')
        drop = m & ((valid_frac < min_valid_frac) | ~valid0)
        low_contrast = bool(con_new < min_contrast_frac * con_ref)
        if not low_contrast:
            border = m & ~ndimage.binary_erosion(m)
            drop |= border & (rel_ref >= rel_thresh) & (rel_new < rel_thresh) & (rel_ref - rel_new >= rel_drop)
        w[i][drop] = 0.
        info[int(rid)] = dict(n_px_ref=int(m_ref.sum()), n_px=int((m & ~drop).sum()), low_contrast=low_contrast)
    return w, info


def weights_to_label(weights: np.ndarray, roi_ids: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Igor-format label mask (ROIs negative, background 1) from ROI weights."""
    label = np.ones(weights.shape[1:], dtype=int)
    wmax = weights.max(axis=0)
    sel = wmax >= threshold
    label[sel] = np.asarray(roi_ids)[np.argmax(weights, axis=0)][sel]
    return label


def crop_weights(w: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Crop a 2D weight image to its non-zero bounding box: (cropped, x0, y0)."""
    nz = np.nonzero(w > 0)
    if len(nz[0]) == 0:
        return np.zeros((0, 0), np.float32), 0, 0
    x0, x1, y0, y1 = nz[0].min(), nz[0].max() + 1, nz[1].min(), nz[1].max() + 1
    return w[x0:x1, y0:y1].astype(np.float32), int(x0), int(y0)


def uncrop_weights(cropped: np.ndarray, x0: int, y0: int, shape: tuple) -> np.ndarray:
    out = np.zeros(shape, np.float32)
    out[x0:x0 + cropped.shape[0], y0:y0 + cropped.shape[1]] = cropped
    return out


def extract_traces_weighted(stack: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted-mean traces (t, n) from ``stack`` (x, y, t) with ``weights`` (n, x, y); NaN pixels are ignored."""
    x = stack.reshape(-1, stack.shape[2])
    f = np.isfinite(x)
    wf = weights.reshape(weights.shape[0], -1)
    return ((wf @ np.where(f, x, 0.)) / np.maximum(wf @ f, 1e-9)).T
