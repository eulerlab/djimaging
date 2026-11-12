"""Refinement of the field ROI mask on the motion-registered reference image.

Igor ROIs are drawn by hand on one (unregistered) presentation. On the registered mean of all presentations of the
reference condition (more signal, no motion blur) the ROI borders can be adjusted to the cell: every ROI keeps its
core (the ROI eroded by ``max_px``) and its border band (``max_px`` inside to ``max_px`` outside the drawn border) is
re-decided by brightness relative to the ROI's local background. ROI ids are kept; ROIs are never added, removed,
merged or split, so ``Roi`` and everything downstream stays valid. Adding missing cells is left to the reviewed
``RoiMask`` correction.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from djimaging.utils.motion.image_utils import fill_nan, valid_region
from djimaging.utils.motion.roi_shifts import roi_ids_from_mask

REFINE_STATUS = ('refined', 'unchanged', 'low_contrast', 'too_small')


def _local_rel(img: np.ndarray, m: np.ndarray, others: np.ndarray) -> tuple[np.ndarray, float]:
    """``rel`` = (img - bg) / (peak - bg) and bg, with bg = 20th percentile of a 1-3 px ring around the ROI (without
    other ROIs) and peak = 90th percentile inside the ROI (as in ``presentation_roi_weights``)."""
    ring = ndimage.binary_dilation(m, iterations=3) & ~ndimage.binary_dilation(m, iterations=1) & ~others
    if ring.sum() < 4:
        ring = ndimage.binary_dilation(m, iterations=3) & ~m
    bg = float(np.percentile(img[ring], 20))
    pk = float(np.percentile(img[m], 90))
    return (img - bg) / max(pk - bg, 1e-6), bg


def refine_roi_mask(img: np.ndarray, roi_mask: np.ndarray, npixartifact: int, max_px: int = 1,
                    rel_thresh: float = 0.5, min_contrast: float = 0.5, min_keep_frac: float = 0.5,
                    grow: bool = True) -> tuple[np.ndarray, dict]:
    """Refine the borders of the ROIs of an Igor-format mask on a (registered) mean image.

    Per ROI:
    1. ``rel`` = (img - bg) / (peak - bg) with bg = 20th percentile of a 1-3 px ring around the ROI (without other
       ROIs) and peak = 90th percentile inside the ROI (as in ``presentation_roi_weights``).
    2. ROIs that are not clearly brighter than their surround are left unchanged: contrast = (median inside - bg) in
       units of the robust standard deviation of the image (offset free; raw ScanM images have a large offset)
       < ``min_contrast``.
    3. The core (ROI eroded by ``max_px``; the brightest pixel if the erosion is empty) is always kept. Pixels of the
       border band (within ``max_px`` of the drawn border, inside and, if ``grow``, outside the ROI) are kept / added
       if ``rel >= rel_thresh``; only parts connected to the core are kept.
    4. Never takes pixels of other ROIs, of the light artifact, or pixels claimed by more than one ROI.
    5. If fewer than ``min_keep_frac`` of the drawn pixels would remain, the ROI is left unchanged.

    Args:
        img: Mean image (x, y), functional channel, in the frame of ``roi_mask``.
        roi_mask: Igor-format mask (ROIs negative, background positive).
        npixartifact: Light-artifact lines.

    Returns:
        Refined Igor-format mask and per-ROI info ``{roi_id (negative): dict(n_px_orig, n_px, n_added, n_removed,
        contrast, status)}``, status in ``REFINE_STATUS``.
    """
    img = fill_nan(np.asarray(img, dtype=float))
    roi_mask = np.asarray(roi_mask)
    roi_ids = roi_ids_from_mask(roi_mask)
    all_rois = roi_mask < 0
    allowed = valid_region(roi_mask.shape, npixartifact)
    struct = ndimage.generate_binary_structure(2, 1)
    vals = img[allowed]
    scale = max(1.4826 * float(np.median(np.abs(vals - np.median(vals)))), 1e-6)

    proposals, info = {}, {}
    for rid in roi_ids:
        m = roi_mask == rid
        n_orig = int(m.sum())
        rel, bg = _local_rel(img, m, all_rois & ~m)
        contrast = (float(np.median(img[m])) - bg) / scale
        info[int(rid)] = dict(n_px_orig=n_orig, n_px=n_orig, n_added=0, n_removed=0, contrast=float(contrast),
                              status='unchanged')
        if not np.isfinite(contrast) or contrast < min_contrast:
            info[int(rid)]['status'] = 'low_contrast'
            continue
        core = ndimage.binary_erosion(m, struct, iterations=max_px)
        if not core.any():
            core = np.zeros_like(m)
            core[np.unravel_index(np.argmax(np.where(m, rel, -np.inf)), m.shape)] = True
        outer = ndimage.binary_dilation(m, struct, iterations=max_px) if grow else m
        band = outer & ~core & allowed & ~(all_rois & ~m)
        keep = core | (band & (rel >= rel_thresh))
        lab, _ = ndimage.label(keep, struct)
        keep = np.isin(lab, np.unique(lab[core & (lab > 0)]))
        if keep.sum() < min_keep_frac * n_orig:
            info[int(rid)]['status'] = 'too_small'
            continue
        proposals[int(rid)] = keep

    # pixels outside the drawn ROIs claimed by more than one ROI go to none of them
    added = [p & ~(roi_mask == rid) for rid, p in proposals.items()]
    contested = np.sum(added, axis=0) > 1 if added else np.zeros(roi_mask.shape, bool)

    refined = np.where(all_rois, 1, roi_mask).astype(roi_mask.dtype)
    refined[refined == 0] = 1
    for rid in roi_ids:
        m = roi_mask == rid
        new = proposals[int(rid)] & ~(contested & ~m) if int(rid) in proposals else m
        refined[new] = rid
        d = info[int(rid)]
        d.update(n_px=int(new.sum()), n_added=int((new & ~m).sum()), n_removed=int((m & ~new).sum()))
        if int(rid) in proposals:
            d['status'] = 'refined' if (d['n_added'] or d['n_removed']) else 'unchanged'
    return refined, info


def pooled_mean_image(mean_imgs: list, n_frames: list | None = None) -> np.ndarray:
    """Frame-weighted mean of registered mean images (NaN pixels, i.e. outside the FOV, ignored)."""
    stack = np.stack([np.asarray(m, float) for m in mean_imgs])
    w = np.ones(len(stack)) if n_frames is None else np.asarray(n_frames, float)
    f = np.isfinite(stack)
    num = np.tensordot(w, np.where(f, stack, 0.), axes=1)
    den = np.tensordot(w, f.astype(float), axes=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 0, num / den, np.nan)
