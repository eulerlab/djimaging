"""Registration of a functional ROI mask onto the HighRes image of the same field.

djimaging has no stored transform between ``HighRes`` and ``Presentation``. The ROI canvas (``InteractiveRoiCanvas``)
simply resizes the HighRes average onto the functional pixel grid, i.e. it assumes both scans cover the same field of
view with the same centre. ``pixel_size_um`` is derived from the setup's x-scale, the zoom and the number of *active*
pixels per line (``setup_utils.get_pixel_size_xy_um``), so this assumption is a convention, not a measurement.

On the local recordings (2026-09-16, setup 1) the assumption does not hold: the HighRes image matches the functional
mean image only after a shift of -5 functional pixels (~8.75 um) along x, the fast scan axis, while scale (n_hr / n_func)
and y agree. This module therefore

1. checks the scan metadata (both xy scans, same field of view and centre within tolerance, integer pixel ratio), and
2. estimates the remaining translation from the images themselves (normalised cross-correlation of the functional
   mean image with the block-averaged HighRes image, in steps of one HighRes pixel), and only accepts it if the
   correlation is clearly better than chance and the optimum is not at the edge of the search range.

The mask is then mapped with integer operations only (``np.kron`` upsampling + integer shift), so labels and ROI IDs are
preserved exactly. A residual scale mismatch (fitted ~1.00-1.02 on the local data) is not corrected; it amounts to at
most ~0.6 functional pixels at the field border and is reported in ``Registration.reason``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np

from djimaging.cellviz.data import HighResImage, PresentationData, RoiLabels

FOV_RTOL = 0.01       # relative tolerance for equal field of view
CENTER_ATOL_PX = 1.0  # tolerance for equal scan centre, in functional pixels
MIN_NCC = 0.4         # minimum correlation of the registered images
MIN_NCC_GAIN = 0.05   # a non-zero shift must improve the correlation by at least this much
MAX_SHIFT_FUNC_PX = 8


@dataclass(frozen=True)
class Registration:
    """Functional pixel ``(y, x)`` covers HighRes pixels ``[y*scale + dy, ...) x [x*scale + dx, ...)``."""
    ok: bool
    reason: str
    scale: int = 0
    shift_hr_px: tuple = (0, 0)  # (dy, dx) in HighRes pixels
    ncc_identity: float = np.nan
    ncc_best: float = np.nan
    method: str = ''

    @property
    def shift_func_px(self) -> tuple:
        return tuple(s / self.scale for s in self.shift_hr_px) if self.scale else (np.nan, np.nan)


def check_metadata(func_meta: Mapping, func_shape: tuple, func_pixel_size_um: float,
                   hr_meta: Mapping, hr_shape: tuple, hr_pixel_size_um: float) -> tuple[bool, str, int]:
    """Return (ok, reason, integer scale) from the scan geometry. Shapes are display shapes ``(ny, nx)``."""
    for name, meta in (('functional', func_meta), ('HighRes', hr_meta)):
        if meta.get('scan_type', 'xy') != 'xy':
            return False, f"{name} recording is not an xy scan ({meta.get('scan_type')})", 0
    if func_meta.get('angle_deg') is not None and hr_meta.get('angle_deg') is not None \
            and not np.isclose(func_meta['angle_deg'], hr_meta['angle_deg']):
        return False, f"scan angles differ ({func_meta['angle_deg']} vs {hr_meta['angle_deg']} deg)", 0
    if not (np.isfinite(func_pixel_size_um) and np.isfinite(hr_pixel_size_um)) or func_pixel_size_um <= 0:
        return False, "pixel sizes are unknown", 0

    fov_func = np.array(func_shape) * func_pixel_size_um
    fov_hr = np.array(hr_shape) * hr_pixel_size_um
    if not np.allclose(fov_func, fov_hr, rtol=FOV_RTOL):
        return False, f"fields of view differ: functional {fov_func} um vs HighRes {fov_hr} um", 0

    ratios = np.array(hr_shape) / np.array(func_shape)
    if not (np.allclose(ratios, ratios[0]) and np.isclose(ratios[0], round(ratios[0])) and ratios[0] >= 1):
        return False, f"HighRes / functional pixel ratio {ratios} is not one integer", 0
    scale = int(round(ratios[0]))

    for k in ('absx', 'absy'):
        a, b = func_meta.get(k), hr_meta.get(k)
        if a is None or b is None:
            return False, f"scan position {k} is unknown", 0
        if abs(a - b) > CENTER_ATOL_PX * func_pixel_size_um:
            return False, f"scan centres differ in {k}: {a} vs {b} um", 0
    return True, f"same field of view ({fov_func[0]:.1f} um) and centre, {scale}x pixels", scale


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return np.nan
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else np.nan


def _block_mean_shifted(hr: np.ndarray, scale: int, shape: tuple, dy: int, dx: int) -> np.ndarray:
    """HighRes block means under each functional pixel, for the shift (dy, dx) in HighRes pixels; NaN outside."""
    ny, nx = shape
    pad = MAX_SHIFT_FUNC_PX * scale + scale
    padded = np.pad(hr.astype(float), pad, constant_values=np.nan)
    win = padded[pad + dy: pad + dy + ny * scale, pad + dx: pad + dx + nx * scale]
    return win.reshape(ny, scale, nx, scale).mean(axis=(1, 3))  # NaN wherever a block leaves the image


def estimate_shift(func_image: np.ndarray, hr_image: np.ndarray, scale: int, npixartifact: int = 0,
                   max_shift_func_px: int = MAX_SHIFT_FUNC_PX) -> tuple[tuple, float, float, bool]:
    """Coarse (functional pixel) then fine (HighRes pixel) search of the translation.

    Returns ((dy, dx) in HighRes pixels, ncc at identity, best ncc, optimum at the edge of the search range).
    ``npixartifact`` columns (x < npixartifact, the light-artifact lines) are ignored.
    """
    func = np.asarray(func_image, dtype=float).copy()
    func[:, :npixartifact] = np.nan
    shape = func.shape

    def score(dy, dx):
        return _ncc(func, _block_mean_shifted(hr_image, scale, shape, dy, dx))

    ncc_identity = score(0, 0)
    coarse = [(score(dy * scale, dx * scale), dy * scale, dx * scale)
              for dy in range(-max_shift_func_px, max_shift_func_px + 1)
              for dx in range(-max_shift_func_px, max_shift_func_px + 1)]
    _, cy, cx = max(coarse, key=lambda c: -np.inf if np.isnan(c[0]) else c[0])
    fine = [(score(cy + dy, cx + dx), cy + dy, cx + dx)
            for dy in range(-scale, scale + 1) for dx in range(-scale, scale + 1)]
    best, by, bx = max(fine, key=lambda c: -np.inf if np.isnan(c[0]) else c[0])
    limit = max_shift_func_px * scale
    at_edge = abs(by) >= limit or abs(bx) >= limit
    return (int(by), int(bx)), ncc_identity, best, at_edge


def register_highres(pres: PresentationData, hr: HighResImage, func_image: Optional[np.ndarray] = None,
                     shift_hr_px: Optional[tuple] = None) -> Registration:
    """Register ``hr`` to the presentation's coordinate system.

    ``func_image`` defaults to the temporal median of the movie. ``shift_hr_px`` = (dy, dx) skips the image-based
    estimate and uses a known transform (metadata must still agree), e.g. one verified on the server.
    """
    ok, reason, scale = check_metadata(pres.meta, pres.frames.shape[1:], pres.pixel_size_um,
                                       hr.meta, hr.image.shape, hr.pixel_size_um)
    if not ok:
        return Registration(False, f"HighRes overlay unavailable: {reason}")

    if shift_hr_px is not None:
        return Registration(True, f"{reason}; user-supplied shift {tuple(shift_hr_px)} HighRes px", scale,
                            tuple(int(s) for s in shift_hr_px), method='given')

    if func_image is None:
        func_image = np.median(pres.frames[::max(1, pres.n_frames // 300)], axis=0)
    (dy, dx), ncc0, ncc, at_edge = estimate_shift(func_image, hr.image, scale, pres.npixartifact)
    shift_text = f"shift dy={dy / scale:+.2f}, dx={dx / scale:+.2f} functional px"
    if at_edge:
        return Registration(False, f"HighRes overlay unavailable: best match at the edge of the search range "
                                   f"({shift_text}); registration not verified", scale, (dy, dx), ncc0, ncc, 'image')
    if not np.isfinite(ncc) or ncc < MIN_NCC:
        return Registration(False, f"HighRes overlay unavailable: images do not correlate (ncc={ncc:.2f} < {MIN_NCC}); "
                                   f"registration not verified", scale, (dy, dx), ncc0, ncc, 'image')
    if (dy, dx) != (0, 0) and ncc - ncc0 < MIN_NCC_GAIN:
        return Registration(False, f"HighRes overlay unavailable: shift is ambiguous (ncc {ncc0:.2f} -> {ncc:.2f})",
                            scale, (dy, dx), ncc0, ncc, 'image')
    return Registration(True, f"{reason}; image-based {shift_text} (ncc {ncc0:.2f} at identity -> {ncc:.2f}); "
                              f"scale not refined (<= ~0.6 px error at the border)",
                        scale, (dy, dx), ncc0, ncc, 'image')


def map_labels_to_highres(rois: RoiLabels, registration: Registration, hr_shape: tuple) -> RoiLabels:
    """Nearest-neighbour upsampling + integer shift of the label image. Labels and ROI IDs are preserved exactly.

    ROIs that fall completely outside the HighRes image are dropped from the mapping (see ``rois_outside``).
    """
    if not registration.ok:
        raise ValueError(registration.reason)
    s = registration.scale
    dy, dx = registration.shift_hr_px
    bg = rois.background_labels[0] if rois.background_labels else 0
    up = np.kron(rois.labels, np.ones((s, s), dtype=rois.labels.dtype))
    out = np.full(hr_shape, bg, dtype=rois.labels.dtype)
    # functional pixel (y, x) -> HighRes rows y*s + dy: destination index = source index - (dy, dx)
    ys, yd = _overlap(up.shape[0], hr_shape[0], -dy)
    xs, xd = _overlap(up.shape[1], hr_shape[1], -dx)
    out[yd, xd] = up[ys, xs]
    present = set(np.unique(out).tolist())
    mapping = {lab: rid for lab, rid in rois.label_to_roi_id.items() if lab in present}
    background = tuple(b for b in rois.background_labels if b in present) or (bg,)
    return RoiLabels(out, mapping, background)


def _overlap(n_src: int, n_dst: int, offset: int) -> tuple[slice, slice]:
    """Slices so that ``dst[i] = src[i + offset]`` where both are in range."""
    d0 = max(0, -offset)
    d1 = min(n_dst, n_src - offset)
    if d1 <= d0:
        return slice(0, 0), slice(0, 0)
    return slice(d0 + offset, d1 + offset), slice(d0, d1)


def rois_outside(rois: RoiLabels, hr_rois: RoiLabels) -> list:
    """ROI IDs that are (partly) cut off by the registration shift."""
    s_area = {r: int(rois.pixels(r).sum()) for r in rois.roi_ids}
    scale2 = hr_rois.shape[0] * hr_rois.shape[1] / (rois.shape[0] * rois.shape[1])
    out = []
    for r in rois.roi_ids:
        n_hr = int(hr_rois.pixels(r).sum()) if int(r) in hr_rois.label_to_roi_id.values() else 0
        if n_hr < s_area[r] * scale2:
            out.append(int(r))
    return out
