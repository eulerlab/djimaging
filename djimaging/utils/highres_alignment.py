"""Alignment of a high-resolution (HighRes) scan of a field to the field's ROI mask, and how well the two match.

The HighRes scan is normally the same field at higher pixel resolution (e.g. 512 x 512 instead of 64 x 64, same field of
view), recorded at the end of the stimulus session; it can be slightly shifted in x, y and z. The HR image is resampled
to the grid of the field recording (area averaging), and then compared with the mean image of the presentation the ROI
mask was drawn on (``RoiMask``; the stored mask is aligned to this image):

- **shift**: rigid subpixel displacement of the HR relative to the ROI mask [field px]: the content at mask position p
  is found at p + shift in the resampled HR. Same registration as the motion correction (band-passed functional +
  structural channel, partially whitened cross-correlation, see ``djimaging.utils.motion``).
- **match**: noise-corrected Pearson correlation of the band-passed images after the shift, per channel. 1 = same
  structure (up to noise); it drops with a z offset (or other changes). The HR has few frames, so its own noise is
  corrected with the split-half reliability of its even / odd frames.
- **dz_um_approx**: a rough z offset from the match, using the calibration on one z-stack (docs/motion_qc.md, "z-stack
  check"): order of magnitude only, and without sign.
- **mask_fit**: ``mask_utils.mask_fit_score`` of the ROI mask on the shifted HR and on the mask's own image.

See docs/motion_qc.md ("HighRes alignment") for the validation.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy import ndimage

from djimaging.utils.motion.channel_cleaning import fill_artifact_lines
from djimaging.utils.motion.image_utils import bandpass, reg_image, spearman_brown, valid_region
from djimaging.utils.motion.registration import xcorr_disp
from djimaging.utils.motion.zproxy import dz_from_similarity


def resample_to_field(hr: np.ndarray, hr_px_um: float, field_shape: tuple, field_px_um: float,
                      offset=(0., 0.)) -> np.ndarray:
    """Resample a HR image (x, y) or stack (x, y, t) to the field grid by area averaging; both FOVs centred.

    ``offset`` [field px]: field pixel p is sampled at HR position p + offset, i.e. the result is the HR moved by
    ``-offset`` (sampled at full HR resolution: no extra interpolation blur at field resolution). Field pixels whose
    area is not fully inside the HR field of view are NaN.
    """
    hr = np.asarray(hr, dtype=np.float64)
    f = field_px_um / hr_px_um
    n = max(int(round(f)), 1)
    sm = ndimage.uniform_filter(hr, size=(n, n) + (1,) * (hr.ndim - 2), mode='nearest')
    # an even window at index k covers k - n/2 .. k + n/2 - 1, i.e. it is centred at k - 0.5: sample half a px further
    corr = 0.5 if n % 2 == 0 else 0.
    cx = (np.arange(field_shape[0]) + offset[0] - (field_shape[0] - 1) / 2.) * f + (hr.shape[0] - 1) / 2.
    cy = (np.arange(field_shape[1]) + offset[1] - (field_shape[1] - 1) / 2.) * f + (hr.shape[1] - 1) / 2.
    gx, gy = np.meshgrid(cx, cy, indexing='ij')
    inside = ((gx - f / 2 >= -0.5 - 1e-6) & (gx + f / 2 <= hr.shape[0] - 0.5 + 1e-6)
              & (gy - f / 2 >= -0.5 - 1e-6) & (gy + f / 2 <= hr.shape[1] - 0.5 + 1e-6))

    def sample(img):
        out = ndimage.map_coordinates(img, [gx + corr, gy + corr], order=1, mode='nearest')
        out[~inside] = np.nan
        return out

    if hr.ndim == 2:
        return sample(sm)
    return np.stack([sample(sm[:, :, i]) for i in range(hr.shape[2])], axis=2)


def _halves(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean, even- and odd-frame means. A single image (e.g. a stored median) has no halves: the same array three
    times, which ``_match`` treats as noise-free."""
    stack = np.asarray(stack, dtype=np.float64)
    if stack.ndim == 2 or stack.shape[2] < 2:
        m = stack if stack.ndim == 2 else stack[:, :, 0]
        return m, m, m
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(stack, 2), np.nanmean(stack[:, :, 0::2], 2), np.nanmean(stack[:, :, 1::2], 2)


def _fill(img: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(img), img, np.nanmedian(img))


def _corr(a: np.ndarray, b: np.ndarray, valid: np.ndarray) -> float:
    a, b = a[valid], b[valid]
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


def _match(ref_img, ref_halves, hr_img, hr_halves, valid, min_rel=0.1):
    """Noise-corrected corr of the band-passed images; returns (match, raw corr, HR reliability, ref reliability)."""
    bp = [bandpass(_fill(x)) for x in (ref_img, hr_img) + tuple(ref_halves) + tuple(hr_halves)]
    r_raw = _corr(bp[0], bp[1], valid)
    r_ref = spearman_brown(_corr(bp[2], bp[3], valid)) if ref_halves[0] is not ref_halves[1] else 1.
    r_hr = spearman_brown(_corr(bp[4], bp[5], valid)) if hr_halves[0] is not hr_halves[1] else 1.
    # no usable halves -> no noise correction (note: max(nan, x) would return nan)
    r_ref, r_hr = [float(r) if np.isfinite(r) else 1. for r in (r_ref, r_hr)]
    match = r_raw / np.sqrt(max(r_ref, min_rel) * max(r_hr, min_rel))
    return float(match), float(r_raw), float(r_hr), float(r_ref)


def align_highres(hr_ch0: np.ndarray, hr_ch1: np.ndarray | None, hr_px_um: float, hr_npixartifact: int,
                  ref_ch0: np.ndarray, ref_ch1: np.ndarray | None, field_px_um: float, field_npixartifact: int,
                  roi_mask: np.ndarray | None = None, ref_ch0_halves: tuple | None = None,
                  ref_ch1_halves: tuple | None = None, max_shift: int = 8, whiten: float = 0.5,
                  smooth_sigma: float = 1., upsample: int = 20) -> dict:
    """Shift and match of a HighRes scan relative to the field ROI mask (its presentation's mean image).

    Args:
        hr_ch0, hr_ch1: HR functional / structural channel, (x, y, t) stacks (for the split-half noise correction) or
            (x, y) images. ``hr_ch1`` may be None.
        hr_px_um, hr_npixartifact: HR pixel size and light-artifact lines (in HR pixels).
        ref_ch0, ref_ch1: Mean (or median) images of the ROI mask's presentation (field grid).
        field_px_um, field_npixartifact: Pixel size and artifact lines of the field recording.
        roi_mask: Field ROI mask (Igor format), for ``mask_fit``.
        ref_ch0_halves, ref_ch1_halves: Optional even / odd half means of the reference. Without them the reference is
            taken as noise-free (mean images over hundreds of frames: negligible pixel noise).

    Returns:
        dict with ``shift`` (2,) [field px], ``shift_um`` (2,), ``peak``, per channel ``match_<ch>``, ``corr_<ch>``,
        ``rel_hr_<ch>``, ``rel_ref_<ch>``, ``dz_um_approx`` (from ch1 if available, else ch0), ``n_valid_px``,
        ``mask_fit_hr`` / ``mask_fit_ref`` (if ``roi_mask``), ``scale`` (field px / HR px) and the shifted, resampled
        HR images ``hr_ch0`` / ``hr_ch1`` in the frame of the ROI mask.
    """
    from djimaging.utils.mask_utils import mask_fit_score

    shape = np.asarray(ref_ch0).shape[:2]
    scale = field_px_um / hr_px_um
    npix = max(int(field_npixartifact), int(np.ceil(hr_npixartifact / scale)))
    has1 = hr_ch1 is not None and ref_ch1 is not None

    def prep_hr(x, offset=(0., 0.)):
        x = np.asarray(x, dtype=np.float32)
        x = fill_artifact_lines(x if x.ndim == 3 else x[:, :, None], int(hr_npixartifact))
        return _halves(resample_to_field(x, hr_px_um, shape, field_px_um, offset=offset))

    def prep_ref(x):
        x = np.asarray(x, dtype=np.float32)
        return fill_artifact_lines(x[:, :, None], npix)[:, :, 0]

    hr0 = prep_hr(hr_ch0)
    hr1 = prep_hr(hr_ch1) if has1 else None
    r0 = prep_ref(ref_ch0)
    r1 = prep_ref(ref_ch1) if has1 else None

    hr_reg = reg_image(_fill(hr0[0]), None if not has1 else _fill(hr1[0]), npix)
    disp, peak = xcorr_disp(reg_image(r0, r1, npix), hr_reg, max_shift=max_shift, whiten=whiten,
                            smooth_sigma=smooth_sigma, upsample=upsample)
    disp = np.asarray(disp, dtype=float)

    # HR in the frame of the ROI mask: content at mask position p is at p + disp in the HR -> resample there
    al0 = prep_hr(hr_ch0, offset=disp)
    al1 = prep_hr(hr_ch1, offset=disp) if has1 else None
    valid = valid_region(shape, npix) & np.isfinite(al0[0])
    valid[:2], valid[-2:], valid[:, :2], valid[:, -2:] = False, False, False, False  # band-pass edge effects
    out = dict(shift=disp, shift_um=disp * field_px_um, peak=float(peak), scale=float(scale),
               n_valid_px=int(valid.sum()), npixartifact=npix)
    ref_h0 = tuple(prep_ref(x) for x in ref_ch0_halves) if ref_ch0_halves is not None else (r0, r0)
    out['match_ch0'], out['corr_ch0'], out['rel_hr_ch0'], out['rel_ref_ch0'] = _match(
        r0, ref_h0, al0[0], (al0[1], al0[2]), valid)
    out['hr_ch0'] = al0[0]
    if has1:
        ref_h1 = tuple(prep_ref(x) for x in ref_ch1_halves) if ref_ch1_halves is not None else (r1, r1)
        out['match_ch1'], out['corr_ch1'], out['rel_hr_ch1'], out['rel_ref_ch1'] = _match(
            r1, ref_h1, al1[0], (al1[1], al1[2]), valid)
        out['hr_ch1'] = al1[0]
    else:
        for k in ('match_ch1', 'corr_ch1', 'rel_hr_ch1', 'rel_ref_ch1'):
            out[k] = np.nan
        out['hr_ch1'] = None
    out['dz_um_approx'] = dz_from_similarity(out['match_ch1'], 'ch1') if np.isfinite(out['match_ch1']) \
        else dz_from_similarity(out['match_ch0'], 'ch0')
    if roi_mask is not None:
        out['mask_fit_hr'] = float(mask_fit_score(roi_mask, _fill(out['hr_ch0']), n_artifact=npix))
        out['mask_fit_ref'] = float(mask_fit_score(roi_mask, r0, n_artifact=npix))
    return out
