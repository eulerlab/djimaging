import warnings

import numpy as np

from djimaging.utils.mask_format_utils import assert_igor_format


def extract_roi_ids(roi_mask, npixartifact=0):
    """Return roi idxs as in ROI mask (i.e. negative values)"""
    assert roi_mask.ndim == 2
    roi_ids = np.unique(roi_mask[npixartifact:, :])
    roi_ids = roi_ids[roi_ids < 0]  # remove background indexes (0 or 1)
    roi_ids = roi_ids[np.argsort(np.abs(roi_ids))]  # Sort by value
    return roi_ids.astype(int)


def fix_first_or_last_n_nan(trace, n):
    """Replace NaNs if they are at the beginning or end of the trace"""
    if n <= 0:
        return trace
    if np.any(np.isnan(trace[:n])):
        trace[:n] = trace[n]
    if np.any(np.isnan(trace[-n:])):
        trace[-n:] = trace[-n - 1]
    return trace


def get_roi2trace(traces, traces_times, roi_ids_traces, roi_ids_subset=None):
    """Get dict that holds traces and times accessible by roi_id"""
    roi_ids_subset = roi_ids_traces if roi_ids_subset is None else roi_ids_subset

    assert np.all(roi_ids_traces >= 1), f"roi_ids_traces {roi_ids_traces} should be >= 1."
    assert np.all(roi_ids_subset >= 1), f"roi_ids_subset {roi_ids_subset} should be >= 1."

    if traces.shape != traces_times.shape:
        raise ValueError(f"traces shape {traces.shape} does not match traces_times shape {traces_times.shape}.")

    if traces.shape[-1] != len(roi_ids_traces):
        warnings.warn(f"Number of roi_ids {len(roi_ids_traces)} does not match traces shape {traces.shape[-1]}.")

    roi2trace = dict()

    for i, roi_id in enumerate(roi_ids_traces):
        if roi_id not in roi_ids_subset:
            continue

        if traces.ndim == 3 and i < traces.shape[-1]:
            trace = traces[:, :, i]
            trace_times = traces_times[:, :, i]
            trace_valid = 1
        elif traces.ndim == 2 and i < traces.shape[-1]:
            trace = traces[:, i]
            trace_times = traces_times[:, i]
            trace_valid = 1
        else:
            trace_valid = 0
            trace = np.zeros(0)
            trace_times = np.zeros(0)

        trace = fix_first_or_last_n_nan(trace, n=3)
        trace_times = fix_first_or_last_n_nan(trace_times, n=3)

        if np.any(~np.isfinite(trace)) or np.any(~np.isfinite(trace_times)):
            warnings.warn(f'NaN trace or tracetime in for ROI{roi_id}.')
            trace_valid = 0

        roi2trace[roi_id] = dict(trace=trace, trace_times=trace_times, trace_valid=trace_valid)

    return roi2trace


def compare_roi_masks(roi_mask1: np.ndarray, roi_mask2: np.ndarray, max_shift=4) -> str:
    """Test if two roi masks are the same"""
    assert_igor_format(roi_mask1)
    assert_igor_format(roi_mask2)

    if roi_mask1.shape != roi_mask2.shape:
        return 'different'
    if np.all(roi_mask1 == roi_mask2):
        return 'same'

    max_shift_x = np.minimum(max_shift, roi_mask1.shape[0] - 2)
    max_shift_y = np.minimum(max_shift, roi_mask1.shape[1] - 2)

    for dx in range(-max_shift_x, max_shift_x + 1):
        for dy in range(-max_shift_y, max_shift_y + 1):
            shifted1 = roi_mask1[dx:, dy:]
            dx = -roi_mask2.shape[0] if dx == 0 else dx  # Handle zero case
            dy = -roi_mask2.shape[0] if dy == 0 else dy
            shifted2 = roi_mask2[:-dx, :-dy]
            if np.all(shifted1 == shifted2):
                return 'shifted'
    return 'different'


def get_roi_center(roi_mask: np.ndarray, roi_id: int) -> (float, float):
    binary_arr = -roi_mask == roi_id
    if not np.any(binary_arr):
        raise ValueError(f'roi_id={roi_id} not found in roi_mask with values {np.unique(roi_mask)}')
    x, y = np.mean(np.stack(np.where(binary_arr), axis=1), axis=0)
    return x, y


def get_roi_centers(roi_mask: np.ndarray, roi_ids: np.ndarray) -> np.ndarray:
    # TODO test if x, y should be swapped
    roi_centers = np.zeros((len(roi_ids), 2))
    for i, roi_id in enumerate(roi_ids):
        x, y = get_roi_center(roi_mask, roi_id)
        roi_centers[i, :] = (x, y)
    return roi_centers


def get_rel_roi_pos(roi_id, roi_mask, pixel_size_um, pixel_size_d2_um=None, ang_deg=0.):
    """Get position relative to plotting axis"""
    # Get relative position in pixel space
    pix_d1, pix_d2 = get_roi_center(roi_mask, roi_id)

    if pixel_size_d2_um is None:
        pixel_size_d2_um = pixel_size_um

    # Get offset to center in um
    d1_um = float((pix_d1 - roi_mask.shape[0] / 2) * pixel_size_um)
    d2_um = float((pix_d2 - roi_mask.shape[1] / 2) * pixel_size_d2_um)

    # Rotate around center
    if ang_deg != 0.:
        ang_rad = ang_deg * np.pi / 180.
        d1_um_rot = d1_um * np.cos(ang_rad) - d2_um * np.sin(ang_rad)
        d2_um = d1_um * np.sin(ang_rad) + d2_um * np.cos(ang_rad)
        d1_um = d1_um_rot

    return d1_um, d2_um
