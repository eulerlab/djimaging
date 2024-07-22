import warnings

import numpy as np

from djimaging.utils.mask_utils import assert_igor_format


def extract_roi_idxs(roi_mask, npixartifact=0):
    """Return roi idxs as in ROI mask (i.e. negative values)"""
    assert roi_mask.ndim == 2
    roi_idxs = np.unique(roi_mask[npixartifact:, :])
    roi_idxs = roi_idxs[roi_idxs < 0]  # remove background indexes (0 or 1)
    roi_idxs = roi_idxs[np.argsort(np.abs(roi_idxs))]  # Sort by value
    return roi_idxs.astype(int)


def get_roi2trace(traces, traces_times, roi_ids):
    """Get dict that holds traces and times accessible by roi_id"""
    assert np.all(roi_ids >= 1)

    roi2trace = dict()

    for roi_id in roi_ids:
        idx = roi_id - 1

        if traces.ndim == 3 and idx < traces.shape[-1]:
            trace = traces[:, :, idx]
            trace_times = traces_times[:, :, idx]
            trace_valid = 1
        elif traces.ndim == 2 and idx < traces.shape[-1]:
            trace = traces[:, idx]
            trace_times = traces_times[:, idx]
            trace_valid = 1
        else:
            trace_valid = 0
            trace = np.zeros(0)
            trace_times = np.zeros(0)

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


def get_rel_roi_pos(roi_id, roi_mask, pixel_size_um, ang_deg=0.):
    """Get position relative to plotting axis"""
    # Get relative position in pixel space
    pix_x, pix_y = get_roi_center(roi_mask, roi_id)

    # Get offset to center in um
    dx_um = float((pix_x - roi_mask.shape[0] / 2) * pixel_size_um)
    dy_um = float((pix_y - roi_mask.shape[1] / 2) * pixel_size_um)

    # Rotate around center
    if ang_deg != 0.:
        ang_rad = ang_deg * np.pi / 180.
        dx_um_rot = dx_um * np.cos(ang_rad) - dy_um * np.sin(ang_rad)
        dy_um = dx_um * np.sin(ang_rad) + dy_um * np.cos(ang_rad)
        dx_um = dx_um_rot

    return dx_um, dy_um
