import numpy as np


def assert_igor_format(roi_mask):
    vmin = np.min(roi_mask)
    vmax = np.max(roi_mask)

    if roi_mask.ndim != 2:
        raise ValueError(f'ROI mask must be 2D, but has shape {roi_mask.shape}')

    if np.any(roi_mask != roi_mask.astype(int)):
        raise ValueError(f'ROI mask must be integer, but has non-integer values')

    if not ((vmax in [0, 1]) and vmin <= 1):
        raise ValueError(f'ROI mask has unexpected values: vmin={vmin}, vmax={vmax}, value={np.unique(roi_mask)}')


def is_igor_format(roi_mask):
    """This method can fail if the igor format is not used consistently."""
    vmin = np.min(roi_mask)
    vmax = np.max(roi_mask)

    if roi_mask.ndim != 2:
        return False

    if np.any(roi_mask != roi_mask.astype(int)):
        return False

    if vmax in [0, 1] and vmin <= 1:
        return True
    else:
        return False


def as_igor_format(roi_mask):
    """Convert from python format to igor format"""
    if is_igor_format(roi_mask):
        return roi_mask
    else:
        return to_igor_format(roi_mask)


def to_igor_format(roi_mask):
    if not is_python_format(roi_mask):
        raise ValueError(f'ROI mask is not in python format; unique values in mask: {np.unique(roi_mask)}')

    roi_mask = roi_mask.copy()
    roi_mask[roi_mask == 0] = -1
    roi_mask = -roi_mask

    return roi_mask


def is_python_format(roi_mask):
    vmin = np.min(roi_mask)
    vmax = np.max(roi_mask)

    if roi_mask.ndim != 2:
        return False

    if np.any(roi_mask != roi_mask.astype(int)):
        return False

    if vmin == 0 and vmax >= 0:
        return True
    else:
        return False


def as_python_format(roi_mask):
    """Convert from igor format to python format if necessary"""
    if is_python_format(roi_mask):
        return roi_mask
    else:
        return to_python_format(roi_mask)


def to_python_format(roi_mask):
    # Some ROI masks have 11 instead of 1's for some reason
    rm_vals = np.unique(roi_mask)
    if set(rm_vals[rm_vals > 0]) == {1, 11}:
        roi_mask = roi_mask.copy()
        roi_mask[roi_mask == 11] = 1

    assert_igor_format(roi_mask)

    vmax = np.max(roi_mask)

    roi_mask = roi_mask.copy()
    roi_mask[roi_mask == vmax] = 0
    roi_mask = np.abs(roi_mask)

    return roi_mask
