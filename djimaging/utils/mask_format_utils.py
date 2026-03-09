import numpy as np


def assert_igor_format(roi_mask: np.ndarray) -> None:
    """Assert that `roi_mask` is in Igor format, raising on violation.

    Igor format requirements: 2-D integer array whose maximum value is 0 or 1
    and whose minimum value is at most 1.

    Parameters
    ----------
    roi_mask : np.ndarray
        ROI mask array to validate.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the mask is not 2-D, contains non-integer values, or has unexpected
        min/max values.
    """
    vmin = np.min(roi_mask)
    vmax = np.max(roi_mask)

    if roi_mask.ndim != 2:
        raise ValueError(f'ROI mask must be 2D, but has shape {roi_mask.shape}')

    if np.any(roi_mask != roi_mask.astype(int)):
        raise ValueError(f'ROI mask must be integer, but has non-integer values')

    if not ((vmax in [0, 1]) and vmin <= 1):
        raise ValueError(f'ROI mask has unexpected values: vmin={vmin}, vmax={vmax}, value={np.unique(roi_mask)}')


def is_igor_format(roi_mask: np.ndarray) -> bool:
    """Check whether `roi_mask` appears to be in Igor format.

    This method can fail if the Igor format is not used consistently.

    Parameters
    ----------
    roi_mask : np.ndarray
        ROI mask array to inspect.

    Returns
    -------
    bool
        True if the mask passes the Igor-format heuristic checks.
    """
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


def as_igor_format(roi_mask: np.ndarray) -> np.ndarray:
    """Return `roi_mask` in Igor format, converting from Python format if necessary.

    Parameters
    ----------
    roi_mask : np.ndarray
        ROI mask in either Igor or Python format.

    Returns
    -------
    np.ndarray
        ROI mask guaranteed to be in Igor format.
    """
    if is_igor_format(roi_mask):
        return roi_mask
    else:
        return to_igor_format(roi_mask)


def to_igor_format(roi_mask: np.ndarray) -> np.ndarray:
    """Convert a Python-format ROI mask to Igor format.

    In Python format background pixels are 0 and ROI pixels are positive
    integers. In Igor format, background pixels are 1 and ROI pixels are
    negative integers (negated Python values with 0 → −1 → 1 mapping).

    Parameters
    ----------
    roi_mask : np.ndarray
        ROI mask in Python format.

    Returns
    -------
    np.ndarray
        ROI mask converted to Igor format.

    Raises
    ------
    ValueError
        If `roi_mask` is not in Python format.
    """
    if not is_python_format(roi_mask):
        raise ValueError(f'ROI mask is not in python format; unique values in mask: {np.unique(roi_mask)}')

    roi_mask = roi_mask.copy()
    roi_mask[roi_mask == 0] = -1
    roi_mask = -roi_mask

    return roi_mask


def is_python_format(roi_mask: np.ndarray) -> bool:
    """Check whether `roi_mask` appears to be in Python format.

    Python format: 2-D integer array with minimum value 0 and non-negative
    values only (background = 0, ROIs = positive integers).

    Parameters
    ----------
    roi_mask : np.ndarray
        ROI mask array to inspect.

    Returns
    -------
    bool
        True if the mask passes the Python-format heuristic checks.
    """
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


def as_python_format(roi_mask: np.ndarray) -> np.ndarray:
    """Return `roi_mask` in Python format, converting from Igor format if necessary.

    Parameters
    ----------
    roi_mask : np.ndarray
        ROI mask in either Igor or Python format.

    Returns
    -------
    np.ndarray
        ROI mask guaranteed to be in Python format.
    """
    if is_python_format(roi_mask):
        return roi_mask
    else:
        return to_python_format(roi_mask)


def to_python_format(roi_mask: np.ndarray) -> np.ndarray:
    """Convert an Igor-format ROI mask to Python format.

    Background pixels (value 1 in Igor format) become 0; ROI pixels (negative
    in Igor format) become positive integers.

    Parameters
    ----------
    roi_mask : np.ndarray
        ROI mask expected to be in Igor format.

    Returns
    -------
    np.ndarray
        ROI mask converted to Python format.

    Raises
    ------
    ValueError
        If `roi_mask` fails the Igor-format assertion (via :func:`assert_igor_format`).
    """
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
