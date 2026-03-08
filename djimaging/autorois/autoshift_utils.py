import itertools
from typing import Callable

import numpy as np


def correlation_on_centered(
        x_centered: np.ndarray,
        y_centered: np.ndarray,
        axis: int = 2,
) -> np.ndarray:
    """Compute normalised cross-correlation of two mean-centered arrays along an axis.

    Parameters
    ----------
    x_centered : np.ndarray
        Mean-centered array.
    y_centered : np.ndarray
        Mean-centered array with the same shape as ``x_centered``.
    axis : int, optional
        Axis along which to sum for the dot product. Default is 2.

    Returns
    -------
    np.ndarray
        Array of correlation coefficients with the ``axis`` dimension reduced.
    """
    return np.sum(x_centered * y_centered, axis=axis) / (
            np.sqrt(np.sum(x_centered ** 2, axis=axis)) * np.sqrt(np.sum(y_centered ** 2, axis=axis)))


def compute_corr_map(stack: np.ndarray) -> np.ndarray:
    """Calculate the local neighbouring-pixel correlation map for a 3-D stack.

    Parameters
    ----------
    stack : np.ndarray
        3-D spatiotemporal data with shape (nx, ny, nt). The first two
        dimensions are the spatial axes and the third is the time axis.

    Returns
    -------
    np.ndarray
        2-D array of shape (nx, ny) with the mean local correlation of each
        pixel to its 8 neighbours (horizontal, vertical and diagonal).
    """
    stack_centered = (stack.T - np.mean(stack, axis=2).T).T

    # Compute neighbor correlations
    x_neighbor_corr = correlation_on_centered(stack_centered[1:, :, :], stack_centered[:-1, :, :])
    y_neighbor_corr = correlation_on_centered(stack_centered[:, 1:], stack_centered[:, :-1, :])
    diag_1_neighbor_corr = correlation_on_centered(stack_centered[1:, 1:], stack_centered[:-1, :-1, :])
    diag_2_neighbor_corr = correlation_on_centered(stack_centered[1:, :-1], stack_centered[:-1, 1:, :])

    # Sum up to corr_map
    neighbor_corr = np.zeros(stack_centered.shape[:2])
    neighbor_corr[1:, :] += x_neighbor_corr
    neighbor_corr[:-1, :] += x_neighbor_corr
    neighbor_corr[:, 1:] += y_neighbor_corr
    neighbor_corr[:, :-1] += y_neighbor_corr
    neighbor_corr[1:, 1:] += diag_1_neighbor_corr
    neighbor_corr[:-1, :-1] += diag_1_neighbor_corr
    neighbor_corr[:-1, 1:] += diag_2_neighbor_corr
    neighbor_corr[1:, :-1] += diag_2_neighbor_corr

    # Count neighbor pixels
    neighbor_count = np.zeros(stack_centered.shape[:2])
    neighbor_count[1:, :] += 1
    neighbor_count[:-1, :] += 1
    neighbor_count[:, 1:] += 1
    neighbor_count[:, :-1] += 1
    neighbor_count[1:, 1:] += 1
    neighbor_count[:-1, :-1] += 1
    neighbor_count[:-1, 1:] += 1
    neighbor_count[1:, :-1] += 1

    corr_map = neighbor_corr / neighbor_count

    return corr_map


def compute_corr_map_match_indexes(
        corr_map: np.ndarray,
        ref_corr_map: np.ndarray,
        shift_max: int = 20,
        metric: str = 'mse',
        verbose: bool = False,
        plot: bool = False,
        fun_progress: Callable | None = None,
) -> np.ndarray:
    """Compute a match-score matrix by comparing a correlation map to shifted versions of a reference.

    Parameters
    ----------
    corr_map : np.ndarray
        2-D correlation map to be matched against ``ref_corr_map``.
    ref_corr_map : np.ndarray
        2-D reference correlation map.
    shift_max : int, optional
        Maximum shift in both x and y directions (inclusive). Default is 20.
    metric : str, optional
        Similarity metric. ``'mse'`` uses negative mean squared error
        (higher is better); ``'corr'`` uses Pearson correlation.
        Default is ``'mse'``.
    verbose : bool, optional
        If ``True``, print the detected shift and maximum score. Default is
        ``False``.
    plot : bool, optional
        If ``True``, plot each shift comparison using matplotlib. Default is
        ``False``.
    fun_progress : callable or None, optional
        Optional callback accepting a ``percent`` keyword argument that is
        called at each iteration to report progress. Default is ``None``.

    Returns
    -------
    np.ndarray
        2-D match-score array of shape (2*shift_max+1, 2*shift_max+1). Entry
        ``[i, j]`` contains the score for shift
        ``(i - shift_max, j - shift_max)``.

    Raises
    ------
    NotImplementedError
        If ``metric`` is not ``'corr'`` or ``'mse'``.
    """

    # Initialize the cross-correlation matrix with zeros
    match_indexes = np.zeros((2 * shift_max + 1, 2 * shift_max + 1))

    dxs = np.arange(-shift_max, shift_max + 1)
    dys = np.arange(-shift_max, shift_max + 1)

    dx_dy_pairs = list(itertools.product(dxs, dys))

    if fun_progress is not None:
        fun_progress(percent=0)

    # Iterate over all possible shifts in both x and y directions within the range [-ShiftMax, ShiftMax]
    for i, (dx, dy) in enumerate(dx_dy_pairs, start=1):
        # Crop the shifted_video and stack_corr images to the same size
        cropped = corr_map[max(0, -dx):min(ref_corr_map.shape[0], ref_corr_map.shape[0] - dx),
                  max(0, -dy):min(ref_corr_map.shape[1], ref_corr_map.shape[1] - dy)]

        ref_cropped = ref_corr_map[max(0, dx):min(ref_corr_map.shape[0], ref_corr_map.shape[0] + dx),
                      max(0, dy):min(ref_corr_map.shape[1], ref_corr_map.shape[1] + dy)]

        # Calculate the correlation coefficient between the cropped images
        if metric == 'corr':
            score = np.corrcoef(cropped.flatten(), ref_cropped.flatten())[0, 1]
        elif metric == 'mse':
            score = -np.mean((cropped - ref_cropped) ** 2)
        else:
            raise NotImplementedError(f"Metric {metric} not implemented")

        if plot:
            from matplotlib import pyplot as plt
            fig = plt.figure(figsize=(6, 6))
            fig.suptitle(f"dx={dx}, dy={dy}, {metric}={score:.2f}")

            ax1 = fig.add_subplot(2, 2, 1)
            ax1.imshow(ref_cropped)

            ax2 = fig.add_subplot(2, 2, 2)
            ax2.imshow(cropped)

            plt.show()

        # Store the correlation coefficient in the cross-correlation matrix
        match_indexes[dx + shift_max, dy + shift_max] = score

        if fun_progress is not None:
            fun_progress(percent=100 * i / len(dx_dy_pairs))

    # Calculate image statistics
    v_max = np.max(match_indexes)
    v_max_row_loc, v_max_col_loc = np.unravel_index(np.argmax(match_indexes), match_indexes.shape)

    # Shift the ROI mask
    x_shift = v_max_row_loc - shift_max
    y_shift = v_max_col_loc - shift_max

    if verbose:
        print("ROI mask shifted by", x_shift, "pixels in x and", y_shift, "pixels in y")
        print("Maximum correlation:", v_max)

    return match_indexes


def extract_best_shift(match_indexes: np.ndarray) -> tuple[int, int]:
    """Extract the best (x, y) shift from a match-score matrix.

    Parameters
    ----------
    match_indexes : np.ndarray
        2-D match-score array as returned by
        :func:`compute_corr_map_match_indexes`. The centre entry corresponds
        to zero shift.

    Returns
    -------
    tuple[int, int]
        A tuple ``(shift_x, shift_y)`` with the shift that maximises the
        match score.
    """
    # Calculate image statistics
    rel_x_max, rel_y_max = np.unravel_index(np.argmax(match_indexes), match_indexes.shape)

    # Calculate the expected shift values
    shift_x = ((match_indexes.shape[0] - 1) // 2) - rel_x_max
    shift_y = ((match_indexes.shape[1] - 1) // 2) - rel_y_max

    return shift_x, shift_y


def shift_img(
        img: np.ndarray,
        shift_x: int,
        shift_y: int,
        fun_cval: Callable = np.median,
        cval: float | None = None,
) -> np.ndarray:
    """Shift a 2-D image by ``(shift_x, shift_y)`` pixels and fill the border.

    Parameters
    ----------
    img : np.ndarray
        2-D image to shift.
    shift_x : int
        Number of pixels to shift along axis 0 (positive = down).
    shift_y : int
        Number of pixels to shift along axis 1 (positive = right).
    fun_cval : callable, optional
        Function applied to ``img`` to derive the fill value when ``cval`` is
        ``None``. Default is ``np.median``.
    cval : float or None, optional
        Constant fill value for the border. If ``None``, ``fun_cval`` is used.
        Default is ``None``.

    Returns
    -------
    np.ndarray
        Shifted image with the same shape as ``img``.
    """
    # TODO: merge with mask_utils.shift_image

    shifted_img = np.roll(img, shift=shift_x, axis=0)
    shifted_img = np.roll(shifted_img, shift=shift_y, axis=1)

    if cval is None:
        cval = fun_cval(img)

    if shift_x > 0:
        shifted_img[:shift_x, :] = cval
    elif shift_x < 0:
        shifted_img[shift_x:, :] = cval
    else:
        pass

    if shift_y > 0:
        shifted_img[:, :shift_y] = cval
    elif shift_y < 0:
        shifted_img[:, shift_y:] = cval
    else:
        pass

    return shifted_img
