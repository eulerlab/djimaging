import itertools

import numpy as np


def compute_corr_map(stack, fun_progress=None):
    """
    Calculate the local neighboring pixel correlation matrix for a 3D stack (x, y, t).

    Input:
    stack (numpy.ndarray): 3D spatiotemporal data, the first two dimensions (nx, ny) correspond to spatial dimensions,
    and the third dimension (_) represents the time series.

    Output:
    corr_map (numpy.ndarray): 2D array with normalized cross-correlations for each pixel in the spatial domain.
    The values represent the local similarity of each pixel to its neighbors within predefined shifts.
    """
    nx, ny, _ = stack.shape
    corr_map = np.zeros((nx, ny))
    counts = np.zeros((nx, ny))

    ixs = np.arange(nx)
    iys = np.arange(ny)

    ix_iy_pairs = list(itertools.product(ixs, iys))

    if fun_progress is not None:
        fun_progress(percent=0)

    for i, (ix, iy) in enumerate(ix_iy_pairs, start=1):
        for idx, idy in [(0, 1), (1, 0), (1, 1), (0, -1), (-1, 0), (-1, -1), (-1, 1), (1, -1)]:
            if ix + idx >= nx or iy + idy >= ny or ix + idx < 0 or iy + idy < 0:
                continue
            else:
                counts[ix, iy] += 1
                corr_map[ix, iy] += np.corrcoef(stack[ix + idx, iy + idy, :], stack[ix, iy, :])[0, 1]

        if fun_progress is not None:
            fun_progress(percent=100 * i / len(ix_iy_pairs))

    # Normalize by the number of neighbors
    corr_map /= counts

    return corr_map


def compute_corr_map_match_indexes(corr_map, ref_corr_map, shift_max=20, metric='mse', verbose=False,
                                   plot=False, fun_progress=None):
    """Calculate the match_indexes between a stack_corr image and its shifted versions.

    Input:
    shifted_video (numpy.ndarray): input video as 3D numpy array (frames, height, width).
    stack_corr (numpy.ndarray): 2D reference image for cross-correlation.
    shift_max (int): maximum shift value in both x and y directions.

    Output:
    corr_map (array): cross-correlation matrix with shape (2*shift_max+1, 2*shift_max+1).
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


def extract_best_shift(match_indexes):
    # Calculate image statistics
    rel_x_max, rel_y_max = np.unravel_index(np.argmax(match_indexes), match_indexes.shape)

    # Calculate the expected shift values
    shift_x = ((match_indexes.shape[0] - 1) // 2) - rel_x_max
    shift_y = ((match_indexes.shape[1] - 1) // 2) - rel_y_max

    return shift_x, shift_y


def shift_img(img, shift_x, shift_y, fun_cval=np.median):
    # TODO: merge with mask_utils.shift_image

    shifted_img = np.roll(img, shift=shift_x, axis=0)
    shifted_img = np.roll(shifted_img, shift=shift_y, axis=1)

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
