import os
import pickle
import queue
import warnings

import numpy as np
from matplotlib import pyplot as plt

from djimaging.utils import scanm_utils
from djimaging.utils.alias_utils import check_shared_alias_str


def create_circular_mask(h, w, center, radius):
    """Create a binary mask"""
    xs, ys = np.ogrid[:w, :h]
    dist_from_center = np.sqrt((xs - center[0]) ** 2 + (ys - center[1]) ** 2)
    mask = np.asarray(dist_from_center <= radius)
    return mask


def extract_connected_mask(mask, i, j):
    """https://stackoverflow.com/questions/35224094/extract-connected-pixels-in-a-binary-image-using-python"""

    mask = mask.copy()

    nx, ny = mask.shape

    x = []
    y = []
    q = queue.Queue()

    if mask[i][j] == 1:
        q.put((i, j))

    while not q.empty():
        u, v = q.get()
        x.append(u)
        y.append(v)

        dxs = [0, 0, 1, -1]
        dys = [1, -1, 0, 0]

        for dx, dy in zip(dxs, dys):
            xx = u + dx
            yy = v + dy
            if xx < nx and yy < ny and mask[xx][yy] == 1:
                mask[xx][yy] = 0
                q.put((xx, yy))

    mask = np.zeros_like(mask)
    mask[x, y] = 1

    return mask


def get_mask_by_cc(seed_ix, seed_iy, data, seed_trace=None, thresh=0.2, max_pixel_dist=10, plot=False):
    nx = data.shape[0]
    ny = data.shape[1]

    if seed_trace is None:
        seed_trace = data[seed_ix, seed_iy]

    ccs = np.zeros((nx, ny), dtype='float')

    dist_mask = create_circular_mask(w=nx, h=ny, center=(seed_ix, seed_iy), radius=max_pixel_dist)

    for ix in np.arange(0, nx):
        for iy in np.arange(0, ny):
            if dist_mask[ix, iy]:
                ccs[ix, iy] = np.corrcoef(seed_trace, data[ix, iy])[0, 1]

    mask = (ccs >= thresh) & dist_mask

    mask_connected = extract_connected_mask(mask, seed_ix, seed_iy)

    if plot and np.any(mask):
        fig, axs = plt.subplots(1, 6, figsize=(20, 3))

        ax = axs[0]
        ax.plot(seed_trace)

        ax = axs[1]
        ax.set_title('data')
        ax.imshow(data.mean(axis=2))

        ax = axs[2]
        ax.set_title('dist mask')
        ax.imshow(dist_mask)

        ax = axs[3]
        ax.set_title('mask')
        ax.imshow(mask)

        ax = axs[4]
        ax.set_title('mask_connected')
        ax.imshow(mask_connected)

        ax = axs[5]
        ax.imshow(data.mean(axis=2), cmap='Greys')
        plot_mask = mask_connected.copy().astype(float)
        plot_mask[~mask] = np.nan
        ax.imshow(plot_mask, cmap='Reds', vmin=0, vmax=1, alpha=0.5, zorder=100)

        plt.show()

    return mask_connected


def get_mask_by_bg(seed_ix, seed_iy, data, ref_value=None, thresh=0.2, max_pixel_dist=10, plot=False):
    """Get mask based on background image pixel values"""
    assert data.ndim == 2

    nx = data.shape[0]
    ny = data.shape[1]

    if ref_value is None:
        ref_value = data[seed_ix, seed_iy]

    value_dists = np.zeros((nx, ny), dtype='float')

    dist_mask = create_circular_mask(w=nx, h=ny, center=(seed_ix, seed_iy), radius=max_pixel_dist)

    for ix in np.arange(0, nx):
        for iy in np.arange(0, ny):
            if dist_mask[ix, iy]:
                value_dists[ix, iy] = np.abs(data[ix, iy] - ref_value)

    mask = (value_dists <= thresh) & dist_mask

    mask_connected = extract_connected_mask(mask, seed_ix, seed_iy)

    if plot and np.any(mask):
        fig, axs = plt.subplots(1, 5, figsize=(20, 3))

        ax = axs[0]
        ax.set_title('data')
        ax.imshow(data)

        ax = axs[1]
        ax.set_title('dist mask')
        ax.imshow(dist_mask)

        ax = axs[2]
        ax.set_title('mask')
        ax.imshow(mask)

        ax = axs[3]
        ax.set_title('mask_connected')
        ax.imshow(mask_connected)

        ax = axs[4]
        ax.imshow(data, cmap='Greys')
        plot_mask = mask_connected.copy().astype(float)
        plot_mask[~mask] = np.nan
        ax.imshow(plot_mask, cmap='Reds', vmin=0, vmax=1, alpha=0.5, zorder=100)

        plt.show()

    return mask_connected


def relabel_mask(mask, connectivity, return_num=False):
    from skimage.measure import label

    if return_num:
        new_mask, num = label(mask.T, connectivity=connectivity, return_num=True)
        return new_mask.T, num
    else:
        return label(mask.T, connectivity=connectivity).T


def fix_disconnected_rois(mask, connectivity=2, verbose=True, return_num=False):
    """
    re-labels ROI mask using `skimage.measure.label()`:
    all connected pixels with the same value get the same label

    connectivity=1: only NSEW neighbors count as connections
    connectivity=2: NSEW neighbors and diagonal neighbors count as connections
    """

    if return_num:
        mask_new, n_comps = relabel_mask(mask, connectivity=connectivity, return_num=return_num)
    else:
        n_comps = None
        mask_new = relabel_mask(mask, connectivity=connectivity)

    if verbose:
        roi_ids_orig = np.unique(mask)
        for roi_id in roi_ids_orig:
            roi_orig_idx = mask == roi_id
            roi_parts, roi_part_sizes = np.unique(mask_new[roi_orig_idx], return_counts=True)
            n_rois_in_old_area = len(roi_parts)
            if n_rois_in_old_area > 1:
                print(f'ROI#{roi_id} is disconnected into {n_rois_in_old_area} parts of sizes {roi_part_sizes}.')

    if return_num:
        return mask_new, n_comps
    else:
        return mask_new


def remove_small_rois(mask, min_size=3, verbose=True):
    """
    Removes all ROIs below the minimum size from the mask.
    """
    mask = mask.copy()
    roi_ids, roi_sizes = np.unique(mask, return_counts=True)
    small_roi_ids = roi_ids[roi_sizes < min_size]
    small_roi_locations = np.isin(mask, small_roi_ids)
    if verbose and small_roi_ids.size > 0:
        warnings.warn(f'{len(small_roi_ids)} ROIs <{min_size}px removed.')
    mask[small_roi_locations] = 0

    return mask


def shrink_light_artifact_rois(mask, n_artifact, verbose=True):
    mask = mask.copy()

    if verbose:
        light_artifact_mask = mask[:n_artifact, :]
        n_rois_in_artifact = len(np.unique(light_artifact_mask)) - 1  # to ignore background
        if n_rois_in_artifact:
            print(f'{n_rois_in_artifact} ROIs in light artifact region. ' +
                  f'Setting all artifact region pixel labels to background.')
    mask[:n_artifact, :] = 0
    return mask


def clean_rois(mask, n_artifact, min_size=3, connectivity=2, verbose=True):
    """
    First, all light artifact regions (top `n_artifact` pixels) are set to background.
    Then splits all disconnected ROIs (according to connectivity rule).
    Then removes ROIs below the minimum size.

    connectivity=1: only NSEW neighbors count as connections
    connectivity=2: NSEW neighbors and diagnonal neighbors count as connections
    """
    mask_new = shrink_light_artifact_rois(mask, n_artifact=n_artifact, verbose=verbose)
    mask_new = fix_disconnected_rois(mask_new, connectivity=connectivity, verbose=verbose)
    mask_new = remove_small_rois(mask_new, min_size=min_size, verbose=verbose)
    mask_new = relabel_mask(mask_new, connectivity=connectivity)  # re-label to fix ROI numbering after cleaning

    return mask_new


def find_neighbor_labels(x, y, labels):
    """For a given pixel coordinate in a 2D image, return the labels of the NSEW neighbors"""

    ymax, xmax = labels.shape

    if y == ymax - 1:
        north_neighbor = 0
    else:
        north_neighbor = labels[y + 1, x]

    if y == 0:
        south_neighbor = 0
    else:
        south_neighbor = labels[y - 1, x]

    if x == xmax - 1:
        east_neighbor = 0
    else:
        east_neighbor = labels[y, x + 1]

    if x == 0:
        west_neighbor = 0
    else:
        west_neighbor = labels[y, x - 1]

    return np.array([north_neighbor, south_neighbor, east_neighbor, west_neighbor])


def add_rois(rois_to_add, rois, connectivity=2, plot=False):
    """
    For each new ROI in `rois_to_add`, add to the existing ones in `rois` by
    performing the following steps:
        *check for intersections with existing ROIs
        *if any, remove from new ROI
        *check if the new ROI touches any existing ROI
         (NSEW neighbors count as touching)
        *if any, remove touching pixels from new ROI
        *check if after these steps, the new ROI is still connected
         (diagonal neighbors/`connectivity=2` count as connected by default)
        *if not, split into separate component
        *check if after these steps, the new ROI(s) are still large enough
        *if not, ignore all ROIs that are too small
        *add remaining ROIs to existing and continue

    returns
    ----
    output_labeled:
        contains the full output - new ROIs added to old
    output_new_labeled:
        contains only the ROIs that where added by this function
    """

    orig_roi_ids = np.unique(rois)
    new_rois_ids = np.unique(rois_to_add)
    new_rois_ids = new_rois_ids[new_rois_ids > 0]  # remove background
    n_output_rois_expected = len(orig_roi_ids) - 1  # -1 to ignore original ROI background label

    # make sure no index is used twice when adding ROIs
    new_roi_id_after_adding = np.max(orig_roi_ids)

    output_rois = rois.copy()
    output_rois_new = np.zeros_like(output_rois)
    for new_roi_id in zip(new_rois_ids):
        suggested_roi = rois_to_add == new_roi_id
        new_rois_no_pruning = (output_rois.copy() > 0).astype(int)
        new_rois_no_pruning[suggested_roi] = 2  # 0:bg 1:old rois 2:new roi

        # find intersection pixels with existing true ROIs
        intersection_pixel = (output_rois > 0) & suggested_roi
        y_int, x_int = np.where(intersection_pixel)
        # if there is an intersection, remove from suggested ROI
        if np.sum(intersection_pixel) > 0:
            suggested_roi[intersection_pixel] = 0
        new_rois_intersection_removed = (output_rois.copy() > 0).astype(int)
        new_rois_intersection_removed[suggested_roi] = 2  # 0:bg 1:old rois 2:new roi

        # if the new ROI touches existing ROI, remove those pixels from the suggestion
        x_remove = []
        y_remove = []
        y_sug, x_sug = np.where(suggested_roi)
        for x, y in zip(x_sug, y_sug):
            side_neighbor_labels = find_neighbor_labels(x, y, new_rois_intersection_removed)
            if any(side_neighbor_labels == 1):  # label 1: old rois
                x_remove.append(x)
                y_remove.append(y)
        suggested_roi[y_remove, x_remove] = 0
        new_rois_touching_removed = (output_rois.copy() > 0).astype(int)
        new_rois_touching_removed[suggested_roi] = 2  # 0:bg 1:old rois 2:new roi

        # split suggested ROI into connected components
        connected_comps, n_comps = fix_disconnected_rois(suggested_roi.astype(int) * new_roi_id,
                                                         connectivity=connectivity, return_num=True)
        # filter small ROIs
        connected_large_comps = remove_small_rois(connected_comps)
        comp_ids = np.unique(connected_large_comps)
        comp_ids = comp_ids[comp_ids > 0]  # ignore background
        for comp_id in comp_ids:
            comp_idx = connected_large_comps == comp_id
            new_roi_id_after_adding += 1
            n_output_rois_expected += 1
            output_rois[comp_idx] = new_roi_id_after_adding
            output_rois_new[comp_idx] = new_roi_id_after_adding

        # only plot if requested AND we alter the suggested ROI in any way before adding
        if plot and (np.sum(intersection_pixel) > 0 or n_comps > 1 or x_remove):
            plt.figure(figsize=(30, 5))
            plt.subplot(171)
            plt.title('new/old ROI overlap')
            plt.imshow(new_rois_no_pruning)
            plt.scatter(x_int, y_int, marker='.', c='tab:red')
            plt.subplot(172)
            plt.title('overlap pixels removed')
            plt.imshow(new_rois_intersection_removed)
            plt.subplot(173)
            plt.title('new/old ROIs touching')
            plt.imshow(new_rois_intersection_removed)
            plt.plot(x_remove, y_remove, '.', c='tab:red')
            plt.subplot(174)
            plt.title('touching pixels removed')
            plt.imshow(new_rois_touching_removed)
            plt.subplot(175)
            plt.title(f'new ROI after pruning\n{n_comps} connected area(s)')
            plt.imshow(connected_comps)
            plt.subplot(176)
            plt.imshow(connected_large_comps)
            plt.title(f'new ROI after size filter')

    # re-label output
    output_labeled, n_output_rois = relabel_mask(output_rois, return_num=True, connectivity=connectivity)
    output_new_labeled = relabel_mask(output_rois_new, connectivity=connectivity)

    if n_output_rois_expected != n_output_rois:
        warnings.warn(f"Expected {n_output_rois_expected} ROIs, obtained {n_output_rois} ROIs")

    return output_labeled, output_new_labeled


def intersection_over_union(masks_true, masks_pred):
    """Copied from cellpose"""
    overlap = label_overlap(masks_true, masks_pred)
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    iou = overlap / (n_pixels_pred + n_pixels_true - overlap)
    iou[np.isnan(iou)] = 0.0
    return iou


def label_overlap(x, y):
    """Copied from cellpose"""
    x = x.ravel()
    y = y.ravel()
    overlap = np.zeros((1 + x.max(), 1 + y.max()), dtype=np.uint)
    for i in range(len(x)):
        overlap[x[i], y[i]] += 1
    return overlap


def generate_roi_suggestions(mask_pred, mask_true, n_artifact, threshold=0.1, verbose=False):
    """
    Each predicted ROI that does not have IoU>`threshold` with any true ROI is considered a suggestion ROI.
    Returns a mask with only the suggestion ROIs.
    Light artifact region of the mask is set to background label.
    """
    mask_pred = clean_rois(mask_pred, n_artifact=n_artifact, verbose=verbose)
    iou_matrix = intersection_over_union(mask_true, mask_pred)[1:, 1:]  # `[1:,1:]` to exclude background
    pred_mask_ids = np.unique(mask_pred[mask_pred > 0])  # `mask_pred>0` to exclude background
    # find "false positive" ROI ids
    is_fp = np.max(iou_matrix, axis=0) < threshold
    # if not np.any(is_fp):
    #     return None

    fp_ids = pred_mask_ids[is_fp]
    # make mask only of FPs
    fp_locations = np.isin(mask_pred, fp_ids)
    rois_to_add = np.zeros_like(mask_pred)
    rois_to_add[fp_locations] = mask_pred[fp_locations]

    # rois_to_add = clean_rois(rois_to_add,n_artifact)

    return rois_to_add


def assert_igor_format(roi_mask):
    roi_mask = np.asarray(roi_mask)

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
    assert_igor_format(roi_mask)

    vmax = np.max(roi_mask)

    roi_mask = roi_mask.copy()
    roi_mask[roi_mask == vmax] = 0
    roi_mask = np.abs(roi_mask)

    return roi_mask


def to_roi_mask_file(data_file, old_suffix=None, new_suffix='_ROIs.pkl',
                     roi_mask_dir=None, old_prefix=None, new_prefix=None):
    """Get ROI mask file path from data file path"""

    f_path, f_name = os.path.split(data_file)
    f_root, f_dir = os.path.split(f_path)

    # Change suffix?
    if old_suffix is None:
        old_suffix = f_name.split('.')[-1]

    if not old_suffix.endswith('.'):
        old_suffix = '.' + old_suffix

    f_name = f_name.replace(old_suffix, '') + new_suffix

    # Change prefix?
    if old_prefix is not None:
        if f_name.startswith(old_prefix):
            f_name = f_name[len(old_prefix):]

    if new_prefix is not None:
        if not f_name.startswith(new_prefix):
            f_name = new_prefix + f_name

    # Change dir?
    if roi_mask_dir is None:
        roi_mask_dir = f_dir

    # Combine, use same root
    roi_mask_file = os.path.join(f_root, roi_mask_dir, f_name)

    return roi_mask_file


def sort_roi_mask_files(files, mask_alias='', highres_alias='', suffix='.h5', as_index=False):
    """Sort files by their relevance for the ROI masks given by the user"""
    files = np.array(files)
    penalties = np.full(files.size, len(mask_alias.split('_')))

    for i, file in enumerate(files):
        if check_shared_alias_str(highres_alias, file.lower().replace(suffix, '')):
            penalties[i] = len(mask_alias.split('_')) + 1

        else:
            for penalty, alias in enumerate(mask_alias.split('_')):
                if alias.lower() in file.lower().replace(suffix, '').split('_'):
                    penalties[i] = penalty

    # Penalize non control conditions
    for i, file in enumerate(files):
        is_control = ('_control' in file.lower()) or ('_ctrl' in file.lower()) or ('_c1' in file.lower())
        if not is_control:
            penalties[i] += 100

    sort_idxs = np.argsort(penalties)
    if as_index:
        return sort_idxs
    else:
        return files[sort_idxs]


def load_preferred_roi_mask_igor(files, mask_alias='', highres_alias='', suffix='.h5'):
    """Load ROI mask for field"""
    sorted_files = sort_roi_mask_files(files=files, mask_alias=mask_alias, highres_alias=highres_alias, suffix=suffix)
    for file in sorted_files:
        if os.path.isfile(file) and file.endswith('.h5'):
            roi_mask = scanm_utils.load_roi_mask_from_h5(filepath=file, ignore_not_found=True)
            if roi_mask is not None:
                return roi_mask, file
    else:
        return None, None


def load_preferred_roi_mask_pickle(files, mask_alias='', highres_alias='',
                                   roi_mask_dir=None, old_prefix=None, new_prefix=None):
    """Load ROI mask for field"""
    sorted_files = sort_roi_mask_files(files=files, mask_alias=mask_alias, highres_alias=highres_alias)
    for file in sorted_files:
        roimask_file = to_roi_mask_file(file, roi_mask_dir=roi_mask_dir, old_prefix=old_prefix, new_prefix=new_prefix)
        if os.path.isfile(roimask_file):
            with open(roimask_file, 'rb') as f:
                roi_mask = pickle.load(f).copy()
            roi_mask = to_igor_format(roi_mask)
            return roi_mask, roimask_file
    else:
        return None, None


def shift_array(img, shift, inplace=False, cval=np.min, n_artifact=0):
    """Shift >2d array in x and/or y. Fill borders with cval."""
    if not inplace:
        img = img.copy()
    img = np.asarray(img)

    if callable(cval):
        _cval = cval(img)
    else:
        _cval = cval

    img[:n_artifact, :] = _cval  # set light artifact region to background before shifting

    shift_ax0, shift_ax1 = shift
    img = np.roll(np.roll(img, shift_ax0, axis=0), shift_ax1, axis=1)

    if shift_ax0 > 0:
        img[:shift_ax0 + n_artifact, :] = _cval
    elif shift_ax0 < 0:
        img[shift_ax0:, :] = _cval
    if shift_ax1 > 0:
        img[:, :shift_ax1] = _cval
    elif shift_ax1 < 0:
        img[:, shift_ax1:] = _cval
    return img


def compare_roi_masks(roi_mask: np.ndarray, ref_roi_mask: np.ndarray, max_shift=5, bg_val=1) -> (str, tuple):
    """Test if two roi masks are the same"""
    assert_igor_format(roi_mask)
    assert_igor_format(ref_roi_mask)

    if roi_mask.shape != ref_roi_mask.shape:
        return 'different', (0, 0)
    if np.all(roi_mask == ref_roi_mask):
        return 'same', (0, 0)

    max_shift_x = np.minimum(max_shift, roi_mask.shape[0] - 2)
    max_shift_y = np.minimum(max_shift, roi_mask.shape[1] - 2)

    for dx in range(-max_shift_x, max_shift_x + 1):
        for dy in range(-max_shift_y, max_shift_y + 1):
            shifted_roi_mask = shift_array(roi_mask, shift=(dx, dy), cval=bg_val)
            if np.all((shifted_roi_mask == ref_roi_mask) | ((shifted_roi_mask == bg_val) & (ref_roi_mask != bg_val))):
                return 'shifted', (dx, dy)

    return 'different', (0, 0)
