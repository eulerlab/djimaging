import os
import pickle
import queue
import warnings
from typing import Callable

import numpy as np
from matplotlib import pyplot as plt

from djimaging.utils.alias_utils import check_shared_alias_str
from djimaging.utils.mask_format_utils import assert_igor_format, to_igor_format
from djimaging.utils.scanm import read_h5_utils
from djimaging.utils.cellpose_utils import intersection_over_union


def create_circular_mask(h: int, w: int, center: tuple, radius: float) -> np.ndarray:
    """Create a binary circular mask for a 2-D grid.

    Parameters
    ----------
    h : int
        Height of the grid (number of rows).
    w : int
        Width of the grid (number of columns).
    center : tuple
        ``(x, y)`` coordinates of the circle centre within the grid.
    radius : float
        Radius of the circle in pixels.

    Returns
    -------
    np.ndarray
        Boolean array of shape ``(w, h)`` that is True inside the circle.
    """
    xs, ys = np.ogrid[:w, :h]
    dist_from_center = np.sqrt((xs - center[0]) ** 2 + (ys - center[1]) ** 2)
    mask = np.asarray(dist_from_center <= radius)
    return mask


def extract_connected_mask(mask: np.ndarray, i: int, j: int) -> np.ndarray:
    """Extract the connected component containing pixel ``(i, j)`` from a binary mask.

    This code contains content from Stack Overflow
    Source: https://stackoverflow.com/a/35224850

    Stack Overflow content is licensed under CC BY-SA 3.0
    (Creative Commons Attribution-ShareAlike 3.0 Unported License)
    https://creativecommons.org/licenses/by-sa/3.0/

    Code by Stack Overflow user: https://stackoverflow.com/users/4613543/philokey
    Modified: Only minor modifications from original code

    Parameters
    ----------
    mask : np.ndarray
        2-D binary integer mask (values 0 or 1).
    i : int
        Row index of the seed pixel.
    j : int
        Column index of the seed pixel.

    Returns
    -------
    np.ndarray
        Binary mask of the same shape as `mask` containing only the connected
        component that includes pixel ``(i, j)``.
    """
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


def get_mask_by_cc(seed_ix: int, seed_iy: int, data: np.ndarray,
                   seed_trace: np.ndarray | None = None, thresh: float = 0.2,
                   max_pixel_dist: int = 10, plot: bool = False) -> np.ndarray:
    """Generate an ROI mask by cross-correlating pixels with a seed trace.

    Pixels within `max_pixel_dist` of the seed whose correlation with
    `seed_trace` exceeds `thresh` are included; the result is further trimmed
    to the largest connected component containing the seed pixel.

    Parameters
    ----------
    seed_ix : int
        Row index of the seed pixel.
    seed_iy : int
        Column index of the seed pixel.
    data : np.ndarray
        3-D data array of shape ``(nx, ny, n_frames)``.
    seed_trace : np.ndarray | None, optional
        1-D reference trace. If None, the trace at ``data[seed_ix, seed_iy]``
        is used.
    thresh : float, optional
        Minimum Pearson correlation coefficient to include a pixel.
        Default is 0.2.
    max_pixel_dist : int, optional
        Maximum pixel distance from the seed to consider. Default is 10.
    plot : bool, optional
        If True, diagnostic plots are shown. Default is False.

    Returns
    -------
    np.ndarray
        Binary mask of shape ``(nx, ny)`` for the extracted ROI.
    """
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


def get_mask_by_bg(seed_ix: int, seed_iy: int, data: np.ndarray,
                   ref_value: float | None = None, thresh: float = 0.2,
                   max_pixel_dist: int = 10, plot: bool = False) -> np.ndarray:
    """Get mask based on background image pixel values.

    Pixels within `max_pixel_dist` of the seed whose absolute difference from
    `ref_value` is at most `thresh` are included; the result is trimmed to the
    connected component containing the seed pixel.

    Parameters
    ----------
    seed_ix : int
        Row index of the seed pixel.
    seed_iy : int
        Column index of the seed pixel.
    data : np.ndarray
        2-D background image array of shape ``(nx, ny)``.
    ref_value : float | None, optional
        Reference intensity value. If None, the value at
        ``data[seed_ix, seed_iy]`` is used.
    thresh : float, optional
        Maximum absolute difference from `ref_value` to include a pixel.
        Default is 0.2.
    max_pixel_dist : int, optional
        Maximum pixel distance from the seed to consider. Default is 10.
    plot : bool, optional
        If True, diagnostic plots are shown. Default is False.

    Returns
    -------
    np.ndarray
        Binary mask of shape ``(nx, ny)`` for the extracted ROI.
    """
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


def relabel_mask(mask: np.ndarray, connectivity: int,
                 return_num: bool = False) -> np.ndarray | tuple[np.ndarray, int]:
    """Relabel connected components of a mask using ``skimage.measure.label``.

    Parameters
    ----------
    mask : np.ndarray
        2-D integer mask to relabel.
    connectivity : int
        Pixel connectivity: 1 for NSEW neighbours only, 2 to include diagonals.
    return_num : bool, optional
        If True, also return the number of connected components. Default is False.

    Returns
    -------
    np.ndarray
        Relabelled mask array.
    tuple[np.ndarray, int]
        ``(relabelled_mask, n_components)`` when `return_num` is True.
    """
    from skimage.measure import label

    if return_num:
        new_mask, num = label(mask.T, connectivity=connectivity, return_num=True)
        return new_mask.T, num
    else:
        return label(mask.T, connectivity=connectivity).T


def fix_disconnected_rois(mask: np.ndarray, connectivity: int = 2,
                          verbose: bool = True,
                          return_num: bool = False) -> np.ndarray | tuple[np.ndarray, int]:
    """Re-label a mask so that each connected component gets a unique label.

    Uses ``skimage.measure.label()``; all connected pixels with the same
    value receive the same label.

    connectivity=1: only NSEW neighbors count as connections
    connectivity=2: NSEW neighbors and diagonal neighbors count as connections

    Parameters
    ----------
    mask : np.ndarray
        2-D integer ROI mask to process.
    connectivity : int, optional
        Pixel connectivity. Default is 2 (8-connected).
    verbose : bool, optional
        If True, prints information about disconnected ROIs. Default is True.
    return_num : bool, optional
        If True, also return the total number of connected components.
        Default is False.

    Returns
    -------
    np.ndarray
        Relabelled mask.
    tuple[np.ndarray, int]
        ``(relabelled_mask, n_components)`` when `return_num` is True.
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


def remove_small_rois(mask: np.ndarray, min_size: int = 3, verbose: bool = True) -> np.ndarray:
    """Remove all ROIs below the minimum pixel count from the mask.

    Parameters
    ----------
    mask : np.ndarray
        2-D integer ROI mask (0 = background).
    min_size : int, optional
        Minimum number of pixels required to keep an ROI. Default is 3.
    verbose : bool, optional
        If True, a warning is issued listing removed ROIs. Default is True.

    Returns
    -------
    np.ndarray
        Modified mask with small ROIs replaced by background (0).
    """
    mask = mask.copy()
    roi_ids, roi_sizes = np.unique(mask, return_counts=True)
    small_roi_ids = roi_ids[roi_sizes < min_size]
    small_roi_locations = np.isin(mask, small_roi_ids)
    if verbose and small_roi_ids.size > 0:
        warnings.warn(f'{len(small_roi_ids)} ROIs <{min_size}px removed.')
    mask[small_roi_locations] = 0

    return mask


def shrink_light_artifact_rois(mask: np.ndarray, n_artifact: int, verbose: bool = True) -> np.ndarray:
    """Set the top `n_artifact` rows of a mask to background to remove light-artifact ROIs.

    Parameters
    ----------
    mask : np.ndarray
        2-D integer ROI mask.
    n_artifact : int
        Number of rows at the top of the mask that form the light-artifact
        region.
    verbose : bool, optional
        If True, prints how many ROIs were affected. Default is True.

    Returns
    -------
    np.ndarray
        Mask with the artifact region set to 0.
    """
    mask = mask.copy()

    if verbose:
        light_artifact_mask = mask[:n_artifact, :]
        n_rois_in_artifact = len(np.unique(light_artifact_mask)) - 1  # to ignore background
        if n_rois_in_artifact:
            print(f'{n_rois_in_artifact} ROIs in light artifact region. ' +
                  f'Setting all artifact region pixel labels to background.')
    mask[:n_artifact, :] = 0
    return mask


def clean_rois(mask: np.ndarray, n_artifact: int, min_size: int = 3,
               connectivity: int = 2, verbose: bool = True) -> np.ndarray:
    """Clean an ROI mask by removing artifacts, splitting disconnected ROIs, and removing small ROIs.

    First, all light artifact regions (top `n_artifact` pixels) are set to background.
    Then splits all disconnected ROIs (according to connectivity rule).
    Then removes ROIs below the minimum size.

    connectivity=1: only NSEW neighbors count as connections
    connectivity=2: NSEW neighbors and diagnonal neighbors count as connections

    Parameters
    ----------
    mask : np.ndarray
        2-D integer ROI mask to clean.
    n_artifact : int
        Number of rows at the top of the mask forming the light-artifact region.
    min_size : int, optional
        Minimum pixel count to keep an ROI. Default is 3.
    connectivity : int, optional
        Pixel connectivity used for splitting and relabelling. Default is 2.
    verbose : bool, optional
        If True, processing messages are printed/warned. Default is True.

    Returns
    -------
    np.ndarray
        Cleaned and relabelled ROI mask.
    """
    mask_new = shrink_light_artifact_rois(mask, n_artifact=n_artifact, verbose=verbose)
    mask_new = fix_disconnected_rois(mask_new, connectivity=connectivity, verbose=verbose)
    mask_new = remove_small_rois(mask_new, min_size=min_size, verbose=verbose)
    mask_new = relabel_mask(mask_new, connectivity=connectivity)  # re-label to fix ROI numbering after cleaning

    return mask_new


def find_neighbor_labels(x: int, y: int, labels: np.ndarray) -> np.ndarray:
    """Return the NSEW neighbour labels for a given pixel in a 2-D label image.

    Parameters
    ----------
    x : int
        Column index of the pixel.
    y : int
        Row index of the pixel.
    labels : np.ndarray
        2-D integer label array of shape ``(height, width)``.

    Returns
    -------
    np.ndarray
        1-D integer array ``[north, south, east, west]`` containing the label
        values of the four cardinal neighbours (0 for out-of-bounds).
    """
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


def add_rois(rois_to_add: np.ndarray, rois: np.ndarray, connectivity: int = 2,
             plot: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Add new ROIs from `rois_to_add` into an existing ROI mask `rois`.

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

    Parameters
    ----------
    rois_to_add : np.ndarray
        2-D integer mask containing proposed new ROIs to be merged.
    rois : np.ndarray
        2-D integer mask of existing ROIs (0 = background).
    connectivity : int, optional
        Pixel connectivity for connected-component analysis. Default is 2.
    plot : bool, optional
        If True, show diagnostic plots whenever a proposed ROI is altered
        before being added. Default is False.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(output_labeled, output_new_labeled)`` where ``output_labeled``
        contains all ROIs (old + newly added) and ``output_new_labeled``
        contains only the ROIs added by this function.
    """
    orig_roi_ids = np.unique(rois)
    new_rois_ids = np.unique(rois_to_add)
    new_rois_ids = new_rois_ids[new_rois_ids > 0]  # remove background
    n_output_rois_expected = len(orig_roi_ids) - 1  # -1 to ignore original ROI background label

    # make sure no index is used twice when adding ROIs
    new_roi_id_after_adding = np.max(orig_roi_ids)

    output_rois = rois.copy()
    output_rois_new = np.zeros_like(output_rois)
    for new_roi_id in new_rois_ids:
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


def generate_roi_suggestions(mask_pred: np.ndarray, mask_true: np.ndarray, n_artifact: int,
                              threshold: float = 0.1, verbose: bool = False) -> np.ndarray:
    """Generate suggested new ROIs from a predicted mask that are absent in the true mask.

    Each predicted ROI that does not have IoU > `threshold` with any true ROI
    is considered a suggestion ROI. Returns a mask with only the suggestion
    ROIs. Light artifact region of the mask is set to background label.

    Parameters
    ----------
    mask_pred : np.ndarray
        2-D integer mask of predicted ROIs.
    mask_true : np.ndarray
        2-D integer mask of ground-truth ROIs.
    n_artifact : int
        Number of rows at the top of the mask forming the light-artifact region.
    threshold : float, optional
        IoU threshold below which a predicted ROI is considered a false positive
        (suggestion). Default is 0.1.
    verbose : bool, optional
        If True, verbose output is passed to :func:`clean_rois`. Default is False.

    Returns
    -------
    np.ndarray
        Mask containing only the suggested (false-positive) ROIs; background is 0.
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


def to_roi_mask_file(data_file: str, old_suffix: str | None = None, new_suffix: str = '_ROIs.pkl',
                     roi_mask_dir: str | None = None, old_prefix: str | None = None,
                     new_prefix: str | None = None) -> str:
    """Derive the ROI mask file path from a data file path.

    Parameters
    ----------
    data_file : str
        Full path to the data file.
    old_suffix : str | None, optional
        Suffix to strip from the filename. If None, the existing file extension
        is used.
    new_suffix : str, optional
        Suffix to append after stripping `old_suffix`. Default is ``'_ROIs.pkl'``.
    roi_mask_dir : str | None, optional
        Subdirectory name where the mask file should be located. If None, the
        same directory as `data_file` is used.
    old_prefix : str | None, optional
        Prefix to strip from the filename before constructing the mask path.
    new_prefix : str | None, optional
        Prefix to prepend to the mask filename.

    Returns
    -------
    str
        Full path to the corresponding ROI mask file.
    """
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


def sort_roi_mask_files(files: list, mask_alias: str = '', highres_alias: str = '',
                        as_index: bool = False) -> np.ndarray:
    """Sort files by their relevance for the ROI masks given by the user.

    Parameters
    ----------
    files : list
        List of file paths to sort.
    mask_alias : str, optional
        Underscore-separated alias string that identifies preferred mask files.
        Tokens earlier in the string receive lower (better) penalty scores.
    highres_alias : str, optional
        Alias string for high-resolution files, which receive a higher penalty
        (lower preference) than regular mask files.
    as_index : bool, optional
        If True, return the sort index array instead of the sorted file list.
        Default is False.

    Returns
    -------
    np.ndarray
        Sorted array of file paths, or integer sort-index array if `as_index`
        is True.
    """
    files_base = np.array([os.path.splitext(os.path.basename(f))[0].lower()
                           for f in files])

    penalties = np.full(files_base.size, len(mask_alias.split('_')), dtype=float)

    for i, file in enumerate(files_base):
        if check_shared_alias_str(highres_alias, file):
            penalties[i] = len(mask_alias.split('_')) + 1

        else:
            for penalty, alias in enumerate(mask_alias.split('_')):
                if alias in file.split('_'):
                    penalties[i] = penalty

    penalties += np.arange(len(files_base))[np.argsort(files_base)] * 0.01

    # Penalize non control conditions
    for i, file in enumerate(files_base):
        is_control = ('_control' in file) or ('_ctrl' in file) or ('_c1' in file)
        if not is_control:
            penalties[i] += 100

    sort_idxs = np.argsort(penalties)
    if as_index:
        return sort_idxs
    else:
        return np.asarray(files)[sort_idxs]


def load_preferred_roi_mask_igor(files: list, mask_alias: str = '',
                                  highres_alias: str = '') -> tuple[np.ndarray | None, str | None]:
    """Load the most preferred ROI mask from a list of HDF5 files (Igor format).

    Parameters
    ----------
    files : list
        List of file paths to search.
    mask_alias : str, optional
        Alias string used to rank mask file preference.
    highres_alias : str, optional
        Alias string for high-resolution files (lower preference).

    Returns
    -------
    tuple[np.ndarray | None, str | None]
        ``(roi_mask, filepath)`` of the first successfully loaded mask, or
        ``(None, None)`` if no valid mask is found.
    """
    sorted_files = sort_roi_mask_files(files, mask_alias=mask_alias, highres_alias=highres_alias)
    for file in sorted_files:
        if os.path.isfile(file) and file.endswith('.h5'):
            roi_mask = read_h5_utils.load_roi_mask(filepath=file, ignore_not_found=True)
            if roi_mask is not None:
                return roi_mask, file
    else:
        return None, None


def load_preferred_roi_mask_pickle(files: list, mask_alias: str = '', highres_alias: str = '',
                                   roi_mask_dir: str | None = None, old_prefix: str | None = None,
                                   new_prefix: str | None = None) -> tuple[np.ndarray | None, str | None]:
    """Load the most preferred ROI mask from a list of files via pickle (Igor format).

    Parameters
    ----------
    files : list
        List of data file paths used to derive ROI mask file paths.
    mask_alias : str, optional
        Alias string used to rank mask file preference.
    highres_alias : str, optional
        Alias string for high-resolution files (lower preference).
    roi_mask_dir : str | None, optional
        Subdirectory override for locating pickle mask files.
    old_prefix : str | None, optional
        Prefix to strip when deriving the mask file name.
    new_prefix : str | None, optional
        Prefix to prepend when deriving the mask file name.

    Returns
    -------
    tuple[np.ndarray | None, str | None]
        ``(roi_mask, filepath)`` in Igor format for the first found mask, or
        ``(None, None)`` if no mask is found.
    """
    sorted_files = sort_roi_mask_files(files, mask_alias=mask_alias, highres_alias=highres_alias)
    for file in sorted_files:
        roimask_file = to_roi_mask_file(file, roi_mask_dir=roi_mask_dir, old_prefix=old_prefix, new_prefix=new_prefix)
        if os.path.isfile(roimask_file):
            with open(roimask_file, 'rb') as f:
                roi_mask = pickle.load(f).copy()
            roi_mask = to_igor_format(roi_mask)
            return roi_mask, roimask_file
    else:
        return None, None


def shift_array(img: np.ndarray, shift: tuple, inplace: bool = False,
                cval: float | Callable = np.min, n_artifact: int = 0) -> np.ndarray:
    """Shift a ≥2-D array in x and/or y, filling borders with a constant value.

    Parameters
    ----------
    img : np.ndarray
        Input array to shift (at least 2-D). First two axes are spatial.
    shift : tuple
        ``(shift_axis0, shift_axis1)`` integer shift amounts. Positive values
        shift the content towards higher indices.
    inplace : bool, optional
        If False (default), a copy of `img` is modified. If True, `img` is
        modified in place.
    cval : float | callable, optional
        Fill value for the newly exposed border pixels. If callable, it is
        called with `img` as argument to determine the value. Default is
        ``np.min``.
    n_artifact : int, optional
        Number of rows at the top of the array to set to `cval` before
        shifting (light-artifact region). Default is 0.

    Returns
    -------
    np.ndarray
        Shifted array of the same shape as `img`.
    """
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


def compare_roi_masks(roi_mask: np.ndarray, ref_roi_mask: np.ndarray,
                      max_shift: int = 5, bg_val: int = 1) -> tuple[str, tuple]:
    """Test whether two Igor-format ROI masks are the same, allowing for small shifts.

    Parameters
    ----------
    roi_mask : np.ndarray
        2-D ROI mask to compare (Igor format).
    ref_roi_mask : np.ndarray
        2-D reference ROI mask (Igor format).
    max_shift : int, optional
        Maximum pixel shift (in each axis) to try when testing for shifts.
        Default is 5.
    bg_val : int, optional
        Background pixel value used when filling shifted borders. Default is 1.

    Returns
    -------
    tuple[str, tuple]
        ``(status, (dx, dy))`` where ``status`` is one of ``'same'``,
        ``'shifted'``, or ``'different'``, and ``(dx, dy)`` is the detected
        shift (0, 0) when the masks are identical or different.

    Raises
    ------
    AssertionError
        If either mask fails the Igor-format assertion.
    """
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
