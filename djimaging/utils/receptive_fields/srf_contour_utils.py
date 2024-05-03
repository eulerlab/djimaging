"""
Some of this code is based on Ziwei Huangs code for Ran et al 2020
"""
from djimaging.utils.plot_utils import srf_extent, plot_srf

try:
    import cv2
except:
    print('Failed to import cv2')

import numpy as np
from matplotlib import pyplot as plt


def compute_contour(srf, level, pixel_size_um, plot=False, ax=None):
    """Draw contour lines around receptive field"""

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(3, 2))

    extent = srf_extent(srf_shape=srf.shape, pixelsize=pixel_size_um)

    if np.any(srf > level) and np.any(srf < level):
        contour_plot = ax.contour(
            srf, levels=[level], colors=['k'], linestyles=['-'], linewidths=3, alpha=0.5, extent=extent)

        cntrs, cntrs_size = compute_closed_cntrs_and_sizes(cntrs=contour_plot.allsegs[0])
    else:
        cntrs = []
        cntrs_size = []

    if plot:
        plot_srf(srf, ax=ax, pixelsize=pixel_size_um, extent=extent)

        title = f"level = {level:.2g}\n"
        if len(cntrs_size) == 1:
            area = cntrs_size[0]
            cdia = np.sqrt(area / np.pi) * 2
            title += f"A={area:.1f} [μm²]\ncdia={cdia:.2f} [μm]"
        elif len(cntrs_size) > 0:
            area_max = np.max(cntrs_size)
            area_min = np.min(cntrs_size)
            title += f"A in [{area_min:.1f},\n {area_max:.1f}] [μm²]"
        else:
            title = f"no contours found\n"
        ax.set(title=title, xlabel='x [μm]', ylabel='y [μm]')
    else:
        plt.clf()
        plt.close()

    return cntrs, cntrs_size


def compute_cntr_sizes(cntrs, area_factor=1.0):
    """Compute closed_cntrs and their respective sizes"""
    cntrs_size = [cv2.contourArea(cntr.astype('float32')) * area_factor for cntr in cntrs]
    return cntrs_size


def compute_closed_cntrs_and_sizes(cntrs, area_factor=1.0):
    """Compute closed_cntrs and their respective sizes"""
    closed_cntrs = [cntr for cntr in cntrs if np.allclose(cntr[0], cntr[-1])]
    cntrs_size = compute_cntr_sizes(closed_cntrs, area_factor=area_factor)
    return closed_cntrs, cntrs_size


def compute_largest_cntr_and_size(cntrs, area_factor=1.0):
    """Compute closed_cntrs and their respective sizes"""
    cntrs_size = [cv2.contourArea(cntr.astype('float32')) * area_factor for cntr in cntrs]
    largest_idx = np.argmax(cntrs_size)
    return cntrs[largest_idx], cntrs_size[largest_idx]


def compute_cntr_center(cntr):
    """Compute center of contour"""
    moment = cv2.moments(cntr.astype('float32'))
    x_center = moment["m10"] / moment["m00"]
    y_center = moment["m01"] / moment["m00"]

    return x_center, y_center


def compute_irregular_index(cntr):
    """Compute how irregular RF is with respect to convex hull"""
    if len(cntr) == 0:
        return 1.

    hull = cv2.convexHull(cntr.astype(np.float32)).flatten().reshape(-1, 2)
    hull = np.vstack([hull, hull[0]])

    rf_area = cv2.contourArea(cntr.astype(np.float32))
    ch_area = cv2.contourArea(hull.astype(np.float32))

    irregular_index = 1 - rf_area / ch_area

    return irregular_index


def get_center_mask(cntr, srf_shape, pixelsize, plot=False):
    from skimage import draw

    mask_center = np.zeros(srf_shape, dtype='bool')
    mask_center[draw.polygon(cntr[:, 1] / pixelsize + srf_shape[0] / 2,
                             cntr[:, 0] / pixelsize + srf_shape[1] / 2)] = 1

    if plot:
        extent = srf_extent(srf_shape=mask_center.shape, pixelsize=pixelsize)

        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        ax.imshow(mask_center, extent=extent, origin='lower', )
        ax.plot(*cntr.T, c='black', ls='--')
        plt.show()

    return mask_center


def create_disk_mask(dist):
    """Create a binary disk shaped 2d-array"""
    from skimage import draw
    mask = np.zeros((2 * dist + 1, 2 * dist + 1), dtype='bool')
    mask[draw.disk(center=(int(mask.shape[0] / 2), int(mask.shape[1] / 2)), radius=dist, shape=mask.shape)] = 1
    return mask


def get_surround_mask(mask_center, pixelsize, d_inner, d_outer, plot=False):
    from scipy.signal import convolve2d

    inner_dist = int(np.ceil(d_inner / pixelsize))
    outer_dist = int(np.ceil(d_outer / pixelsize))

    if d_inner > 0:
        inner_disk = create_disk_mask(inner_dist)
        inner_mask = convolve2d(mask_center.astype(int), inner_disk, mode='same').astype(bool)
    else:
        inner_mask = mask_center.copy()

    outer_disk = create_disk_mask(outer_dist)
    outer_mask = convolve2d(mask_center.astype(int), outer_disk, mode='same').astype(bool)

    mask_surround = outer_mask.copy()
    mask_surround[np.where(inner_mask)] = 0

    mask_full_surround = ~inner_mask

    if plot:
        plot_center_and_surround_masks(mask_center, mask_surround, mask_full_surround, pixelsize)
        plt.show()

    return mask_surround, mask_full_surround


def get_center_and_surround_masks(cntr, srf_shape, pixelsize, d_inner, d_outer):
    mask_center = get_center_mask(cntr, srf_shape, pixelsize, plot=False)
    mask_surround, mask_full_surround = get_surround_mask(mask_center, pixelsize, d_inner, d_outer, plot=False)

    return mask_center, mask_surround, mask_full_surround


def plot_center_and_surround_masks(mask_center, mask_surround, mask_full_surround, pixelsize, axs=None):
    if axs is None:
        fig, axs = plt.subplots(1, 4, figsize=(12, 6), squeeze=True)

    extent = srf_extent(srf_shape=mask_center.shape, pixelsize=pixelsize)

    ax = axs[0]
    ax.imshow(mask_center, extent=extent, origin='lower', interpolation='none')
    ax.set(title='mask_center')

    ax = axs[1]
    ax.imshow(mask_full_surround, extent=extent, origin='lower', interpolation='none')
    ax.set(title='mask_full_surround')

    ax = axs[2]
    ax.imshow(mask_surround, extent=extent, origin='lower', interpolation='none')
    ax.set(title='mask_surround')

    ax = axs[3]
    ax.imshow(mask_center + mask_surround, extent=extent, origin='lower', interpolation='none')
    ax.set(title='both')

    return axs
