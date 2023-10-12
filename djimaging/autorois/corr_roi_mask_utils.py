import warnings

import numpy as np
import itertools
from collections import deque
from matplotlib import pyplot as plt


class CorrRoiMask:
    """12.6 um² ~= pi * ((4 um) / 2)² """

    def __init__(
            self, cut_x=(1, 1), cut_z=(1, 1), min_area_um2=0.8, max_area_um2=12.6,
            n_pix_min=None, n_pix_max=None, line_threshold_q=70, line_threshold_min=0.1,
            grow_only_local_thresh_pixels=True, use_ch0_stack=True, grow_use_corr_map=False, grow_threshold=None):
        self.cut_x = cut_x
        self.cut_z = cut_z

        self.min_area_um2 = min_area_um2
        self.max_area_um2 = max_area_um2
        self.n_pix_min = n_pix_min
        self.n_pix_max = n_pix_max
        self.line_threshold_q = line_threshold_q
        self.line_threshold_min = line_threshold_min
        self.grow_only_local_thresh_pixels = grow_only_local_thresh_pixels
        self.use_ch0_stack = use_ch0_stack
        self.grow_use_corr_map = grow_use_corr_map
        self.grow_threshold = grow_threshold

    def create_mask_from_data(self, ch0_stack, ch1_stack, n_artifact, pixel_size_um=(1., 1.), plot=False, **kwargs):
        if len(kwargs) > 0:
            warnings.warn(f'ignoring kwargs: {kwargs}')

        cut_x = np.array(self.cut_x)
        cut_x[0] = np.maximum(cut_x[0], n_artifact)

        stack = ch0_stack if self.use_ch0_stack else ch1_stack

        n_pix_min = self.n_pix_min or 2
        n_pix_min = int(np.maximum(np.round(self.min_area_um2 / (pixel_size_um[0] * pixel_size_um[1])), n_pix_min))

        n_pix_max = self.n_pix_max or 2
        n_pix_max = int(np.maximum(np.round(self.max_area_um2 / (pixel_size_um[0] * pixel_size_um[1])), n_pix_max))

        # At most twice as long as wide
        n_pix_max_x = int(np.maximum(np.sqrt(2 * self.max_area_um2), 1))
        n_pix_max_z = int(np.maximum(np.sqrt(2 * self.max_area_um2), 1))

        print(f'Creating ROI mask for stack of shape={stack.shape}, '
              f'with n_pix_min={n_pix_min}, n_pix_max={n_pix_max}, '
              f'n_pix_max_x={n_pix_max_x}, n_pix_max_z={n_pix_max_z}, '
              f'pixel_size_um={pixel_size_um}')

        if n_pix_max < n_pix_min:
            n_pix_max = n_pix_min

        roi_mask = corr_map_rois(
            stack, cut_x=cut_x, cut_z=self.cut_z, n_pix_max_x=n_pix_max_x, n_pix_max_z=n_pix_max_z,
            n_pix_max=n_pix_max, n_pix_min=n_pix_min,
            line_threshold_q=self.line_threshold_q, line_threshold_min=self.line_threshold_min,
            only_local_thresh_pixels=self.grow_only_local_thresh_pixels, grow_use_corr_map=self.grow_use_corr_map,
            grow_threshold=self.grow_threshold, plot=plot)

        return roi_mask


def stack_corr_image(stack, cut_x=(1, 1), cut_z=(1, 1)):
    """Zhijian Zhao 2020: First, we estimated a correlation image by correlating the trace of every pixel with the
    trace of its eight neighbouring pixels and calculating the mean local correlation (ρ_local)"""
    nx, nz, nt = stack.shape

    counts = np.zeros((nx, nz), dtype=int)
    corr_map = np.zeros((nx, nz), dtype=float)

    for ix, iz, dx, dz in itertools.product(range(0, nx), range(0, nz), [-1, 0, 1], [-1, 0, 1]):
        if cut_x[0] <= ix < (nx - cut_x[1]) and cut_z[0] <= iz < (nz - cut_z[1]):
            if 0 <= (ix + dx) < nx and 0 <= (iz + dz) < nz and (dx != 0 or dz != 0):
                counts[ix, iz] += 1
                corr_map[ix, iz] += np.corrcoef(stack[ix, iz, :], stack[ix + dx, iz + dz, :])[0, 1]

    corr_map[counts > 0] = corr_map[counts > 0] / counts[counts > 0]

    return corr_map


def stack_corr_line_threshold(corr_map, cut_x=(1, 1), cut_z=(1, 1), q=70, line_threshold_min=0.1):
    """Zhijian Zhao 2020: In contrast to previous x-y recordings, local correlation of neighbouring pixels varied
    with IPL depth in x-z scans [...]. To account for that, an IPL depth-specific correlation threshold
    (ρ_threshold) was defined as the 70th percentile of all local correlation values in each z-axis scan line"""

    nx, nz = corr_map.shape
    line_threshold = np.full(nz, 1.)
    line_threshold[cut_z[0]:nz - cut_z[1]] = np.percentile(
        corr_map[cut_x[0]:nx - cut_x[1], cut_z[0]:nz - cut_z[1]], q=q, axis=0)
    line_threshold[line_threshold < line_threshold_min] = line_threshold_min
    return line_threshold


def corr_map_rois(stack, cut_x, cut_z, n_pix_max_x=5, n_pix_max_z=5, n_pix_max=25, n_pix_min=3.,
                  only_local_thresh_pixels=False, corr_map=None, grow_use_corr_map=False,
                  line_threshold=None, line_threshold_q=70, line_threshold_min=0.1, grow_threshold=None, plot=False):
    """For every pixel with ρ_local>ρ_threshold (“seed pixel”), we grouped neighbouring pixels with ρ_local>ρ_threshold
    into one ROI. To match ROI sizes with the sizes of BC axon terminals, we restricted ROI diameters
    (estimated as effective diameter of area-equivalent circle) to range between 1 and 4 μm."""

    if corr_map is None:
        corr_map = stack_corr_image(stack, cut_x=cut_x, cut_z=cut_z)

    if line_threshold is None:
        line_threshold = stack_corr_line_threshold(
            corr_map, cut_x=cut_x, cut_z=cut_z, q=line_threshold_q, line_threshold_min=line_threshold_min)

    nx, nz, nt = stack.shape
    seed_ixs, seed_iys = np.argwhere(corr_map >= line_threshold).T
    roi_mask = np.zeros((nx, nz), dtype=int)

    roi_id = 1
    ignore_map = np.zeros(corr_map.shape, dtype=bool)
    if only_local_thresh_pixels:
        ignore_map[corr_map < line_threshold] = True
    ignore_map[:cut_x[0], :] = True
    ignore_map[nx - cut_x[1]:, :] = True
    ignore_map[:, cut_z[0]] = True
    ignore_map[:, nz - cut_z[1]:] = True

    p_threshold = grow_threshold if grow_threshold is not None else line_threshold

    for seed_ix, seed_iz in sorted(zip(seed_ixs, seed_iys), key=lambda v: corr_map[v[0], v[1]]):
        if roi_mask[seed_ix, seed_iz] != 0:
            continue

        roi_pixels = corr_map_grow_roi(
            seed_ix, seed_iz, n_pix_max=n_pix_max, n_pix_max_x=n_pix_max_x, n_pix_max_z=n_pix_max_z,
            p_threshold=p_threshold, stack=stack, ignore_map=ignore_map,
            corr_map=corr_map if grow_use_corr_map else None)

        if len(roi_pixels) == 0:
            continue

        roi_ixs, roi_iys = np.array(roi_pixels).T

        if roi_ixs.size >= n_pix_min:
            roi_mask[roi_ixs, roi_iys] = roi_id
            ignore_map[roi_ixs, roi_iys] = True
            roi_id += 1

    if plot:
        fig, axs = plt.subplots(1, 5, figsize=(15, 2.2))
        axs[0].imshow(np.mean(stack, axis=2).T, cmap='gray', origin='lower')
        axs[1].imshow(corr_map.T, cmap='viridis', origin='lower')
        axs[2].plot(line_threshold, np.arange(line_threshold.size))
        axs[2].set_xlim(0, 1)
        axs[3].imshow((corr_map >= line_threshold).T, cmap='viridis', origin='lower')
        axs[4].imshow(roi_mask.T, cmap='jet', origin='lower')
        plt.show()

    return roi_mask


def corr_map_grow_roi(seed_ix, seed_iz, n_pix_max, n_pix_max_x, n_pix_max_z, p_threshold, stack,
                      ignore_map=None, corr_map=None):
    """Starting from a seed pixel, collect all correlated pixels until pixel limit is reached"""

    nx, nz, nt = stack.shape
    visited = np.zeros((nx, nz), dtype=bool)
    if ignore_map is not None:
        visited[ignore_map] = True

    p_threshold = np.asarray(p_threshold)
    if p_threshold.size == 1:
        p_threshold = np.full(nz, p_threshold)

    queue = deque([(seed_ix, seed_iz, 1.)])
    roi_pixels = []
    min_x, max_x = seed_ix, seed_ix
    min_z, max_z = seed_iz, seed_iz

    reached_x_max = False
    reached_z_max = False

    while queue and len(roi_pixels) < n_pix_max:
        ix, iz, corr_i = queue.pop()
        if visited[ix, iz]:
            continue

        visited[ix, iz] = True

        if corr_i >= p_threshold[iz]:
            roi_pixels.append((ix, iz))

            # update range
            if not reached_x_max:
                min_x = np.minimum(ix, min_x)
                max_x = np.maximum(ix, max_x)

                if (max_x - min_x) >= n_pix_max_x - 1:
                    visited[:min_x, :] = True
                    visited[max_x + 1:, :] = True
                    reached_x_max = True

            if not reached_z_max:
                min_z = np.minimum(iz, min_z)
                max_z = np.maximum(iz, max_z)

                if (max_z - min_z) >= n_pix_max_z - 1:
                    visited[:, :min_z] = True
                    visited[:, max_z + 1:] = True
                    reached_z_max = True

            for dx, dz in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_ix, new_iy = ix + dx, iz + dz
                if 0 <= new_ix < nx and 0 <= new_iy < nz and not visited[new_ix, new_iy]:
                    if corr_map is not None:
                        new_corr_i = corr_map[new_ix, new_iy]
                    else:
                        new_corr_i = np.corrcoef(stack[seed_ix, seed_iz, :], stack[new_ix, new_iy, :])[0, 1]
                    queue.append((new_ix, new_iy, new_corr_i))

        queue = deque(sorted(queue, key=lambda v: v[2]))

    return roi_pixels
