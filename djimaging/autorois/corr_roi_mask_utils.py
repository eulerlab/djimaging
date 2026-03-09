import warnings
from collections import deque
from typing import Callable

import numpy as np
from matplotlib import pyplot as plt

from djimaging.autorois.autoshift_utils import compute_corr_map


class CorrRoiMask:
    """ROI mask creator based on local pixel-correlation maps.

    Implements the algorithm described in Zhijian Zhao 2020: pixels with a
    local correlation above a depth-dependent threshold are used as seeds, and
    neighbouring pixels with sufficient correlation are merged into one ROI.

    Parameters
    ----------
    cut_x : tuple, optional
        Number of pixels to exclude at the (start, end) of the x axis.
        Default is ``(1, 1)``.
    cut_z : tuple, optional
        Number of pixels to exclude at the (start, end) of the z axis.
        Default is ``(1, 1)``.
    min_area_um2 : float, optional
        Minimum ROI area in µm². Default is 0.8.
    max_area_um2 : float, optional
        Maximum ROI area in µm². Approximately pi*(2 µm)² = 12.6 µm².
        Default is 12.6.
    n_pix_min : int or None, optional
        Hard override for minimum pixel count per ROI. Default is ``None``.
    n_pix_max : int or None, optional
        Hard override for maximum pixel count per ROI. Default is ``None``.
    line_threshold_q : int, optional
        Percentile used for the depth-dependent correlation threshold.
        Default is 70.
    line_threshold_min : float, optional
        Minimum value of the line threshold. Default is 0.1.
    n_pix_max_x : int or None, optional
        Maximum ROI extent in the x direction in pixels. Default is ``None``
        (derived from ``max_area_um2``).
    n_pix_max_z : int or None, optional
        Maximum ROI extent in the z direction in pixels. Default is ``None``
        (derived from ``max_area_um2``).
    grow_only_local_thresh_pixels : bool, optional
        If ``True``, only consider pixels above the threshold during growing.
        Default is ``True``.
    use_ch0_stack : bool, optional
        If ``True``, use channel 0; otherwise use channel 1. Default is
        ``True``.
    grow_use_corr_map : bool, optional
        If ``True``, use the precomputed correlation map during growing instead
        of computing pairwise correlations on the fly. Default is ``False``.
    grow_threshold : float or None, optional
        Correlation threshold used during growing. If ``None``, the
        line-threshold is used. Default is ``None``.
    """

    def __init__(
            self, cut_x=(1, 1), cut_z=(1, 1), min_area_um2=0.8, max_area_um2=12.6,
            n_pix_min=None, n_pix_max=None, line_threshold_q=70, line_threshold_min=0.1,
            n_pix_max_x=None, n_pix_max_z=None,
            grow_only_local_thresh_pixels=True, use_ch0_stack=True, grow_use_corr_map=False, grow_threshold=None):
        """12.6 um² ~= pi * ((4 um) / 2)² """

        self.cut_x = cut_x
        self.cut_z = cut_z

        self.min_area_um2 = min_area_um2
        self.max_area_um2 = max_area_um2
        self.n_pix_min = n_pix_min
        self.n_pix_max = n_pix_max

        self.n_pix_max_x = n_pix_max_x
        self.n_pix_max_z = n_pix_max_z

        self.line_threshold_q = line_threshold_q
        self.line_threshold_min = line_threshold_min
        self.grow_only_local_thresh_pixels = grow_only_local_thresh_pixels
        self.use_ch0_stack = use_ch0_stack
        self.grow_use_corr_map = grow_use_corr_map
        self.grow_threshold = grow_threshold

    def create_mask_from_data(
            self,
            ch0_stack: np.ndarray,
            ch1_stack: np.ndarray | None = None,
            n_artifact: int = 0,
            pixel_size_um: tuple = (1., 1.),
            plot: bool = False,
            **kwargs,
    ) -> np.ndarray:
        """Create a ROI mask from an imaging stack using the correlation map method.

        Parameters
        ----------
        ch0_stack : np.ndarray
            Primary channel stack of shape (nx, nz, nt).
        ch1_stack : np.ndarray or None, optional
            Secondary channel stack. Used when ``use_ch0_stack=False``.
            Default is ``None``.
        n_artifact : int, optional
            Number of artifact rows to exclude at the top. Default is 0.
        pixel_size_um : tuple, optional
            Pixel size in microns ``(dx, dz)`` used to convert area thresholds
            to pixel counts. Default is ``(1., 1.)``.
        plot : bool, optional
            If ``True``, display diagnostic plots. Default is ``False``.
        **kwargs
            Additional keyword arguments (not used; a warning is emitted if
            any are passed).

        Returns
        -------
        np.ndarray
            Integer 2-D ROI mask of shape (nx, nz).
        """
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
        n_pix_max_x = int(
            np.maximum(np.sqrt(2 * self.max_area_um2), 1)) if self.n_pix_max_x is None else self.n_pix_max_x
        n_pix_max_z = int(
            np.maximum(np.sqrt(2 * self.max_area_um2), 1)) if self.n_pix_max_z is None else self.n_pix_max_z

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


def stack_corr_image(
        stack: np.ndarray,
        cut_x: tuple = (1, 1),
        cut_z: tuple = (1, 1),
) -> np.ndarray:
    """Compute the correlation map for a stack and zero out border regions.

    Parameters
    ----------
    stack : np.ndarray
        3-D array of shape (nx, nz, nt).
    cut_x : tuple, optional
        Number of border pixels to zero in the x direction ``(left, right)``.
        Default is ``(1, 1)``.
    cut_z : tuple, optional
        Number of border pixels to zero in the z direction ``(left, right)``.
        Default is ``(1, 1)``.

    Returns
    -------
    np.ndarray
        2-D correlation map of shape (nx, nz) with borders set to 0.
    """
    corr_map = compute_corr_map(stack)
    corr_map[:cut_x[0], :] = 0
    corr_map[stack.shape[0] - cut_x[1]:, :] = 0
    corr_map[:, :cut_z[0]] = 0
    corr_map[:, stack.shape[1] - cut_z[1]:] = 0
    return corr_map


def stack_corr_line_threshold(
        corr_map: np.ndarray,
        cut_x: tuple = (1, 1),
        cut_z: tuple = (1, 1),
        q: int = 70,
        line_threshold_min: float = 0.1,
) -> np.ndarray:
    """Compute a depth-dependent correlation threshold per z scan line.

    Implements the method from Zhijian Zhao 2020: the threshold for each z
    scan line is defined as the ``q``-th percentile of all local correlation
    values in that line.

    Parameters
    ----------
    corr_map : np.ndarray
        2-D correlation map of shape (nx, nz).
    cut_x : tuple, optional
        Border pixels to exclude from percentile estimation in x.
        Default is ``(1, 1)``.
    cut_z : tuple, optional
        Border pixels to exclude from percentile estimation in z.
        Default is ``(1, 1)``.
    q : int, optional
        Percentile used for the line threshold. Default is 70.
    line_threshold_min : float, optional
        Minimum threshold value to avoid too-low thresholds. Default is 0.1.

    Returns
    -------
    np.ndarray
        1-D array of shape (nz,) with one threshold value per z line.
    """
    nx, nz = corr_map.shape
    line_threshold = np.full(nz, 1.)
    line_threshold[cut_z[0]:nz - cut_z[1]] = np.percentile(
        corr_map[cut_x[0]:nx - cut_x[1], cut_z[0]:nz - cut_z[1]], q=q, axis=0)
    line_threshold[line_threshold < line_threshold_min] = line_threshold_min
    return line_threshold


def corr_map_rois(
        stack: np.ndarray,
        cut_x: tuple,
        cut_z: tuple,
        n_pix_max_x: int = 5,
        n_pix_max_z: int = 5,
        n_pix_max: int = 25,
        n_pix_min: float = 3.,
        only_local_thresh_pixels: bool = False,
        corr_map: np.ndarray | None = None,
        grow_use_corr_map: bool = False,
        line_threshold: np.ndarray | None = None,
        line_threshold_q: int = 70,
        line_threshold_min: float = 0.1,
        grow_threshold: float | None = None,
        plot: bool = False,
) -> np.ndarray:
    """Grow ROIs from seed pixels in a correlation map.

    Implements the algorithm from Zhijian Zhao 2020: for every pixel with
    local correlation above the depth-specific threshold (seed pixel), group
    neighbouring pixels into one ROI and restrict size to match BC axon
    terminal dimensions.

    Parameters
    ----------
    stack : np.ndarray
        3-D imaging stack of shape (nx, nz, nt).
    cut_x : tuple
        Number of border pixels to exclude in x ``(left, right)``.
    cut_z : tuple
        Number of border pixels to exclude in z ``(left, right)``.
    n_pix_max_x : int, optional
        Maximum ROI extent in x (pixels). Default is 5.
    n_pix_max_z : int, optional
        Maximum ROI extent in z (pixels). Default is 5.
    n_pix_max : int, optional
        Maximum total number of pixels per ROI. Default is 25.
    n_pix_min : float, optional
        Minimum number of pixels for a ROI to be kept. Default is 3.
    only_local_thresh_pixels : bool, optional
        If ``True``, only pixels above the threshold can be added during
        growing. Default is ``False``.
    corr_map : np.ndarray or None, optional
        Pre-computed correlation map. Computed from ``stack`` if ``None``.
        Default is ``None``.
    grow_use_corr_map : bool, optional
        If ``True``, use ``corr_map`` values during growing. Default is
        ``False``.
    line_threshold : np.ndarray or None, optional
        Pre-computed per-z-line threshold. Computed if ``None``. Default is
        ``None``.
    line_threshold_q : int, optional
        Percentile for computing the line threshold. Default is 70.
    line_threshold_min : float, optional
        Minimum line threshold. Default is 0.1.
    grow_threshold : float or None, optional
        Threshold used during growing. Uses ``line_threshold`` if ``None``.
        Default is ``None``.
    plot : bool, optional
        If ``True``, display diagnostic plots. Default is ``False``.

    Returns
    -------
    np.ndarray
        Integer 2-D ROI mask of shape (nx, nz).
    """
    if corr_map is None:
        corr_map = stack_corr_image(stack, cut_x=cut_x, cut_z=cut_z)

    if line_threshold is None:
        line_threshold = stack_corr_line_threshold(
            corr_map, cut_x=cut_x, cut_z=cut_z, q=line_threshold_q, line_threshold_min=line_threshold_min)

    nx, nz, nt = stack.shape
    seed_ixs, seed_iys = np.argwhere(corr_map >= line_threshold).T
    # Sort by neighborhood correlation
    sort_idxs = np.argsort(corr_map[seed_ixs, seed_iys])[::-1]
    seed_ixs = seed_ixs[sort_idxs]
    seed_iys = seed_iys[sort_idxs]

    roi_mask = np.zeros((nx, nz), dtype=int)

    roi_id = 1
    ignore_map = np.zeros(corr_map.shape, dtype=bool)
    if only_local_thresh_pixels:
        ignore_map[corr_map < line_threshold] = True
    ignore_map[:cut_x[0], :] = True
    ignore_map[nx - cut_x[1]:, :] = True
    ignore_map[:, :cut_z[0]] = True
    ignore_map[:, nz - cut_z[1]:] = True

    p_threshold = grow_threshold if grow_threshold is not None else line_threshold

    for seed_ix, seed_iz in zip(seed_ixs, seed_iys):
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
        fig, axs = plt.subplots(2, 6, figsize=(15, 3), height_ratios=(8, 1))

        mean = np.mean(stack, axis=2)
        mean[:cut_x[0], :] = np.nan

        seed_pixels = (corr_map >= line_threshold)

        corr_map_nan = corr_map.copy()
        corr_map_nan[~seed_pixels] = np.nan

        corr_min = np.minimum(0., np.nanmin(corr_map))
        corr_max = np.nanmax(corr_map)

        im = axs[0, 0].imshow(mean.T, cmap='gray', origin='lower')
        axs[0, 0].set_title('Mean projection')
        plt.colorbar(im, ax=axs[0, 0], cax=axs[1, 0], orientation='horizontal')

        axs[0, 1].imshow(mean.T, cmap='gray', origin='lower')
        im = axs[0, 1].imshow(corr_map_nan.T, interpolation='none',
                              cmap='Reds', origin='lower', alpha=0.7, vmin=corr_min, vmax=corr_max, zorder=100)
        axs[0, 1].set_title('< Overlay >')
        plt.colorbar(im, ax=axs[0, 1], cax=axs[1, 1], orientation='horizontal')

        im = axs[0, 2].imshow(corr_map.T, cmap='Reds', origin='lower', vmin=corr_min, vmax=corr_max,
                              interpolation='none')
        axs[0, 2].set_title('Correlation map')
        plt.colorbar(im, ax=axs[0, 2], cax=axs[1, 2], orientation='horizontal')

        axs[0, 3].plot(line_threshold, np.arange(line_threshold.size), '.-')
        axs[0, 3].axvline(line_threshold_min, c='k', ls='--')
        axs[0, 3].set_title('Line threshold')
        axs[0, 3].set_xlim(-0.03, 1.03)
        axs[0, 3].set_ylim(0, line_threshold.size - 1)
        axs[1, 3].axis('off')

        im = axs[0, 4].imshow(corr_map_nan.T, cmap='viridis', origin='lower', interpolation='none')
        axs[0, 4].set_title('Seed pixels')
        plt.colorbar(im, ax=axs[0, 4], cax=axs[1, 4], orientation='horizontal')

        im = axs[0, 5].imshow(roi_mask.T, cmap='jet', origin='lower', interpolation='none')
        axs[0, 5].set_title('ROI mask')
        plt.colorbar(im, ax=axs[0, 5], cax=axs[1, 5], orientation='horizontal')

        plt.tight_layout()
        plt.show()

    return roi_mask


def corr_map_grow_roi(
        seed_ix: int,
        seed_iz: int,
        n_pix_max: int,
        n_pix_max_x: int,
        n_pix_max_z: int,
        p_threshold: np.ndarray | float,
        stack: np.ndarray,
        ignore_map: np.ndarray | None = None,
        corr_map: np.ndarray | None = None,
) -> list:
    """Grow a single ROI from a seed pixel by collecting correlated neighbours.

    Starting from the seed pixel, the algorithm performs a breadth-first
    expansion (sorted by correlation) until the pixel count or spatial extent
    limits are reached.

    Parameters
    ----------
    seed_ix : int
        x-index of the seed pixel.
    seed_iz : int
        z-index of the seed pixel.
    n_pix_max : int
        Maximum total pixels in the ROI.
    n_pix_max_x : int
        Maximum extent of the ROI along the x axis in pixels.
    n_pix_max_z : int
        Maximum extent of the ROI along the z axis in pixels.
    p_threshold : np.ndarray or float
        Per-z-line correlation threshold (shape ``(nz,)``) or a scalar.
        Pixels below the threshold are not added to the ROI.
    stack : np.ndarray
        3-D imaging stack of shape (nx, nz, nt) used for pairwise correlation
        when ``corr_map`` is ``None``.
    ignore_map : np.ndarray or None, optional
        Boolean 2-D mask of shape (nx, nz); ``True`` pixels are skipped.
        Default is ``None``.
    corr_map : np.ndarray or None, optional
        Pre-computed 2-D correlation map. If provided, its values are used
        instead of computing pairwise correlations on the fly. Default is
        ``None``.

    Returns
    -------
    list
        List of ``(ix, iz)`` tuples for each pixel belonging to the ROI.
    """
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
