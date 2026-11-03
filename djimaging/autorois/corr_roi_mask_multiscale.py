import heapq
import warnings

import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import relabel_sequential, watershed

from djimaging.autorois.corr_roi_mask_utils import stack_corr_line_threshold, get_roi_pixel_limits

DEFAULT_SHIFT_WEIGHTS = {1: 0.6, 2: 0.3, 3: 0.1}


class CorrRoiMaskMultiscale:
    """ROI mask creator based on a multi-scale local correlation map and adaptive seeding.

    Developed for photoreceptors. Compared to :class:`CorrRoiMask`, the correlation map combines several pixel
    shifts, seeds are local maxima that stand out from their neighbourhood (instead of all pixels above a
    per-line threshold), and ROIs are either grown from the seeds on the correlation map (default) or obtained
    with a watershed on the correlation map (``use_watershed=True``).

    Parameters
    ----------
    cut_x, cut_z : tuple, optional
        Number of pixels to exclude at the (start, end) of the x / z axis. Default is ``(1, 1)``.
    min_area_um2, max_area_um2 : float, optional
        Minimum and maximum ROI area in µm². Default is 0.5 and 20.
    n_pix_min, n_pix_max : int or None, optional
        Hard override for the minimum / maximum pixel count per ROI. In watershed mode, ``n_pix_min`` is the
        minimum size of a ROI and ``n_pix_max`` and the extent limits are not applied.
    n_pix_max_x, n_pix_max_z : int or None, optional
        Maximum ROI extent in pixels. Default is ``None`` (derived from ``max_area_um2``).
    line_threshold_q, line_threshold_min : optional
        Percentile and minimum of the depth-dependent line threshold. In watershed mode, only pixels above the
        line threshold can be part of a ROI. It is not used otherwise.
    use_ch0_stack : bool, optional
        If ``True``, use channel 0; otherwise use channel 1. Default is ``True``.
    shift_weights : dict or None, optional
        Weight of the correlation at each pixel shift. Default is ``{1: 0.6, 2: 0.3, 3: 0.1}``.
    smooth_sigma : float, optional
        Sigma of the gaussian filter applied to the correlation map of each shift. Default is 1.
    clip_corr_map : bool, optional
        If ``True``, clip the normalised correlation map to [0, 1]. Default is ``False``.
    seed_window : int, optional
        Window size in pixels for the local statistics of the seed threshold. Default is 15.
    seed_std_factor : float, optional
        Seeds must be above local mean + ``seed_std_factor`` * local std. Default is 1.
    seed_min_distance : int, optional
        Minimum distance between seeds in pixels. Default is 3.
    seed_global_percentile : float or None, optional
        If given, seeds must also be above this percentile of the correlation map. Default is ``None``.
    seed_min_absolute : float, optional
        Minimum correlation of a seed, only used together with ``seed_global_percentile``. Default is 0.05.
    grow_threshold : float or None, optional
        Correlation (map value) that pixels need to be added to a ROI. If ``None``, the ``grow_percentile``
        of the correlation map is used. Not used in watershed mode.
    grow_percentile : float, optional
        See ``grow_threshold``. Default is 65.
    use_watershed : bool, optional
        If ``True``, use a watershed on the correlation map instead of growing. Default is ``False``.
    merge_distance : float, optional
        Watershed mode only. Regions smaller than ``n_pix_min`` are merged into the nearest larger region if the
        centroids are not further apart (in pixels), otherwise they are dropped. Default is 3.
    """

    def __init__(
            self, cut_x=(1, 1), cut_z=(1, 1), min_area_um2=0.5, max_area_um2=20,
            n_pix_min=None, n_pix_max=None, n_pix_max_x=None, n_pix_max_z=None,
            line_threshold_q=70, line_threshold_min=0.1, use_ch0_stack=True,
            shift_weights=None, smooth_sigma=1., clip_corr_map=False,
            seed_window=15, seed_std_factor=1., seed_min_distance=3,
            seed_global_percentile=None, seed_min_absolute=0.05,
            grow_threshold=None, grow_percentile=65,
            use_watershed=False, merge_distance=3):
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
        self.use_ch0_stack = use_ch0_stack

        self.shift_weights = shift_weights
        self.smooth_sigma = smooth_sigma
        self.clip_corr_map = clip_corr_map

        self.seed_window = seed_window
        self.seed_std_factor = seed_std_factor
        self.seed_min_distance = seed_min_distance
        self.seed_global_percentile = seed_global_percentile
        self.seed_min_absolute = seed_min_absolute

        self.grow_threshold = grow_threshold
        self.grow_percentile = grow_percentile
        self.use_watershed = use_watershed
        self.merge_distance = merge_distance

    def create_mask_from_data(
            self,
            ch0_stack: np.ndarray,
            ch1_stack: np.ndarray | None = None,
            n_artifact: int = 0,
            pixel_size_um: tuple = (1., 1.),
            plot: bool = False,
            **kwargs,
    ) -> np.ndarray:
        """Create a ROI mask from an imaging stack.

        Parameters
        ----------
        ch0_stack : np.ndarray
            Primary channel stack of shape (nx, nz, nt).
        ch1_stack : np.ndarray or None, optional
            Secondary channel stack. Used when ``use_ch0_stack=False``.
        n_artifact : int, optional
            Number of artifact rows to exclude at the start of x. Default is 0.
        pixel_size_um : tuple, optional
            Pixel size in microns ``(dx, dz)`` used to convert area thresholds to pixel counts.
        plot : bool, optional
            If ``True``, display diagnostic plots. Default is ``False``.
        **kwargs
            Not used; a warning is emitted if any are passed.

        Returns
        -------
        np.ndarray
            Integer 2-D ROI mask of shape (nx, nz).
        """
        if len(kwargs) > 0:
            warnings.warn(f'ignoring kwargs: {kwargs}')

        cut_x = (max(self.cut_x[0], n_artifact), self.cut_x[1])

        stack = ch0_stack if self.use_ch0_stack else ch1_stack
        if stack is None:
            raise ValueError("ch1_stack must be provided when use_ch0_stack=False")

        n_pix_min, n_pix_max, n_pix_max_x, n_pix_max_z = get_roi_pixel_limits(
            self.min_area_um2, self.max_area_um2, pixel_size_um,
            n_pix_min=self.n_pix_min, n_pix_max=self.n_pix_max,
            n_pix_max_x=self.n_pix_max_x, n_pix_max_z=self.n_pix_max_z)

        print(f'Creating ROI mask for stack of shape={stack.shape}, '
              f'with n_pix_min={n_pix_min}, n_pix_max={n_pix_max}, '
              f'n_pix_max_x={n_pix_max_x}, n_pix_max_z={n_pix_max_z}, '
              f'pixel_size_um={pixel_size_um}')

        corr_map = compute_corr_map_multiscale(
            stack, weights=self.shift_weights, smooth_sigma=self.smooth_sigma, clip=self.clip_corr_map)

        line_threshold = stack_corr_line_threshold(
            corr_map, cut_x=cut_x, cut_z=self.cut_z, q=self.line_threshold_q,
            line_threshold_min=self.line_threshold_min)

        seed_ixs, seed_izs = find_seed_pixels(
            corr_map, window=self.seed_window, std_factor=self.seed_std_factor, min_distance=self.seed_min_distance,
            global_percentile=self.seed_global_percentile, min_absolute=self.seed_min_absolute)

        valid = np.ones(corr_map.shape, dtype=bool)
        valid[:cut_x[0], :] = False
        valid[corr_map.shape[0] - cut_x[1]:, :] = False
        valid[:, :self.cut_z[0]] = False
        valid[:, corr_map.shape[1] - self.cut_z[1]:] = False

        if self.use_watershed:
            roi_mask = watershed_rois(
                corr_map, seed_ixs, seed_izs, mask=valid & (corr_map >= line_threshold),
                n_pix_min=n_pix_min, merge_distance=self.merge_distance)
        else:
            grow_threshold = self.grow_threshold
            if grow_threshold is None:
                grow_threshold = np.percentile(corr_map, self.grow_percentile)
            roi_mask = grow_rois(
                corr_map, seed_ixs, seed_izs, ignore_map=~valid, p_threshold=grow_threshold,
                n_pix_min=n_pix_min, n_pix_max=n_pix_max, n_pix_max_x=n_pix_max_x, n_pix_max_z=n_pix_max_z)

        if plot:
            plot_corr_roi_mask(stack, corr_map, line_threshold, seed_ixs, seed_izs, roi_mask,
                               cut_x=cut_x, line_threshold_min=self.line_threshold_min)

        return roi_mask


def compute_corr_map_multiscale(
        stack: np.ndarray,
        weights: dict | None = None,
        smooth_sigma: float = 1.,
        normalize_each_scale: bool = True,
        clip: bool = False,
) -> np.ndarray:
    """Local correlation map combining the correlation to neighbours at several pixel shifts.

    For each shift ``d`` in ``weights``, the correlation of every pixel to the pixel ``d`` steps away in the
    four directions (x, z and both diagonals) is averaged, optionally normalised to the 1st-99th percentile
    and smoothed. The maps of all shifts are summed with the given weights and normalised again.

    Parameters
    ----------
    stack : np.ndarray
        3-D array of shape (nx, nz, nt).
    weights : dict or None, optional
        Mapping from pixel shift to weight. Default is ``{1: 0.6, 2: 0.3, 3: 0.1}``.
    smooth_sigma : float, optional
        Sigma of the gaussian filter applied to each shift's map. No smoothing if 0 or None.
    normalize_each_scale : bool, optional
        Normalise the map of each shift to the 1st-99th percentile, so that larger shifts don't dominate.
    clip : bool, optional
        Clip the final map to [0, 1]. Default is ``False``.

    Returns
    -------
    np.ndarray
        2-D map of shape (nx, nz), normalised so that the 1st-99th percentile spans [0, 1].
    """
    if weights is None:
        weights = DEFAULT_SHIFT_WEIGHTS

    stack = np.asarray(stack, dtype=float)
    stack_z = (stack - stack.mean(axis=2, keepdims=True)) / (stack.std(axis=2, keepdims=True) + 1e-9)
    nx, nz, _ = stack_z.shape
    multi_map = np.zeros((nx, nz), dtype=float)

    for d, weight in weights.items():
        if d >= min(nx, nz):
            continue

        # Each pair correlation is assigned to the pixel that has a neighbour d steps before it
        corr_d = np.zeros((nx, nz), dtype=float)
        count = np.zeros((nx, nz), dtype=int)
        for sl_to, sl_a, sl_b in [
            ((slice(d, None), slice(None)), (slice(d, None), slice(None)), (slice(None, -d), slice(None))),
            ((slice(None), slice(d, None)), (slice(None), slice(d, None)), (slice(None), slice(None, -d))),
            ((slice(d, None), slice(d, None)), (slice(d, None), slice(d, None)), (slice(None, -d), slice(None, -d))),
            ((slice(d, None), slice(None, -d)), (slice(d, None), slice(None, -d)), (slice(None, -d), slice(d, None))),
        ]:
            corr_d[sl_to] += np.mean(stack_z[sl_a] * stack_z[sl_b], axis=2)
            count[sl_to] += 1
        corr_d /= np.maximum(count, 1)

        if normalize_each_scale:
            lo, hi = np.percentile(corr_d, [1, 99])
            corr_d = (corr_d - lo) / (hi - lo + 1e-9)

        if smooth_sigma:
            corr_d = ndimage.gaussian_filter(corr_d, smooth_sigma)

        multi_map += weight * corr_d

    lo, hi = np.percentile(multi_map, [1, 99])
    multi_map = (multi_map - lo) / (hi - lo + 1e-9)
    if clip:
        multi_map = np.clip(multi_map, 0., 1.)
    return multi_map


def find_seed_pixels(
        corr_map: np.ndarray,
        window: int = 15,
        std_factor: float = 1.,
        min_distance: int = 3,
        global_percentile: float | None = None,
        min_absolute: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Find seed pixels as local maxima of the correlation map that stand out from their neighbourhood.

    A pixel is a candidate if it is above ``local mean + std_factor * local std`` (over a ``window`` x ``window``
    neighbourhood) and, if ``global_percentile`` is given, above that percentile of the map (but at least
    ``min_absolute``).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        x and z indices of the seeds, sorted by decreasing correlation.
    """
    mean_local = ndimage.uniform_filter(corr_map, size=window)
    var_local = np.maximum(ndimage.uniform_filter(corr_map ** 2, size=window) - mean_local ** 2, 0)
    candidates = corr_map > mean_local + std_factor * np.sqrt(var_local)

    if global_percentile is not None:
        candidates &= corr_map >= max(np.percentile(corr_map, global_percentile), min_absolute)

    coords = peak_local_max(corr_map, min_distance=min_distance, threshold_abs=0, labels=candidates)
    if len(coords) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    coords = coords[np.argsort(corr_map[coords[:, 0], coords[:, 1]])[::-1]]
    return coords[:, 0], coords[:, 1]


def grow_rois(
        corr_map: np.ndarray,
        seed_ixs: np.ndarray,
        seed_izs: np.ndarray,
        ignore_map: np.ndarray,
        p_threshold: float | np.ndarray,
        n_pix_min: int,
        n_pix_max: int,
        n_pix_max_x: int,
        n_pix_max_z: int,
) -> np.ndarray:
    """Grow one ROI per seed (in the given order) and keep those with at least ``n_pix_min`` pixels."""
    roi_mask = np.zeros(corr_map.shape, dtype=int)
    ignore_map = ignore_map.copy()
    roi_id = 1

    for seed_ix, seed_iz in zip(seed_ixs, seed_izs):
        if roi_mask[seed_ix, seed_iz] != 0:
            continue

        roi_pixels = grow_roi(
            corr_map, seed_ix, seed_iz, p_threshold=p_threshold, ignore_map=ignore_map,
            n_pix_max=n_pix_max, n_pix_max_x=n_pix_max_x, n_pix_max_z=n_pix_max_z)

        if len(roi_pixels) >= n_pix_min:
            roi_ixs, roi_izs = np.array(roi_pixels).T
            roi_mask[roi_ixs, roi_izs] = roi_id
            ignore_map[roi_ixs, roi_izs] = True
            roi_id += 1

    return roi_mask


def grow_roi(
        corr_map: np.ndarray,
        seed_ix: int,
        seed_iz: int,
        p_threshold: float | np.ndarray,
        ignore_map: np.ndarray,
        n_pix_max: int,
        n_pix_max_x: int,
        n_pix_max_z: int,
) -> list:
    """Grow a ROI from a seed by repeatedly adding the neighbour with the highest correlation map value.

    Pixels with a correlation map value below ``p_threshold`` (scalar or one value per z line) are not added.
    Growth stops at ``n_pix_max`` pixels, and a ROI can not extend over more than ``n_pix_max_x`` / ``n_pix_max_z``
    pixels.

    Returns
    -------
    list
        ``(ix, iz)`` tuples of the pixels of the ROI.
    """
    nx, nz = corr_map.shape
    p_threshold = np.broadcast_to(np.asarray(p_threshold, dtype=float), (nz,))

    visited = ignore_map.copy()
    queued = ignore_map.copy()
    queued[seed_ix, seed_iz] = True
    heap = [(-1., seed_ix, seed_iz)]

    roi_pixels = []
    min_x = max_x = seed_ix
    min_z = max_z = seed_iz
    reached_x_max = reached_z_max = False

    while heap and len(roi_pixels) < n_pix_max:
        neg_corr, ix, iz = heapq.heappop(heap)
        if visited[ix, iz]:
            continue
        visited[ix, iz] = True

        if -neg_corr < p_threshold[iz]:
            continue

        roi_pixels.append((ix, iz))

        if not reached_x_max:
            min_x, max_x = min(ix, min_x), max(ix, max_x)
            if max_x - min_x >= n_pix_max_x - 1:
                visited[:min_x, :] = visited[max_x + 1:, :] = True
                queued[:min_x, :] = queued[max_x + 1:, :] = True
                reached_x_max = True

        if not reached_z_max:
            min_z, max_z = min(iz, min_z), max(iz, max_z)
            if max_z - min_z >= n_pix_max_z - 1:
                visited[:, :min_z] = visited[:, max_z + 1:] = True
                queued[:, :min_z] = queued[:, max_z + 1:] = True
                reached_z_max = True

        for new_ix, new_iz in [(ix - 1, iz), (ix + 1, iz), (ix, iz - 1), (ix, iz + 1)]:
            if 0 <= new_ix < nx and 0 <= new_iz < nz and not queued[new_ix, new_iz]:
                queued[new_ix, new_iz] = True
                heapq.heappush(heap, (-corr_map[new_ix, new_iz], new_ix, new_iz))

    return roi_pixels


def watershed_rois(
        corr_map: np.ndarray,
        seed_ixs: np.ndarray,
        seed_izs: np.ndarray,
        mask: np.ndarray,
        n_pix_min: int,
        merge_distance: float = 3,
        n_pix_min_basin: int = 2,
        n_pix_max_basin: int = 2000,
) -> np.ndarray:
    """Create ROIs by a watershed on the (inverted) correlation map, flooding from the seeds within ``mask``.

    Basins outside of ``[n_pix_min_basin, n_pix_max_basin]`` pixels are dropped. Remaining ROIs smaller than
    ``n_pix_min`` are merged into the nearest larger ROI (within ``merge_distance``) or dropped.
    """
    if len(seed_ixs) == 0:
        return np.zeros(corr_map.shape, dtype=int)

    markers = np.zeros(corr_map.shape, dtype=int)
    markers[seed_ixs, seed_izs] = np.arange(1, len(seed_ixs) + 1)

    labels = watershed(-corr_map, markers, mask=mask)
    labels = _filter_labels_by_area(labels, n_pix_min_basin, n_pix_max_basin)
    labels = _merge_small_rois(labels, n_pix_min=n_pix_min, merge_distance=merge_distance)
    return _filter_labels_by_area(labels, n_pix_min, None)


def _filter_labels_by_area(labels: np.ndarray, min_area: int, max_area: int | None) -> np.ndarray:
    """Remove labels with an area outside [min_area, max_area] and relabel compactly (1..n)."""
    areas = np.bincount(labels.ravel())
    drop = areas < min_area
    if max_area is not None:
        drop |= areas > max_area
    drop[0] = False
    labels = np.where(drop[labels], 0, labels)
    return relabel_sequential(labels)[0].astype(int)


def _merge_small_rois(labels: np.ndarray, n_pix_min: int, merge_distance: float) -> np.ndarray:
    """Merge ROIs with fewer than ``n_pix_min`` pixels into the nearest larger ROI if close, else remove them."""
    ids = np.unique(labels)
    ids = ids[ids != 0]
    if len(ids) == 0:
        return labels

    areas = ndimage.sum(np.ones_like(labels), labels, ids)
    centroids = np.array(ndimage.center_of_mass(np.ones_like(labels), labels, ids))
    is_small = areas < n_pix_min
    if not is_small.any():
        return labels

    new_ids = {}
    for i in np.flatnonzero(is_small):
        new_ids[ids[i]] = 0
        if (~is_small).any():
            dists = np.hypot(*(centroids[~is_small] - centroids[i]).T)
            j = np.argmin(dists)
            if dists[j] <= merge_distance:
                new_ids[ids[i]] = ids[~is_small][j]

    out = labels.copy()
    for old_id, new_id in new_ids.items():
        out[labels == old_id] = new_id
    return relabel_sequential(out)[0].astype(int)


def plot_corr_roi_mask(stack, corr_map, line_threshold, seed_ixs, seed_izs, roi_mask, cut_x=(1, 1),
                       line_threshold_min=0.1):
    """Plot mean projection, correlation map with seeds, line threshold and the resulting ROI mask."""
    mean = np.mean(stack, axis=2)
    mean[:cut_x[0], :] = np.nan

    fig, axs = plt.subplots(2, 5, figsize=(13, 3), height_ratios=(8, 1))

    im = axs[0, 0].imshow(mean.T, cmap='gray', origin='lower')
    axs[0, 0].set_title('Mean projection')
    plt.colorbar(im, ax=axs[0, 0], cax=axs[1, 0], orientation='horizontal')

    im = axs[0, 1].imshow(corr_map.T, cmap='Reds', origin='lower', interpolation='none')
    axs[0, 1].set_title('Correlation map')
    plt.colorbar(im, ax=axs[0, 1], cax=axs[1, 1], orientation='horizontal')

    axs[0, 2].imshow(corr_map.T, cmap='Reds', origin='lower', interpolation='none')
    axs[0, 2].plot(seed_ixs, seed_izs, 'k.', ms=3)
    axs[0, 2].set_title(f'Seeds (n={len(seed_ixs)})')
    axs[1, 2].axis('off')

    axs[0, 3].plot(line_threshold, np.arange(line_threshold.size), '.-')
    axs[0, 3].axvline(line_threshold_min, c='k', ls='--')
    axs[0, 3].set_title('Line threshold')
    axs[0, 3].set_ylim(0, line_threshold.size - 1)
    axs[1, 3].axis('off')

    im = axs[0, 4].imshow(np.ma.masked_equal(roi_mask, 0).T, cmap='jet', origin='lower', interpolation='none')
    axs[0, 4].set_title(f'ROI mask (n={roi_mask.max()})')
    plt.colorbar(im, ax=axs[0, 4], cax=axs[1, 4], orientation='horizontal')

    plt.tight_layout()
    plt.show()
