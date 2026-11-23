import matplotlib

matplotlib.use('Agg')

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip('cv2')
pytest.importorskip('skimage')

from djimaging.utils.receptive_fields.srf_contour_utils import (  # noqa: E402
    compute_closed_cntrs_and_sizes, compute_cntr_center, compute_cntr_sizes, compute_contour,
    compute_irregular_index, compute_largest_cntr_and_size, create_disk_mask, get_center_and_surround_masks,
    get_center_mask, get_surround_mask)


def _cone(shape, row0, col0, radius):
    """Linear cone: the level-0.5 contour is a circle with radius `radius / 2` pixels around (row0, col0)."""
    yy, xx = np.mgrid[:shape[0], :shape[1]].astype(float)
    return 1. - np.sqrt((yy - row0) ** 2 + (xx - col0) ** 2) / radius


def _square(x0, y0, side):
    return np.array([[x0, y0], [x0 + side, y0], [x0 + side, y0 + side], [x0, y0 + side], [x0, y0]], dtype=float)


# --- contour primitives ---

def test_cntr_sizes_and_area_factor():
    sizes = compute_cntr_sizes([_square(0, 0, 2), _square(5, 5, 3)], area_factor=10.)
    assert sizes == pytest.approx([40., 90.])


def test_closed_cntrs_filter_open_contours():
    closed = _square(0, 0, 2)
    open_ = closed[:-1]
    cntrs, sizes = compute_closed_cntrs_and_sizes([closed, open_])
    assert len(cntrs) == 1
    assert cntrs[0] is closed
    assert sizes == pytest.approx([4.])


def test_largest_cntr():
    cntr, size = compute_largest_cntr_and_size([_square(0, 0, 2), _square(0, 0, 3), _square(0, 0, 1)])
    assert size == pytest.approx(9.)
    np.testing.assert_array_equal(cntr, _square(0, 0, 3))


def test_cntr_center():
    x, y = compute_cntr_center(_square(-1., 3., 4.))
    assert (x, y) == pytest.approx((1., 5.))


def test_irregular_index():
    assert compute_irregular_index(_square(0, 0, 2)) == pytest.approx(0.)
    l_shape = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2], [0, 0]], dtype=float)
    assert compute_irregular_index(l_shape) == pytest.approx(1 - 3 / 3.5)
    assert compute_irregular_index(np.zeros((0, 2))) == 1.


# --- compute_contour ---

def test_contour_no_crossing():
    srf = _cone((11, 11), 5, 5, 6.)
    assert compute_contour(srf, level=2., pixel_size_um=10.) == ([], [])
    assert compute_contour(srf, level=-100., pixel_size_um=10.) == ([], [])


def test_contour_area_in_um2_scales_with_pixel_size():
    srf = _cone((21, 31), 10, 15, 8.)
    _, sizes1 = compute_contour(srf, level=0.5, pixel_size_um=1.)
    _, sizes10 = compute_contour(srf, level=0.5, pixel_size_um=10.)
    assert len(sizes1) == 1
    assert sizes10[0] == pytest.approx(100 * sizes1[0])


def test_contour_centered_rf_has_zero_center():
    srf = _cone((21, 31), 10, 15, 8.)
    cntrs, _ = compute_contour(srf, level=0.5, pixel_size_um=10.)
    assert compute_cntr_center(cntrs[0]) == pytest.approx((0., 0.), abs=1e-6)


def test_contour_two_blobs_and_plot():
    srf = np.clip(_cone((21, 41), 10, 10, 6.), 0, None) + np.clip(_cone((21, 41), 10, 30, 8.), 0, None)
    cntrs, sizes = compute_contour(srf, level=0.5, pixel_size_um=5., plot=True)
    assert len(cntrs) == 2
    # The left blob (x < 0) is smaller than the right blob
    centers_x = [compute_cntr_center(c)[0] for c in cntrs]
    assert sizes[int(np.argmin(centers_x))] < sizes[int(np.argmax(centers_x))]


@pytest.mark.xfail(strict=True, reason="BUG: compute_contour passes pixel-edge extent to ax.contour (point positions), "
                                       "inflating coordinates by n/(n-1)")
@pytest.mark.parametrize('shape', [(15, 15), (11, 31), (41, 61)])
def test_contour_area_independent_of_array_size(shape):
    pixel_size_um = 10.
    srf = _cone(shape, shape[0] // 2, shape[1] // 2, 8.)  # circle of radius 4 px at level 0.5
    _, sizes = compute_contour(srf, level=0.5, pixel_size_um=pixel_size_um)
    # Marching squares slightly underestimates the circle (~1%)
    assert sizes[0] == pytest.approx(np.pi * (4 * pixel_size_um) ** 2, rel=0.02)


@pytest.mark.xfail(strict=True, reason="BUG: compute_contour coordinates are scaled by n/(n-1) w.r.t. stimulus center")
def test_contour_center_matches_gauss_fit_convention():
    # RfOffsetTemplate: offset = ((x_mean + 0.5) - nx / 2) * pix_size for Gaussian fits.
    # The contour-based offset must use the same convention.
    shape, row0, col0, pixel_size_um = (15, 25), 4, 17, 10.
    srf = _cone(shape, row0, col0, 6.)
    cntrs, _ = compute_contour(srf, level=0.5, pixel_size_um=pixel_size_um)
    x, y = compute_cntr_center(cntrs[0])
    assert x == pytest.approx(((col0 + 0.5) - shape[1] / 2) * pixel_size_um, abs=0.5)
    assert y == pytest.approx(((row0 + 0.5) - shape[0] / 2) * pixel_size_um, abs=0.5)


# --- masks ---

def test_create_disk_mask():
    mask = create_disk_mask(2)
    assert mask.shape == (5, 5)
    assert mask[2, 2]
    np.testing.assert_array_equal(mask, mask.T)
    np.testing.assert_array_equal(mask, mask[::-1])
    rr, cc = np.nonzero(mask)
    # skimage.draw.disk is strict: pixels at exactly `dist` are excluded
    assert np.sqrt((rr - 2) ** 2 + (cc - 2) ** 2).max() < 2


def test_surround_mask_ring():
    mask_center = np.zeros((21, 21), dtype=bool)
    mask_center[10, 10] = True
    pixelsize = 10.
    mask_surround, mask_full_surround = get_surround_mask(mask_center, pixelsize, d_inner=20., d_outer=50.)

    rr, cc = np.nonzero(mask_surround)
    dist = np.sqrt((rr - 10) ** 2 + (cc - 10) ** 2)
    assert dist.min() >= 2 and dist.max() < 5
    assert not np.any(mask_surround & mask_center)
    assert not mask_full_surround[10, 10]
    assert mask_full_surround[0, 0]
    # Ring is contained in the full surround
    assert np.all(mask_full_surround[mask_surround])


def test_surround_mask_without_inner_border():
    mask_center = np.zeros((11, 11), dtype=bool)
    mask_center[4:7, 4:7] = True
    mask_surround, mask_full_surround = get_surround_mask(mask_center, 1., d_inner=0, d_outer=2.)
    np.testing.assert_array_equal(mask_full_surround, ~mask_center)
    assert not np.any(mask_surround & mask_center)
    assert mask_surround[3, 5] and mask_surround[7, 5] and mask_surround[5, 3] and mask_surround[5, 7]
    assert not mask_surround[2, 5]  # distance 2 is excluded (strict disk)


def test_center_and_surround_masks_are_disjoint():
    shape, pixelsize = (21, 25), 10.
    srf = _cone(shape, 10, 12, 8.)
    cntrs, _ = compute_contour(srf, level=0.5, pixel_size_um=pixelsize)
    mask_center, mask_surround, mask_full_surround = get_center_and_surround_masks(
        cntrs[0], shape, pixelsize, d_inner=10., d_outer=50.)
    for mask in (mask_center, mask_surround, mask_full_surround):
        assert mask.shape == shape
    assert mask_center.sum() > 0 and mask_surround.sum() > 0
    assert not np.any(mask_center & mask_surround)
    assert not np.any(mask_center & mask_full_surround)


@pytest.mark.xfail(strict=True, reason="BUG: get_center_mask maps contour to pixels with +n/2 instead of +n/2-0.5 "
                                       "(and compute_contour scales by n/(n-1)), shifting the mask")
def test_center_mask_from_contour_is_centered_on_rf():
    shape, row0, col0, pixelsize = (15, 25), 4, 17, 10.
    srf = _cone(shape, row0, col0, 6.4)  # radius 3.2 px: no pixel center lies on the contour
    cntrs, _ = compute_contour(srf, level=0.5, pixel_size_um=pixelsize)
    mask = get_center_mask(cntrs[0], shape, pixelsize)
    rr, cc = np.nonzero(mask)
    assert rr.mean() == pytest.approx(row0, abs=0.15)
    assert cc.mean() == pytest.approx(col0, abs=0.15)
    np.testing.assert_array_equal(mask, srf > 0.5)


@pytest.mark.xfail(strict=True, reason="BUG: get_center_mask shifts the mask by half a pixel (+n/2 instead of +n/2-0.5)")
def test_center_mask_of_centered_square():
    # Square of 3x3 pixels around the center pixel of a 9x9 sRF, in um relative to the sRF center.
    pixelsize = 10.
    cntr = _square(-1.4 * pixelsize, -1.4 * pixelsize, 2.8 * pixelsize)
    mask = get_center_mask(cntr, (9, 9), pixelsize)
    expected = np.zeros((9, 9), dtype=bool)
    expected[3:6, 3:6] = True
    np.testing.assert_array_equal(mask, expected)


def test_center_mask_plot_smoke():
    cntr = _square(-20., -20., 40.)
    mask = get_center_mask(cntr, (9, 9), 10., plot=True)
    assert mask.dtype == bool and mask.sum() > 0


def test_plot_smoke():
    from matplotlib import pyplot as plt
    from djimaging.utils.receptive_fields.srf_contour_utils import plot_center_and_surround_masks

    srf = _cone((15, 15), 7, 7, 6.)
    for level in (0.5, 5.):  # one contour / no contour
        compute_contour(srf, level=level, pixel_size_um=10., plot=True)
    mask_center = srf > 0.5
    mask_surround, mask_full_surround = get_surround_mask(mask_center, 10., d_inner=10., d_outer=30., plot=True)
    axs = plot_center_and_surround_masks(mask_center, mask_surround, mask_full_surround, 10.)
    assert len(axs) == 4
    plt.close('all')
