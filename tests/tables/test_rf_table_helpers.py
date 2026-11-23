"""Module-level helpers of djimaging.tables.receptivefield that do not need a database."""
import matplotlib

matplotlib.use('Agg')

from types import SimpleNamespace  # noqa: E402

import datajoint as dj  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip('cv2')

from djimaging.tables.receptivefield.rf_offset import _format_affine, fit_affine_roi_to_rf  # noqa: E402
from djimaging.tables.receptivefield.split_strf import SplitRFTemplate, fetch1_rf_time  # noqa: E402
from djimaging.tables.receptivefield.srf_contour import _compute_contours  # noqa: E402
from djimaging.utils.receptive_fields.split_rf_utils import merge_strf  # noqa: E402


def _rot(deg):
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


# --- fit_affine_roi_to_rf ---

def test_affine_fit_recovers_rotation_and_scale_with_group_intercepts():
    rng = np.random.default_rng(0)
    A_true = _rot(20.) @ np.diag([1.5, 0.8])
    roi_xy = rng.uniform(-50, 50, size=(60, 2))
    groups = np.repeat([0, 1, 2], 20)
    intercepts = np.array([[10., -5.], [-30., 7.], [0., 100.]])
    rf_xy = roi_xy @ A_true.T + intercepts[groups]

    fit = fit_affine_roi_to_rf(roi_xy, rf_xy, groups=groups)

    np.testing.assert_allclose(fit['A'], A_true, atol=1e-10)
    assert fit['r2'] == pytest.approx(1.)
    assert fit['n'] == 60 and fit['n_groups'] == 3
    assert not fit['reflection']
    # Polar decomposition A = R @ S
    S = np.array([[fit['scale_x'], fit['shear']], [fit['shear'], fit['scale_y']]])
    np.testing.assert_allclose(_rot(fit['rotation_deg']) @ S, A_true, atol=1e-10)


def test_affine_fit_pure_rotation():
    rng = np.random.default_rng(1)
    roi_xy = rng.normal(size=(30, 2))
    fit = fit_affine_roi_to_rf(roi_xy, roi_xy @ _rot(-35.).T + 3.)
    assert fit['rotation_deg'] == pytest.approx(-35.)
    assert fit['scale_x'] == pytest.approx(1.) and fit['scale_y'] == pytest.approx(1.)
    assert fit['shear'] == pytest.approx(0., abs=1e-10)


def test_affine_fit_detects_reflection_and_drops_nans():
    rng = np.random.default_rng(2)
    roi_xy = rng.normal(size=(20, 2))
    rf_xy = roi_xy * np.array([1., -1.])
    roi_xy[3] = np.nan
    rf_xy[5, 1] = np.inf
    fit = fit_affine_roi_to_rf(roi_xy, rf_xy)
    assert fit['reflection']
    assert fit['n'] == 18
    assert 'REFLECTION' in _format_affine(fit)


def test_affine_fit_too_few_points():
    roi_xy = np.array([[0., 0.], [1., 0.], [0., 1.], [5., 5.]])
    groups = np.array([0, 0, 1, 1])
    fit = fit_affine_roi_to_rf(roi_xy, roi_xy, groups=groups)
    assert np.all(np.isnan(fit['A']))
    assert np.isnan(fit['r2'])
    assert not fit['reflection']
    assert 'REFLECTION' not in _format_affine(fit)


# --- _compute_contours ---

def _cone(shape, row0, col0, radius):
    yy, xx = np.mgrid[:shape[0], :shape[1]].astype(float)
    return 1. - np.sqrt((yy - row0) ** 2 + (xx - col0) ** 2) / radius


def test_compute_contours_sorted_by_size_with_cdia():
    shape = (21, 41)
    srf = np.clip(_cone(shape, 10, 10, 6.), 0, None) + np.clip(_cone(shape, 10, 30, 10.), 0, None)
    levels = [0.5, 0.7]
    contours, areas, cdias = _compute_contours(
        srf, pixel_size=10., norm_kind='amp_one', blur_std=0, blur_npix=0, upsample_srf_scale=1, levels=levels)
    assert list(contours.keys()) == levels
    for level in levels:
        assert len(contours[level]) == 2
        assert areas[level][0] > areas[level][1]
        np.testing.assert_allclose(cdias[level], 2 * np.sqrt(np.asarray(areas[level]) / np.pi))
    # Higher level -> smaller contours
    assert areas[0.7][0] < areas[0.5][0]


def test_compute_contours_upsampling_keeps_um_units():
    srf = _cone((21, 21), 10, 10, 8.)
    _, areas1, _ = _compute_contours(srf, 10., 'none', 0, 0, 1, [0.5])
    _, areas4, _ = _compute_contours(srf, 10., 'none', 0, 0, 4, [0.5])
    assert areas4[0.5][0] == pytest.approx(areas1[0.5][0], rel=0.15)


def test_compute_contours_plot_smoke():
    srf = _cone((15, 15), 7, 7, 6.)
    contours, _, _ = _compute_contours(srf, 10., 'amp_one', 1., 1, 2, [0.5, 0.6], plot=True, title='key')
    assert len(contours[0.5]) == 1


@pytest.mark.xfail(strict=True, reason="BUG: _compute_contours divides pixel_size by upsample_srf_scale=0 "
                                       "(0 means no upsampling, cf. SplitRFParams default)")
def test_compute_contours_without_upsampling_scale_zero():
    srf = _cone((15, 15), 7, 7, 6.)
    _, areas0, _ = _compute_contours(srf, 10., 'amp_one', 0, 0, 0, [0.5])
    _, areas1, _ = _compute_contours(srf, 10., 'amp_one', 0, 0, 1, [0.5])
    assert areas0[0.5] == pytest.approx(areas1[0.5])


# --- fetch1_rf_time ---

class _FakeRel:
    def __init__(self, row):
        self.row = row

    def __and__(self, key):
        return self

    def fetch1(self, attr):
        if attr not in self.row:
            raise dj.DataJointError(f'{attr} not found')
        return self.row[attr]


def test_fetch1_rf_time_fallbacks():
    rf_time = np.linspace(-1, 0, 5)
    assert fetch1_rf_time(_FakeRel(dict(rf_time=rf_time)), {}) is not None
    np.testing.assert_array_equal(fetch1_rf_time(_FakeRel(dict(rf_time=rf_time)), {}), rf_time)
    np.testing.assert_array_equal(fetch1_rf_time(_FakeRel(dict(model_dict=dict(rf_time=rf_time))), {}), rf_time)

    table = _FakeRel(dict())
    table.params_table = _FakeRel(dict(rf_time=rf_time))
    np.testing.assert_array_equal(fetch1_rf_time(table, {}), rf_time)


# --- SplitRFTemplate._make_compute ---

def _strf(sign):
    rf_time = np.linspace(-0.95, 0., 20)
    trf = np.exp(-0.5 * ((rf_time + 0.2) / 0.05) ** 2) - 0.4 * np.exp(-0.5 * ((rf_time + 0.5) / 0.08) ** 2)
    yy, xx = np.mgrid[:8, :12]
    srf = np.exp(-0.5 * ((yy - 2) ** 2 + (xx - 9) ** 2) / 1.5 ** 2)
    return sign * merge_strf(srf, trf), rf_time


@pytest.mark.parametrize('sign', [1, -1])
def test_split_make_compute_svd_trf_positive_srf_carries_sign(sign):
    strf, rf_time = _strf(sign)
    fake_self = SimpleNamespace(_max_dt_future=np.inf)
    srf, trf, polarity, split_qidx, peak_idxs = SplitRFTemplate._make_compute(
        fake_self, strf, rf_time, method='SVD', blur_std=0, blur_npix=0, upsample_srf_scale=0, peak_nstd=1,
        npeaks_max=0)
    assert split_qidx == pytest.approx(1.)
    # tRF main (latest) peak is positive, sRF carries the polarity
    assert trf[peak_idxs[-1]] > 0
    assert np.sign(srf[2, 9]) == sign
    np.testing.assert_allclose(merge_strf(srf, trf), strf, atol=1e-10)


@pytest.mark.parametrize('method', ['sd', 'peak'])
def test_split_make_compute_other_methods_compute_qidx(method):
    strf, rf_time = _strf(-1)
    fake_self = SimpleNamespace(_max_dt_future=np.inf)
    srf, trf, polarity, split_qidx, peak_idxs = SplitRFTemplate._make_compute(
        fake_self, strf, rf_time, method=method, blur_std=0, blur_npix=0, upsample_srf_scale=2, peak_nstd=1,
        npeaks_max=0)
    assert srf.shape == (16, 24)
    assert polarity == -1  # not flipped for non-SVD methods
    assert np.isfinite(split_qidx)


@pytest.mark.xfail(strict=True, reason="BUG: SplitRF split_qidx for 'sd'/'peak' compares unscaled sRF x tRF with the "
                                       "STRF, so it depends on STRF amplitude and sign")
@pytest.mark.parametrize('method', ['sd', 'peak'])
@pytest.mark.parametrize('amp', [2., -1.])
def test_split_qidx_of_separable_strf_is_one(method, amp):
    strf, rf_time = _strf(1)
    fake_self = SimpleNamespace(_max_dt_future=np.inf)
    split_qidx = SplitRFTemplate._make_compute(
        fake_self, amp * strf, rf_time, method=method, blur_std=0, blur_npix=0, upsample_srf_scale=0, peak_nstd=1,
        npeaks_max=0)[3]
    assert split_qidx == pytest.approx(1., abs=0.01)
