import warnings

import numpy as np
import pytest

from djimaging.utils.receptive_fields.split_rf_utils import (
    compute_explained_rf, merge_strf, normalize_strf, resize_srf, smooth_rf, split_rf_peak, split_rf_sd,
    split_rf_svd, split_strf)


def _gauss2d(shape, row0, col0, sigma):
    yy, xx = np.mgrid[:shape[0], :shape[1]].astype(float)
    return np.exp(-0.5 * ((yy - row0) ** 2 + (xx - col0) ** 2) / sigma ** 2)


def _trf(nt=20, main=1., pre=-0.5):
    trf = np.zeros(nt)
    trf[-3] = main
    trf[-4] = 0.6 * main
    trf[-7] = pre
    return trf


def _strf(sign=1, shape=(10, 14), row0=3, col0=10):
    return sign * merge_strf(srf=_gauss2d(shape, row0, col0, 2.), trf=_trf())


# --- basics ---

def test_explained_rf():
    rf = np.random.default_rng(0).normal(size=(5, 6))
    assert compute_explained_rf(rf, rf) == pytest.approx(1.)
    assert compute_explained_rf(rf, np.zeros_like(rf)) == pytest.approx(0.)
    assert compute_explained_rf(rf, 0.5 * rf) == pytest.approx(0.75)


def test_merge_strf_is_outer_product():
    srf = np.arange(6.).reshape(2, 3)
    trf = np.array([1., -2., 0.5, 3.])
    strf = merge_strf(srf, trf)
    assert strf.shape == (4, 2, 3)
    np.testing.assert_allclose(strf, trf[:, None, None] * srf[None])


def test_merge_strf_rejects_wrong_dims():
    with pytest.raises(AssertionError):
        merge_strf(np.zeros((2, 3)), np.zeros((2, 2)))
    with pytest.raises(AssertionError):
        merge_strf(np.zeros((2, 2, 2)), np.zeros(2))


def test_resize_srf():
    srf = _gauss2d((6, 8), 2, 5, 1.5)
    assert resize_srf(srf, scale=3).shape == (18, 24)
    assert resize_srf(srf, output_shape=(12, 4)).shape == (12, 4)
    with pytest.raises(AssertionError):
        resize_srf(srf)


def test_normalize_strf():
    strf = _strf(sign=1)
    norm = normalize_strf(strf)
    assert np.sum(norm[norm > 0]) == pytest.approx(1.)
    norm = normalize_strf(-strf)
    # Dominant (negative) lobe sums to -1, sign is kept
    assert np.sum(norm[norm < 0]) == pytest.approx(-1.)
    np.testing.assert_allclose(normalize_strf(3. * strf), normalize_strf(strf))


# --- SVD ---

@pytest.mark.parametrize('sign', [1, -1])
def test_svd_recovers_separable_strf(sign):
    strf = _strf(sign=sign)
    with warnings.catch_warnings():
        warnings.simplefilter('error')  # rescale must not warn about a mismatch
        srf, trf, explained_variance = split_rf_svd(strf)

    assert srf.shape == strf.shape[1:]
    assert trf.shape == strf.shape[:1]
    assert explained_variance == pytest.approx(1.)
    assert np.max(np.abs(trf)) == pytest.approx(1.)
    # Peak of the sRF is at (row, col) = (3, 10)
    assert np.unravel_index(np.argmax(np.abs(srf)), srf.shape) == (3, 10)
    # Product (including sign and amplitude) reproduces the STRF
    np.testing.assert_allclose(merge_strf(srf, trf), strf, atol=1e-10)


def test_svd_trf_and_srf_proportional_to_ground_truth():
    srf_true = _gauss2d((9, 13), 6, 2, 1.5) - 0.3 * _gauss2d((9, 13), 6, 2, 4.)
    trf_true = _trf()
    srf, trf, _ = split_rf_svd(merge_strf(srf_true, trf_true))
    assert abs(np.corrcoef(srf.ravel(), srf_true.ravel())[0, 1]) == pytest.approx(1.)
    assert abs(np.corrcoef(trf, trf_true)[0, 1]) == pytest.approx(1.)


def test_svd_2d_strf():
    srf_true = np.sin(np.linspace(0, np.pi, 15))
    strf = np.outer(_trf(), srf_true)
    srf, trf, explained_variance = split_rf_svd(strf)
    assert srf.shape == (15,)
    assert explained_variance == pytest.approx(1.)
    np.testing.assert_allclose(np.outer(trf, srf), strf, atol=1e-10)


def test_svd_4d_strf_is_split_per_color():
    strf = np.stack([_strf(sign=1, row0=2, col0=3), _strf(sign=-1, row0=7, col0=11)], axis=-1)
    srf, trf, explained_variance = split_rf_svd(strf)
    assert srf.shape == (10, 14, 2)
    assert trf.shape == (20, 2)
    assert explained_variance == pytest.approx(1.)
    for i in range(2):
        np.testing.assert_allclose(merge_strf(srf[..., i], trf[:, i]), strf[..., i], atol=1e-10)


def test_svd_wrong_dims():
    with pytest.raises(NotImplementedError):
        split_rf_svd(np.zeros(5))


def test_svd_explained_variance_of_noise():
    rng = np.random.default_rng(0)
    strf = rng.normal(size=(20, 15, 20))
    _, _, explained_variance = split_rf_svd(strf)
    sv = np.linalg.svd(strf.reshape(20, -1), compute_uv=False)
    assert explained_variance == pytest.approx(sv[0] ** 2 / np.sum(sv ** 2), rel=0.01)


# --- SD and peak ---

def test_split_sd_uses_row_col_of_max_std():
    strf = _strf(sign=-1, shape=(7, 12), row0=5, col0=2)
    srf, trf = split_rf_sd(strf)
    np.testing.assert_allclose(srf, np.std(strf, axis=0))
    np.testing.assert_allclose(trf, strf[:, 5, 2])
    assert np.all(srf >= 0)


def test_split_peak():
    strf = _strf(sign=-1, shape=(7, 12), row0=5, col0=2)
    srf, trf = split_rf_peak(strf)
    it = int(np.argmax(np.abs(_trf())))
    np.testing.assert_allclose(srf, strf[it])
    np.testing.assert_allclose(trf, strf[:, 5, 2])
    assert srf[5, 2] < 0


# --- split_strf ---

@pytest.mark.parametrize('method', ['SVD', 'svd', 'sd', 'SD', 'peak'])
def test_split_strf_methods(method):
    strf = _strf()
    srf, trf, explained_variance = split_strf(strf, method=method)
    assert srf.shape == strf.shape[1:]
    assert trf.shape == strf.shape[:1]
    if method.lower() == 'svd':
        assert explained_variance == pytest.approx(1.)
    else:
        assert explained_variance is None
    assert np.unravel_index(np.argmax(np.abs(srf)), srf.shape) == (3, 10)


def test_split_strf_unknown_method():
    with pytest.raises(NotImplementedError):
        split_strf(_strf(), method='MAX')


def test_split_strf_upsample_and_blur():
    strf = _strf()
    srf, trf, _ = split_strf(strf, upsample_srf_scale=2)
    assert srf.shape == (20, 28)
    assert trf.shape == (20,)
    srf_blur, _, _ = split_strf(strf, blur_std=1., blur_npix=2)
    srf_raw, _, _ = split_strf(strf)
    assert srf_blur.shape == srf_raw.shape
    assert np.max(np.abs(srf_blur)) < np.max(np.abs(srf_raw))


@pytest.mark.xfail(strict=True, reason="BUG: split_strf blurs 4D (t, y, x, color) STRFs across the color axis")
def test_split_strf_blur_does_not_mix_colors():
    strf = np.zeros((5, 9, 9, 2))
    strf[:, 4, 4, 0] = np.arange(5.)
    strf[:, 1, 1, 1] = np.arange(5.)
    srf, _, _ = split_strf(strf, blur_std=1., blur_npix=2)
    srf0, _, _ = split_strf(strf[..., 0], blur_std=1., blur_npix=2)
    np.testing.assert_allclose(srf[..., 0], srf0, atol=1e-10)


# --- smoothing ---

def test_smooth_rf_spreads_delta_within_npix():
    delta = np.zeros((9, 9))
    delta[4, 4] = 1.
    out = smooth_rf(delta, blur_std=1., blur_npix=2)
    assert out[4, 4] < 1.
    assert out[4, 2] > 0. and out[2, 4] > 0.
    assert out[4, 1] == 0. and out[1, 4] == 0.
    assert out.sum() == pytest.approx(1.)


def test_smooth_rf_kernel_radius_is_npix():
    delta = np.zeros((9, 9))
    delta[4, 4] = 1.
    # std=2, npix=1: floor(0.5) = 0 -> no smoothing at all
    out = smooth_rf(delta, blur_std=2., blur_npix=1)
    assert out[4, 3] > 0. and out[4, 2] == 0.
    # std=2, npix=3: floor(1.5) = 1 -> radius 2 instead of 3
    out = smooth_rf(delta, blur_std=2., blur_npix=3)
    assert out[4, 1] > 0. and out[4, 0] == 0.
