import numpy as np
import pytest
from astropy.modeling.functional_models import Gaussian2D

from djimaging.utils.receptive_fields.spatial_rf_utils import (
    compute_center_index, compute_gauss_srf_area, compute_surround_index, estimate_srf_center_model_init_params,
    fit_rf_model, srf_dog_model_from_params, srf_gauss_model)


def _gauss(shape, x0, y0, sx, sy, theta=0., amp=1.):
    yy, xx = np.mgrid[:shape[0], :shape[1]].astype(float)
    return Gaussian2D.evaluate(xx, yy, amp, x0, y0, sx, sy, theta)


def _dog(shape, x0, y0, sc, ss, rel_amp, amp=1.):
    return _gauss(shape, x0, y0, sc, sc, amp=amp) - rel_amp * _gauss(shape, x0, y0, ss, ss, amp=amp)


# --- area and indexes ---

def test_gauss_srf_area_is_two_sigma_ellipse_in_um():
    area, diameter = compute_gauss_srf_area(dict(x_stddev=2., y_stddev=3.), pix_scale_x_um=5., pix_scale_y_um=10.)
    assert area == pytest.approx(np.pi * (2 * 2. * 5.) * (2 * 3. * 10.))
    # Diameter of the circle with the same area
    assert diameter == pytest.approx(2 * np.sqrt(area / np.pi))


def test_gauss_srf_area_scales_with_pixel_area():
    params = dict(x_stddev=1.5, y_stddev=1.5)
    area1, dia1 = compute_gauss_srf_area(params, 1., 1.)
    area2, dia2 = compute_gauss_srf_area(params, 3., 3.)
    assert area2 == pytest.approx(9 * area1)
    assert dia2 == pytest.approx(3 * dia1)


def test_center_and_surround_index():
    srf = np.array([[-1., 0.], [3., 0.]])
    center = np.array([[0., 0.], [3., 0.]])
    assert compute_center_index(srf, srf_center=center) == pytest.approx(3 / 4)
    # Surround = srf - center = -1
    assert compute_surround_index(srf, srf_center=center) == pytest.approx(-1 / 4)
    assert compute_surround_index(srf, srf_surround=srf - center) == pytest.approx(-1 / 4)


@pytest.mark.parametrize('kwargs', [dict(), dict(srf_center=np.ones((2, 2)), srf_surround=np.ones((2, 2)))])
def test_surround_index_requires_exactly_one_of_center_or_surround(kwargs):
    with pytest.raises(ValueError):
        compute_surround_index(np.ones((2, 2)), **kwargs)


# --- initial parameters ---

def test_init_params_use_col_as_x_and_row_as_y():
    srf = np.zeros((7, 11))
    srf[2, 8] = 1.
    srf[5, 1] = -0.5
    x0, y0, amp0, std0, xlim, ylim, amplim = estimate_srf_center_model_init_params(srf, polarity=1)
    assert (x0, y0) == (8, 2)
    assert amp0 == 1.
    assert xlim == (0, 11) and ylim == (0, 7)
    assert amplim == (0., 2.)

    x0, y0, amp0, *_, amplim = estimate_srf_center_model_init_params(srf, polarity=-1)
    assert (x0, y0) == (1, 5)
    assert amp0 == -0.5
    assert amplim == (-1., 0.)


def test_init_params_unknown_polarity():
    with pytest.raises(NotImplementedError):
        estimate_srf_center_model_init_params(np.ones((3, 3)), polarity=2)


def test_init_std_estimate_is_of_the_order_of_sigma():
    srf = _gauss((31, 41), 20, 15, 4., 4.)
    std0 = estimate_srf_center_model_init_params(srf, polarity=1)[3]
    # Distance at which the profile drops below 35% of the peak (~1.45 sigma)
    assert 4. <= std0 <= 8.


@pytest.mark.xfail(strict=True, reason="BUG: estimate_srf_center_model_init_params asserts pol == None for polarity=None")
def test_gauss_model_without_polarity():
    srf = _gauss((15, 21), 13., 6., 2., 2., amp=-1.)
    model = srf_gauss_model(srf, polarity=None)
    assert model.x_mean.value == 13
    assert model.y_mean.value == 6
    assert model.amplitude.value == pytest.approx(-1.)


# --- Gaussian fits ---

def test_fit_gauss_recovers_rotated_gaussian_on_non_square_array():
    rng = np.random.default_rng(1)
    srf = _gauss((20, 30), x0=18.3, y0=6.7, sx=3., sy=1.5, theta=0.4, amp=-2.)
    srf += 0.01 * rng.normal(size=srf.shape)

    srf_fit, params, qi = fit_rf_model(srf, kind='gauss')

    assert srf_fit.shape == srf.shape
    assert qi > 0.99
    assert params['amplitude'] == pytest.approx(-2., rel=0.02)
    # x is the column, y is the row
    assert params['x_mean'] == pytest.approx(18.3, abs=0.05)
    assert params['y_mean'] == pytest.approx(6.7, abs=0.05)
    assert params['x_stddev'] == pytest.approx(3., rel=0.02)
    assert params['y_stddev'] == pytest.approx(1.5, rel=0.02)
    assert params['theta'] == pytest.approx(0.4, abs=0.02)
    np.testing.assert_allclose(srf_fit, _gauss(srf.shape, params['x_mean'], params['y_mean'], params['x_stddev'],
                                               params['y_stddev'], params['theta'], params['amplitude']))


def test_fit_gauss_kind_aliases_and_explicit_polarity():
    srf = _gauss((15, 17), 5., 9., 2., 2.) - 0.001
    _, params_a, qi_a = fit_rf_model(srf, kind='gauss', polarity=1)
    _, params_b, qi_b = fit_rf_model(srf, kind='gaussian', polarity=1)
    assert qi_a == pytest.approx(qi_b)
    assert params_a['x_mean'] == pytest.approx(5., abs=0.01)
    assert params_a['y_mean'] == pytest.approx(9., abs=0.01)


def test_fit_gauss_fixed_center():
    srf = _gauss((20, 30), 18., 7., 3., 2.) - 0.001
    _, params, qi = fit_rf_model(srf, kind='gauss', center=(17., 7.))
    assert params['x_mean'] == 17.
    assert params['y_mean'] == 7.
    assert qi < 1.


def test_fit_unknown_kind():
    with pytest.raises(NotImplementedError):
        fit_rf_model(_gauss((9, 9), 4, 4, 1, 1) - 0.01, kind='box')


def test_fit_gauss_rejects_non_2d():
    with pytest.raises(AssertionError):
        fit_rf_model(np.zeros((2, 3, 4)))


@pytest.mark.xfail(strict=True,
                   reason="BUG: fit_rf_model(polarity=None) crashes (AssertionError) if the sRF has no negative pixel")
def test_fit_gauss_strictly_positive_srf():
    # Default of FitGauss2DRF (_polarity = None): both polarities are tried, but an sRF without any negative pixel
    # (e.g. a Gaussian on a small positive offset) raises in the polarity=-1 attempt.
    srf = _gauss((15, 25), 14., 6., 2., 3.) + 0.01
    _, params, qi = fit_rf_model(srf, kind='gauss')
    assert params['x_mean'] == pytest.approx(14., abs=0.1)
    assert params['y_mean'] == pytest.approx(6., abs=0.1)


# --- DoG fits ---

@pytest.mark.parametrize('sign', [1, -1])
def test_fit_dog_recovers_center_surround(sign):
    rng = np.random.default_rng(2)
    shape = (31, 41)
    srf = sign * _dog(shape, 25., 12., 2., 6., 0.3) + 0.001 * rng.normal(size=shape)

    srf_fit, srf_c_fit, srf_s_fit, params, eff_polarity, qi = fit_rf_model(srf, kind='dog')

    assert qi > 0.99
    assert eff_polarity == sign
    assert params['x_mean_0'] == pytest.approx(25., abs=0.05)
    assert params['y_mean_0'] == pytest.approx(12., abs=0.05)
    # Surround is tied to the center
    assert params['x_mean_1'] == pytest.approx(params['x_mean_0'])
    assert params['y_mean_1'] == pytest.approx(params['y_mean_0'])
    assert params['x_stddev_0'] == pytest.approx(2., rel=0.03)
    assert params['x_stddev_1'] == pytest.approx(6., rel=0.03)
    assert params['amplitude_1'] / params['amplitude_0'] == pytest.approx(0.3, rel=0.03)
    # Center and (signed) surround add up to the full fit
    np.testing.assert_allclose(srf_c_fit + srf_s_fit, srf_fit, atol=1e-12)
    assert np.sign(srf_c_fit[12, 25]) == sign
    assert np.sign(srf_s_fit[12, 25]) == -sign


def test_dog_model_from_params_reproduces_fit():
    shape = (31, 41)
    srf = _dog(shape, 20., 14., 2.5, 7., 0.2)
    srf_fit, _, _, params, _, _ = fit_rf_model(srf, kind='dog', polarity=1)
    model = srf_dog_model_from_params(params)
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    # theta is not passed to the model, so this only holds for (nearly) circular fits
    np.testing.assert_allclose(model(xx, yy), srf_fit, atol=1e-3)


@pytest.mark.xfail(strict=True, reason="BUG: srf_dog_model_from_params ignores theta_0/theta_1 of rotated fits")
def test_dog_model_from_params_keeps_rotation():
    params = dict(amplitude_0=1., x_mean_0=12., y_mean_0=9., x_stddev_0=3., y_stddev_0=1., theta_0=0.6,
                  amplitude_1=0.2, x_mean_1=12., y_mean_1=9., x_stddev_1=9., y_stddev_1=3., theta_1=0.6)
    expected = (_gauss((19, 25), 12., 9., 3., 1., 0.6, 1.) - _gauss((19, 25), 12., 9., 9., 3., 0.6, 0.2))
    yy, xx = np.mgrid[:19, :25]
    np.testing.assert_allclose(srf_dog_model_from_params(params)(xx, yy), expected, atol=1e-6)


def test_init_params_plot_smoke():
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    srf = _gauss((15, 21), 13., 6., 2., 2.) - 0.01
    x0, y0, *_ = estimate_srf_center_model_init_params(srf, polarity=1, plot=True)
    assert (x0, y0) == (13, 6)
    plt.close('all')
