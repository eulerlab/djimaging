import numpy as np
import pytest
from astropy.modeling.functional_models import Gaussian2D

from djimaging.utils.receptive_fields.joint_srf_fit_utils import fit_srfs_joint


def _gauss(shape, x0, y0, sx, sy, theta=0., amp=1.):
    yy, xx = np.mgrid[:shape[0], :shape[1]].astype(float)
    return Gaussian2D.evaluate(xx, yy, amp, x0, y0, sx, sy, theta)


@pytest.mark.parametrize('kind', ['gauss', 'dog'])
def test_recovers_rotated_gaussian(kind):
    srf = _gauss((20, 30), x0=12.3, y0=8.7, sx=3., sy=1.5, theta=0.4, amp=-2.)
    fit, center, surround, params = fit_srfs_joint(srf, kind=kind, center_prior_weight=0., coupling_weight=0.)
    assert fit.shape == srf.shape
    assert params['r2'] > 0.99
    assert params['polarity'] == -1
    assert params['x_mean'] == pytest.approx(12.3, abs=0.05)
    assert params['y_mean'] == pytest.approx(8.7, abs=0.05)
    # Same parametrization as astropy (and thus FitGauss2DRF / plot_srf_gauss_fit)
    assert np.allclose(_gauss(srf.shape, params['x_mean'], params['y_mean'], params['x_stddev'], params['y_stddev'],
                              params['theta'], params['amplitude']), center, atol=1e-3)
    np.testing.assert_allclose(center + surround, fit)


def test_recovers_dog():
    shape = (31, 31)
    srf = _gauss(shape, 15, 15, 2., 2.) - 0.3 * _gauss(shape, 15, 15, 6., 6.)
    _, _, _, params = fit_srfs_joint(srf, kind='dog', center_prior_weight=0., coupling_weight=0.)
    assert params['r2'] > 0.99
    assert params['surround_ratio'] == pytest.approx(3., rel=0.05)
    assert params['surround_rel_amp'] == pytest.approx(0.3, rel=0.05)


def test_coupling_pulls_weak_channel_to_strong_channel():
    rng = np.random.default_rng(0)
    shape = (20, 20)
    strong = _gauss(shape, 6., 7., 2., 2.)
    weak = 0.15 * _gauss(shape, 6., 7., 2., 2.) + 0.1 * rng.standard_normal(shape)
    weak[16, 16] = 0.6  # noise peak far away from the RF
    srfs = np.stack([strong, weak])

    _, _, _, params_free = fit_srfs_joint(srfs, kind='gauss', center_prior_weight=0., coupling_weight=0.)
    _, _, _, params_coupled = fit_srfs_joint(srfs, kind='gauss', center_prior_weight=0., coupling_weight=0.5)

    def dist(p):
        return np.hypot(p['x_mean'] - 6., p['y_mean'] - 7.)

    assert dist(params_coupled[0]) < 0.2
    assert dist(params_coupled[1]) < dist(params_free[1])
    assert dist(params_coupled[1]) < 1.


def test_center_prior_pulls_towards_prior_center():
    shape = (20, 20)
    srf = _gauss(shape, 4., 4., 2., 2.)
    _, _, _, p_free = fit_srfs_joint(srf, kind='gauss', center_prior_weight=0.)
    _, _, _, p_prior = fit_srfs_joint(srf, kind='gauss', prior_center=(10., 10.), center_prior_sd=2.,
                                      center_prior_weight=1.)
    assert p_free['x_mean'] == pytest.approx(4., abs=0.05)
    assert 4.1 < p_prior['x_mean'] < 10. and 4.1 < p_prior['y_mean'] < 10.


def test_nan_and_zero_srfs_return_nan():
    for srfs in [np.zeros((2, 5, 5)), np.full((2, 5, 5), np.nan)]:
        fits, centers, surrounds, params = fit_srfs_joint(srfs, kind='dog')
        assert np.all(np.isnan(fits)) and np.all(np.isnan(centers))
        assert len(params) == 2 and np.isnan(params[0]['x_mean'])
