import matplotlib

matplotlib.use('Agg')

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402

from djimaging.utils.receptive_fields.plot_rf_utils import (  # noqa: E402
    flatten_color_channels, plot_rf_frames, plot_srf_gauss_fit)


def test_flatten_two_color_channels():
    rf = np.stack([np.zeros((2, 3, 4)), np.ones((2, 3, 4))], axis=-1)
    flat = flatten_color_channels(rf)
    assert flat.shape == (2, 3, 9)
    np.testing.assert_array_equal(flat[..., :4], 0.)
    assert np.all(np.isnan(flat[..., 4]))
    np.testing.assert_array_equal(flat[..., 5:], 1.)


@pytest.mark.xfail(strict=True, reason="BUG: flatten_color_channels uses shift=1 for all channels > 0, so channels 1 "
                                       "and 2 are not separated by NaN and the last column stays NaN")
def test_flatten_three_color_channels():
    rf = np.stack([np.full((2, 3, 4), float(i)) for i in range(3)], axis=-1)
    flat = flatten_color_channels(rf)
    assert flat.shape == (2, 3, 14)
    for i in range(3):
        np.testing.assert_array_equal(flat[..., 5 * i:5 * i + 4], float(i))
    assert np.all(np.isnan(flat[..., [4, 9]]))


@pytest.mark.parametrize('shape', [(6, 5), (6, 4, 5), (6, 4, 5, 2)])
def test_plot_rf_frames_smoke(shape):
    rf = np.random.default_rng(0).normal(size=shape)
    fig, axs = plot_rf_frames(rf, rf_time=np.linspace(-0.5, 0, shape[0]), downsample=2)
    assert fig is not None
    plt.close(fig)


def test_plot_rf_frames_wrong_dims():
    with pytest.raises(NotImplementedError):
        plot_rf_frames(np.zeros(5), rf_time=np.arange(5.))


def test_plot_srf_gauss_fit_smoke():
    fig, ax = plt.subplots()
    srf = np.random.default_rng(0).normal(size=(10, 12))
    params = dict(x_mean=6., y_mean=3., x_stddev=2., y_stddev=1., theta=0.3)
    plot_srf_gauss_fit(ax, srf=srf, srf_params=params, plot_cb=True)
    plot_srf_gauss_fit(ax, srf=srf, vabsmax=1.)
    assert len(ax.patches) == 1
    ellipse = ax.patches[0]
    assert ellipse.center == (6., 3.)
    assert ellipse.width == pytest.approx(8.) and ellipse.height == pytest.approx(4.)
    assert ellipse.angle == pytest.approx(np.rad2deg(0.3))
    plt.close(fig)
