import warnings

import matplotlib.pyplot as plt
import numpy as np
from astropy.modeling.fitting import SLSQPLSQFitter
from astropy.modeling.functional_models import Gaussian2D

from djimaging.utils.receptive_fields.split_rf_utils import compute_explained_rf


def compute_gauss_srf_area(srf_params, pix_scale_x_um, pix_scale_y_um):
    x_std = srf_params['x_stddev']
    y_std = srf_params['y_stddev']

    area = np.pi * (2. * x_std * pix_scale_x_um) * (2. * y_std * pix_scale_y_um)
    diameter = np.sqrt(area / np.pi) * 2

    return area, diameter


def compute_surround_index(srf, srf_center=None, srf_surround=None):
    if (srf_center is None) == (srf_surround is None):
        raise ValueError('Provide either srf_center or srf_surround')
    srf_surround = srf - srf_center if srf_surround is None else srf_surround
    return np.sum(srf_surround) / np.sum(np.abs(srf))


def compute_center_index(srf, srf_center):
    return np.sum(srf_center) / np.sum(np.abs(srf))


def fit_rf_model(srf, kind='gaussian', polarity=None, center=None):
    assert srf.ndim == 2, 'Provide 2d RF'

    xx, yy = np.meshgrid(np.arange(0, srf.shape[1]), np.arange(0, srf.shape[0]))

    polarities = (-1, 1) if polarity is None else (polarity,)

    model, model_fit, model_params = None, None, None
    qi = -1.

    for polarity_i in polarities:
        if kind == 'gauss':
            model_i = srf_gauss_model(srf, polarity=polarity_i, center=center)
        elif kind == 'dog':
            model_i = srf_dog_model(srf, polarity=polarity_i, center=center)
        else:
            raise NotImplementedError(kind)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_i = SLSQPLSQFitter()(model=model_i, x=xx, y=yy, z=srf, verblevel=0)

        model_i_params = {k: v for k, v in zip(model_i.param_names, model_i.param_sets.flatten())}
        model_i_fit = model_i(xx, yy)
        qi_i = compute_explained_rf(srf, model_i_fit)

        if qi_i > qi:
            model, model_fit, model_params, qi = model_i, model_i_fit, model_i_params, qi_i

    if kind == 'gauss':
        return model_fit, model_params, qi
    elif kind == 'dog':
        model_c_fit = model[0](xx, yy)
        model_s_fit = -model[1](xx, yy)
        eff_polarity = int(np.sign(model(model_params['x_mean_0'], model_params['y_mean_0'])))

        return model_fit, model_c_fit, model_s_fit, model_params, eff_polarity, qi


def estimate_srf_center_model_init_params(srf, polarity=None, amplimscale=2., plot=False):
    assert srf.ndim == 2, 'Provide 2d RF'

    if polarity is None:
        y0, x0 = np.unravel_index(np.argmax(np.abs(srf)), shape=srf.shape)
    elif polarity == 1:
        y0, x0 = np.unravel_index(np.argmax(srf), shape=srf.shape)
    elif polarity == -1:
        y0, x0 = np.unravel_index(np.argmin(srf), shape=srf.shape)
    else:
        raise NotImplementedError(polarity)

    x0 = int(x0)
    y0 = int(y0)

    pol = np.sign(srf[y0, x0])
    assert pol == polarity

    srf_cut = np.clip(srf[y0] * pol, 0, None)

    std_left = np.append(np.where((srf_cut[:x0 + 1][::-1]) < (0.35 * srf_cut[x0]))[0], 1)[0]
    std_right = np.append(np.where((srf_cut[x0:]) < (0.35 * srf_cut[x0]))[0], 1)[0]

    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(f"polarity={pol}")
        im = axs[0].imshow(srf, vmin=-np.max(np.abs(srf)), vmax=np.max(np.abs(srf)), cmap='coolwarm')
        plt.colorbar(im, ax=axs[0])
        axs[0].plot(x0, y0, 'kX')
        axs[1].plot(srf_cut)
        axs[1].axhline(pol * srf[y0, x0], c='k')
        plt.axvline(x=x0 - std_left, c='r')
        plt.axvline(x=x0, c='k')
        plt.axvline(x=x0 + std_right, c='r')
        plt.show()

    std0 = np.mean([std_left, std_right])

    xlim = (0, srf.shape[1])
    ylim = (0, srf.shape[0])

    if polarity is None:
        amp0 = srf.flat[np.argmax(np.abs(srf))]
        amplim = (-amplimscale * np.abs(amp0), amplimscale * np.abs(amp0))
    elif polarity == 1:
        amp0 = np.max(srf)
        assert amp0 > 0, amp0
        amplim = (0., amplimscale * amp0)
    elif polarity == -1:
        amp0 = np.min(srf)
        assert amp0 < 0, amp0
        amplim = (amplimscale * amp0, 0.)
    else:
        raise NotImplementedError(polarity)

    return x0, y0, amp0, std0, xlim, ylim, amplim


def srf_gauss_model(srf, polarity=None, center=None):
    assert srf.ndim == 2, 'Provide 2d RF'

    x0, y0, amp0, std0, xlim, ylim, amplim = estimate_srf_center_model_init_params(
        srf=srf, polarity=polarity, amplimscale=2.)

    if center is not None:
        x0, y0 = center
        fixed_params = dict(x_mean=True, y_mean=True)
    else:
        fixed_params = dict()

    model = Gaussian2D(
        amplitude=amp0, x_mean=x0, y_mean=y0, x_stddev=std0, y_stddev=std0,
        bounds={'amplitude': amplim, 'x_mean': xlim, 'y_mean': ylim}, fixed=fixed_params)

    return model


def srf_dog_model(srf, polarity=None, center=None):
    assert srf.ndim == 2, 'Provide 2d RF'

    x0, y0, amp0, std0, xlim, ylim, amplim = estimate_srf_center_model_init_params(
        srf=srf, polarity=polarity, amplimscale=10.)

    if center is not None:
        x0, y0 = center
        fixed_params = dict(x_mean=True, y_mean=True)
    else:
        fixed_params = dict()

    xlim = (np.maximum(xlim[0], x0 - std0), np.minimum(xlim[1], x0 + std0))
    ylim = (np.maximum(ylim[0], y0 - std0), np.minimum(ylim[1], y0 + std0))
    stdlim = (0.1 * std0, 10. * std0)

    f_c = Gaussian2D(
        amplitude=amp0, x_mean=x0, y_mean=y0, x_stddev=std0, y_stddev=std0,
        bounds={'amplitude': amplim, 'x_mean': xlim, 'y_mean': ylim, 'x_stddev': stdlim, 'y_stddev': stdlim},
        fixed=fixed_params)

    f_s = Gaussian2D(
        amplitude=1. / 50. * amp0, x_mean=x0, y_mean=y0, x_stddev=5. * std0, y_stddev=5. * std0,
        bounds={'amplitude': amplim, 'x_mean': xlim, 'y_mean': ylim},
        fixed=fixed_params)

    model = f_c - f_s

    f_s.x_mean.tied = lambda _model: _model.x_mean_0
    f_s.y_mean.tied = lambda _model: _model.y_mean_0
    f_s.theta.tied = lambda _model: _model.theta_0
    f_s.x_stddev.tied = lambda _model: _model.y_stddev_1 * _model.x_stddev_0 / _model.y_stddev_0

    return model


def srf_dog_model_from_params(params):
    f_c = Gaussian2D(
        amplitude=params['amplitude_0'], x_mean=params['x_mean_0'], y_mean=params['y_mean_0'],
        x_stddev=params['x_stddev_0'], y_stddev=params['y_stddev_0'])

    f_s = Gaussian2D(
        amplitude=params['amplitude_1'], x_mean=params['x_mean_1'], y_mean=params['y_mean_1'],
        x_stddev=params['x_stddev_1'], y_stddev=params['y_stddev_1'])

    model = f_c - f_s

    return model
