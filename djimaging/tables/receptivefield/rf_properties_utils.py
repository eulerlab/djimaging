import warnings

import numpy as np
from astropy.modeling.fitting import SLSQPLSQFitter
from astropy.modeling.functional_models import Gaussian2D
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize


def compute_gauss_srf_area(srf_params, pix_scale_x_um, pix_scale_y_um):
    x_std = srf_params['x_stddev']
    y_std = srf_params['y_stddev']

    area = np.pi * (2. * x_std * pix_scale_x_um) * (2. * y_std * pix_scale_y_um)
    diameter = np.sqrt(area / np.pi) * 2

    return area, diameter


def compute_surround_index(srf, srf_center):
    return np.sum(srf - srf_center) / np.sum(np.abs(srf))


def fit_rf_model(srf, kind='gaussian', polarity=None):
    assert srf.ndim == 2, 'Provide 2d RF'

    if kind == 'gauss':
        model = srf_gauss_model(srf, polarity=polarity)
    elif kind == 'dog':
        model = srf_dog_model(srf, polarity=polarity)
    else:
        raise NotImplementedError(kind)

    xx, yy = np.meshgrid(np.arange(0, srf.shape[1]), np.arange(0, srf.shape[0]))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SLSQPLSQFitter()(model=model, x=xx, y=yy, z=srf, verblevel=0)

    model_params = {k: v for k, v in zip(model.param_names, model.param_sets.flatten())}
    model_fit = model(xx, yy)

    if kind == 'gauss':
        return model_fit, model_params
    elif kind == 'dog':
        model_c_fit = model[0](xx, yy)
        model_s_fit = -model[1](xx, yy)
        eff_polarity = int(np.sign(model(model_params['x_mean_0'], model_params['y_mean_0'])))

        return model_fit, model_c_fit, model_s_fit, model_params, eff_polarity


def estimate_srf_center_model_init_params(srf, polarity=None, amplimscale=2.):
    assert srf.ndim == 2, 'Provide 2d RF'

    if polarity is None:
        y0, x0 = np.unravel_index(np.argmax(np.abs(srf)), shape=srf.shape)
    elif polarity == 1:
        y0, x0 = np.unravel_index(np.argmax(srf), shape=srf.shape)
    elif polarity == -1:
        y0, x0 = np.unravel_index(np.argmin(srf), shape=srf.shape)
    else:
        raise NotImplementedError(polarity)

    srf_cut = np.abs(srf[:, int(x0)] - np.min(srf[:, int(x0)]))
    srf_cut_profile = np.cumsum(srf_cut) / np.sum(srf_cut)
    std0 = np.maximum(1, 0.5 * (np.argmin(np.abs(srf_cut_profile - 0.7)) - np.argmin(np.abs(srf_cut_profile - 0.3))))

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


def srf_gauss_model(srf, polarity=None):
    assert srf.ndim == 2, 'Provide 2d RF'

    x0, y0, amp0, std0, xlim, ylim, amplim = estimate_srf_center_model_init_params(
        srf=srf, polarity=polarity, amplimscale=2.)

    model = Gaussian2D(
        amplitude=amp0, x_mean=x0, y_mean=y0, x_stddev=std0, y_stddev=std0,
        bounds={'amplitude': amplim, 'x_mean': xlim, 'y_mean': ylim})

    return model


def srf_dog_model(srf, polarity=None):
    assert srf.ndim == 2, 'Provide 2d RF'

    x0, y0, amp0, std0, xlim, ylim, amplim = estimate_srf_center_model_init_params(
        srf=srf, polarity=polarity, amplimscale=10.)

    xlim = (np.maximum(xlim[0], x0 - std0), np.minimum(xlim[1], x0 + std0))
    ylim = (np.maximum(ylim[0], y0 - std0), np.minimum(ylim[1], y0 + std0))
    stdlim = (0.1 * std0, 10. * std0)

    f_c = Gaussian2D(
        amplitude=amp0, x_mean=x0, y_mean=y0, x_stddev=std0, y_stddev=std0,
        bounds={'amplitude': amplim, 'x_mean': xlim, 'y_mean': ylim, 'x_stddev': stdlim, 'y_stddev': stdlim})

    f_s = Gaussian2D(
        amplitude=1. / 50. * amp0, x_mean=x0, y_mean=y0, x_stddev=5. * std0, y_stddev=5. * std0,
        bounds={'amplitude': amplim, 'x_mean': xlim, 'y_mean': ylim})

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


def get_main_and_pre_peak(rf_time, trf, trf_peak_idxs):
    """Compute amplitude and time of main peak, and pre peak if available (e.g. in biphasic tRFs)"""
    trf_peak_idxs = trf_peak_idxs[rf_time[trf_peak_idxs] <= 0]
    trf_peak_idxs = trf_peak_idxs[np.argsort(np.abs(trf[trf_peak_idxs]))[-2:]]

    main_peak_idx = trf_peak_idxs[0]
    main_peak = trf[main_peak_idx]
    t_main_peak = rf_time[main_peak_idx]

    if trf_peak_idxs.size > 1:
        pre_peak_idx = trf_peak_idxs[1]
    else:
        pre_peak_idx = np.argmin(trf[:main_peak_idx]) if main_peak > 0 else np.argmax(trf[:main_peak_idx])

    pre_peak = trf[pre_peak_idx]
    t_pre_peak = rf_time[pre_peak_idx]

    return main_peak, t_main_peak, main_peak_idx, pre_peak, t_pre_peak, pre_peak_idx


def compute_trf_transience_index(rf_time, trf, trf_peak_idxs):
    """Compute transience index of temporal receptive field. Requires precomputed peak indexes"""

    main_peak, t_main_peak, main_peak_idx, pre_peak, t_pre_peak, pre_peak_idx = \
        get_main_and_pre_peak(rf_time, trf, trf_peak_idxs)

    if np.sign(pre_peak) != np.sign(main_peak):
        if np.abs(pre_peak) > np.abs(main_peak):
            transience_index = 1.
        else:
            transience_index = 2 * np.abs(pre_peak) / (np.abs(main_peak) + np.abs(pre_peak))
    else:
        transience_index = 0.

    return transience_index


def compute_half_amp_width(rf_time, trf, trf_peak_idxs, plot=False):
    """Compute max amplitude half width of temporal receptive field"""

    main_peak, t_main_peak, main_peak_idx, *_ = get_main_and_pre_peak(rf_time, trf, trf_peak_idxs)

    half_amp = main_peak / 2

    t_trf_int = np.linspace(rf_time[0], rf_time[-1], 1001)
    trf_fun = CubicSpline(x=rf_time, y=trf, bc_type='natural')

    roots = trf_fun.roots()
    pre_root = roots[(roots - t_main_peak) < 0][-1] if np.any(roots[(roots - t_main_peak) < 0]) else rf_time[0]
    post_root = roots[(roots - t_main_peak) > 0][0] if np.any(roots[(roots - t_main_peak) > 0]) else rf_time[-1]

    right_sol = minimize(lambda x: (trf_fun(x) - half_amp) ** 2, x0=np.array([t_main_peak * 0.8]),
                         bounds=[(t_main_peak, 0)])
    left_sol = minimize(lambda x: (trf_fun(x) - half_amp) ** 2, x0=np.array([t_main_peak * 1.2]),
                        bounds=[(np.min(rf_time), t_main_peak)])

    assert right_sol.success and left_sol.success

    half_amp_width = float(np.abs(right_sol.x - left_sol.x))

    if plot:
        plt.figure()
        plt.fill_between(rf_time, trf, label='trf', alpha=0.5)
        plt.fill_between(t_trf_int, trf_fun(t_trf_int), color='gray', label='trf (interpolated)', alpha=0.5)
        plt.axvline(pre_root, c='c')
        plt.axvline(post_root, c='purple')
        plt.plot([left_sol.x, right_sol.x], [half_amp, half_amp], c='red', lw=2, alpha=0.7, label='solution')
        plt.legend()
        plt.show()

    return half_amp_width


def compute_main_peak_lag(rf_time, trf, trf_peak_idxs, plot=False):
    main_peak, t_main_peak, main_peak_idx, *_ = get_main_and_pre_peak(rf_time, trf, trf_peak_idxs)

    trf_fun = CubicSpline(x=rf_time, y=trf, bc_type='natural')

    peak_dt_approx = rf_time[main_peak_idx]

    rf_time_int = np.linspace(peak_dt_approx - 0.1, peak_dt_approx + 0.1, 1001)
    trf_int = trf_fun(rf_time_int)

    peak_dt = rf_time_int[np.argmax(trf_int)]

    if plot:
        plt.figure()
        plt.fill_between(rf_time, trf, label='trf', alpha=0.5)
        plt.plot(rf_time_int, trf_int, alpha=0.8)
        plt.axvline(peak_dt_approx, c='red')
        plt.axvline(peak_dt, c='cyan', ls='--')
        plt.show()

    return np.abs(peak_dt)
