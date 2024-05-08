import warnings

import numpy as np
from astropy.modeling.fitting import SLSQPLSQFitter
from astropy.modeling.functional_models import Gaussian2D
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

from djimaging.tables.receptivefield.rf_utils import compute_explained_rf


def compute_gauss_srf_area(srf_params, pix_scale_x_um, pix_scale_y_um):
    x_std = srf_params['x_stddev']
    y_std = srf_params['y_stddev']

    area = np.pi * (2. * x_std * pix_scale_x_um) * (2. * y_std * pix_scale_y_um)
    diameter = np.sqrt(area / np.pi) * 2

    return area, diameter


def compute_surround_index(srf, srf_center):
    return np.sum(srf - srf_center) / np.sum(np.abs(srf))


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


def get_main_and_pre_peak(rf_time, trf, trf_peak_idxs, max_dt_future=np.inf):
    """Compute amplitude and time of main peak, and pre peak if available (e.g. in biphasic tRFs)"""
    trf_peak_idxs = trf_peak_idxs[rf_time[trf_peak_idxs] <= max_dt_future]  # Remove peaks far in the future
    trf_peak_idxs = trf_peak_idxs[np.argsort(np.abs(trf[trf_peak_idxs]))[-2:]]  # Consider two biggest peaks
    trf_peak_idxs = trf_peak_idxs[np.argsort(rf_time[trf_peak_idxs])][::-1]  # Order by time

    if trf_peak_idxs.size == 0:
        return None, None, None, None, None, None

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


def compute_trf_transience_index(rf_time, trf, trf_peak_idxs, max_dt_future=np.inf):
    """Compute transience index of temporal receptive field. Requires precomputed peak indexes"""

    main_peak, t_main_peak, main_peak_idx, pre_peak, t_pre_peak, pre_peak_idx = \
        get_main_and_pre_peak(rf_time, trf, trf_peak_idxs, max_dt_future=max_dt_future)

    if main_peak is None:
        return None

    if np.sign(pre_peak) != np.sign(main_peak):
        if np.abs(pre_peak) > np.abs(main_peak):
            transience_index = 1.
        else:
            transience_index = 2 * np.abs(pre_peak) / (np.abs(main_peak) + np.abs(pre_peak))
    else:
        transience_index = 0.

    return transience_index


def compute_half_amp_width(rf_time, trf, trf_peak_idxs, plot=False, max_dt_future=np.inf):
    """Compute max amplitude half width of temporal receptive field"""

    main_peak, t_main_peak, main_peak_idx, *_ = get_main_and_pre_peak(
        rf_time, trf, trf_peak_idxs, max_dt_future=max_dt_future)

    if main_peak is None:
        return None

    half_amp = main_peak / 2

    t_trf_int = np.linspace(rf_time[0], rf_time[-1], 1001)
    trf_fun = CubicSpline(x=rf_time, y=trf, bc_type='natural')

    roots = trf_fun.roots()
    pre_root = roots[(roots - t_main_peak) < 0][-1] if np.any(roots[(roots - t_main_peak) < 0]) else rf_time[0]
    post_root = roots[(roots - t_main_peak) > 0][0] if np.any(roots[(roots - t_main_peak) > 0]) else rf_time[-1]

    pre_root = np.maximum(pre_root, rf_time[0])
    post_root = np.minimum(post_root, rf_time[-1])

    t_right = t_main_peak
    t_left = t_main_peak

    def min_fun(x):
        return (trf_fun(x) - half_amp) ** 2

    for w in [0.5, 0.9, 0.99]:
        right_sol = minimize(min_fun, x0=w * t_main_peak + (1 - w) * post_root, bounds=[(t_main_peak, post_root)])
        if right_sol.success and right_sol.fun <= min_fun(t_right):
            t_right = float(right_sol.x)

        left_sol = minimize(min_fun, x0=w * t_main_peak + (1 - w) * pre_root, bounds=[(pre_root, t_main_peak)])
        if left_sol.success and left_sol.fun <= min_fun(t_left):
            t_left = float(left_sol.x)

    half_amp_width = float(np.abs(t_right - t_left))

    if plot:
        plt.figure()
        plt.title('Half amplitude width')
        plt.fill_between(rf_time, trf, label='trf', alpha=0.5)
        plt.fill_between(t_trf_int, trf_fun(t_trf_int), color='gray', label='trf (interpolated)', alpha=0.5)
        plt.axvline(t_main_peak, c='green', label='main peak')
        plt.axvline(pre_root, c='c', label='search start')
        plt.axvline(post_root, c='purple', label='search end')
        plt.plot([t_left, t_right], [half_amp, half_amp], c='red', lw=2, alpha=0.7, marker='o',
                 label='solution')
        plt.legend()
        plt.show()

    return half_amp_width


def compute_main_peak_lag(rf_time, trf, trf_peak_idxs, plot=False, max_dt_future=np.inf):
    main_peak, t_main_peak, main_peak_idx, *_ = get_main_and_pre_peak(
        rf_time, trf, trf_peak_idxs, max_dt_future=max_dt_future)

    if main_peak is None:
        return None

    trf_fun = CubicSpline(x=rf_time, y=trf, bc_type='natural')

    peak_dt_approx = rf_time[main_peak_idx]

    rf_time_int = np.linspace(peak_dt_approx - 0.1, peak_dt_approx + 0.1, 1001)
    trf_int = trf_fun(rf_time_int)

    peak_dt = rf_time_int[np.argmax(trf_int)]

    if plot:
        plt.figure()
        plt.title('Main peak lag')
        plt.fill_between(rf_time, trf, label='trf', alpha=0.5)
        plt.plot(rf_time_int, trf_int, alpha=0.8)
        plt.axvline(peak_dt_approx, c='red', label='Approx solution')
        plt.axvline(peak_dt, c='cyan', ls='--', label='Solution')
        plt.legend()
        plt.show()

    return -peak_dt
