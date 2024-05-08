import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from scipy.signal import find_peaks


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

    return np.abs(peak_dt)


def compute_polarity_and_peak_idxs(trf, nstd=1., npeaks_max=None, rf_time=None, max_dt_future=np.inf):
    """Estimate polarity. 1 for ON-cells, -1 for OFF-cells, 0 for uncertain cells"""

    trf = trf.copy()
    std_trf = np.std(trf)

    if (rf_time is not None) or (np.isfinite(max_dt_future)):
        trf[rf_time > max_dt_future] = 0.

    pos_peak_idxs, _ = find_peaks(trf, prominence=nstd * std_trf / 2., height=nstd * std_trf)
    neg_peak_idxs, _ = find_peaks(-trf, prominence=nstd * std_trf / 2., height=nstd * std_trf)

    peak_idxs = np.sort(np.concatenate([pos_peak_idxs, neg_peak_idxs]))

    if (npeaks_max is not None) and (peak_idxs.size > npeaks_max):
        peak_idxs = peak_idxs[-npeaks_max:]

    if peak_idxs.size > 0:
        polarity = (trf[peak_idxs[-1]] > 0) * 2 - 1
    else:
        polarity = 0

    return polarity, peak_idxs


def compute_rel_weight_baseline(rf_time, trf, dt_baseline):
    idxs_baseline = rf_time <= (np.min(rf_time) + dt_baseline)
    rel_weight_baseline = np.sum(np.abs(trf[idxs_baseline])) / np.sum(np.abs(trf))
    return rel_weight_baseline
