import warnings

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks

from djimaging.utils import filter_utils, math_utils
from djimaging.utils.filter_utils import lowpass_filter_trace, upsample_stim
from djimaging.utils.trace_utils import get_mean_dt, align_stim_to_trace, align_trace_to_stim


def prepare_noise_data(stim, triggertimes, trace, tracetime, ntrigger_per_frame=1,
                       fupsample_trace=None, fupsample_stim=None, fit_kind='trace',
                       lowpass_cutoff=0, ref_time='trace', pre_blur_sigma_s=0, post_blur_sigma_s=0, dt_rtol=0.1):
    assert triggertimes.ndim == 1, triggertimes.shape
    assert trace.ndim == 1, trace.shape
    assert tracetime.ndim == 1, tracetime.shape
    assert trace.size == tracetime.size

    if ntrigger_per_frame > 1:
        stimtime = np.repeat(triggertimes, ntrigger_per_frame)
        trigger_dt, _ = get_mean_dt(triggertimes)
        stimtime += np.tile(np.arange(ntrigger_per_frame) * trigger_dt / ntrigger_per_frame,
                            stim.shape[0] // ntrigger_per_frame)
    else:
        stimtime = triggertimes

    if stim.shape[0] < stimtime.size:
        raise ValueError(
            f"More triggertimes than expected: stim-len {stim.shape[0]} != stimtime-len {stimtime.size}")
    elif stim.shape[0] > stimtime.size:
        warnings.warn(f"Less triggertimes than expected: stim-len {stim.shape[0]} != stimtime-len {stimtime.size}")
        stim = stim[:stimtime.size].copy()

    dt, dt_rel_error = get_mean_dt(tracetime)

    if dt_rel_error > dt_rtol:
        warnings.warn('Inconsistent step-sizes in trace, resample trace.')
        tracetime, trace = filter_utils.resample_trace(tracetime=tracetime, trace=trace, dt=dt)

    if lowpass_cutoff > 0:
        trace = lowpass_filter_trace(trace=trace, fs=1. / dt, f_cutoff=lowpass_cutoff)

    if pre_blur_sigma_s > 0:
        trace = gaussian_filter(trace, sigma=pre_blur_sigma_s / dt, mode='nearest')

    tracetime, trace = prepare_trace(tracetime, trace, kind=fit_kind, fupsample=fupsample_trace, dt=dt)

    if (fupsample_stim is not None) and (fupsample_stim > 1):
        stimtime, stim = upsample_stim(stimtime, stim, fupsample=fupsample_stim)

    if ref_time == 'trace':
        stim, trace, dt, t0, dt_rel_error = align_stim_to_trace(
            stim=stim, stimtime=stimtime, trace=trace, tracetime=tracetime)
    elif ref_time == 'stim':
        stim, trace, dt, t0, dt_rel_error = align_trace_to_stim(
            stim=stim, stimtime=stimtime, trace=trace, tracetime=tracetime)
    else:
        raise NotImplementedError

    if post_blur_sigma_s > 0:
        trace = gaussian_filter(trace, sigma=post_blur_sigma_s / dt, mode='nearest')

    trace = trace / np.std(trace)

    if stim.ndim > 1:  # Otherwise assume stimulus indexes have been passed
        if 'bool' in str(stim.dtype) or set(np.unique(stim).astype(float)) == {0., 1.}:
            stim = 2 * stim.astype(np.int8) - 1
        else:
            stim = math_utils.normalize_zscore(stim)

    assert stim.shape[0] == trace.shape[0], (stim.shape[0], trace.shape[0])

    return stim, trace, dt, t0, dt_rel_error


def prepare_trace(tracetime, trace, kind='trace', fupsample=None, dt=None):
    if fupsample is None:
        fupsample = 1
    else:
        assert dt is not None

    if kind == 'trace':
        if fupsample > 1:
            fit_tracetime, fit_trace = filter_utils.upsample_trace(
                tracetime=tracetime, trace=trace, fupsample=fupsample)
        else:
            fit_tracetime, fit_trace = tracetime, trace

    elif kind == 'gradient':
        diff_trace = np.append(0, np.diff(trace))
        if fupsample > 1:
            fit_tracetime, fit_trace = filter_utils.upsample_trace(
                tracetime=tracetime, trace=diff_trace, fupsample=fupsample)
        else:
            fit_tracetime, fit_trace = tracetime, diff_trace

        fit_trace = np.clip(fit_trace, 0, None)

    elif kind == 'events':
        # Baden et al 2016
        diff_trace = np.append(0, np.diff(trace))
        if fupsample > 1:
            fit_tracetime, diff_trace = filter_utils.upsample_trace(
                tracetime=tracetime, trace=diff_trace, fupsample=fupsample)
        else:
            fit_tracetime = tracetime

        robust_std = np.median(np.abs(diff_trace)) / 0.6745
        peaks, props = find_peaks(diff_trace, height=robust_std)

        fit_trace = np.zeros(fit_tracetime.size)
        fit_trace[peaks] = props['peak_heights']
    else:
        raise NotImplementedError(kind)

    assert fit_tracetime.size == fit_trace.size, f"{fit_tracetime.size} != {fit_trace.size}"

    return fit_tracetime, fit_trace
