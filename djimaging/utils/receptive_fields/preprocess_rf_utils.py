import warnings

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks

from djimaging.utils import filter_utils, math_utils
from djimaging.utils.filter_utils import lowpass_filter_trace, upsample_stim
from djimaging.utils.trace_utils import get_mean_dt, align_stim_to_trace, align_trace_to_stim


def prepare_noise_data(stim: np.ndarray, triggertimes: np.ndarray,
                       trace: np.ndarray, tracetime: np.ndarray,
                       ntrigger_per_frame: int = 1,
                       fupsample_trace: int = None,
                       fupsample_stim: int = None,
                       fit_kind: str = 'trace',
                       lowpass_cutoff: float = 0,
                       ref_time: str = 'trace',
                       pre_blur_sigma_s: float = 0,
                       post_blur_sigma_s: float = 0,
                       dt_rtol: float = 0.1) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Prepare stimulus and trace data for noise RF estimation.

    Handles resampling, filtering, alignment, and normalization of both
    stimulus and neural trace.

    Parameters
    ----------
    stim : np.ndarray
        Stimulus array, shape (n_frames, ...).
    triggertimes : np.ndarray
        Trigger timestamps in seconds, shape (n_triggers,).
    trace : np.ndarray
        Neural response trace, shape (n_timepoints,).
    tracetime : np.ndarray
        Timestamps for the trace, shape (n_timepoints,).
    ntrigger_per_frame : int, optional
        Number of triggers per stimulus frame. Default is 1.
    fupsample_trace : int, optional
        Upsampling factor for the trace. Default is None.
    fupsample_stim : int, optional
        Upsampling factor for the stimulus. Default is None.
    fit_kind : str, optional
        Transformation applied to the trace: 'trace', 'gradient', or 'events'. Default is 'trace'.
    lowpass_cutoff : float, optional
        Low-pass filter cutoff frequency in Hz. 0 disables filtering. Default is 0.
    ref_time : str, optional
        Reference time axis for alignment: 'trace' or 'stim'. Default is 'trace'.
    pre_blur_sigma_s : float, optional
        Gaussian blur sigma (seconds) applied before other transforms. Default is 0.
    post_blur_sigma_s : float, optional
        Gaussian blur sigma (seconds) applied after normalization. Default is 0.
    dt_rtol : float, optional
        Relative tolerance for step-size consistency check. Default is 0.1.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, float, float, float]
        stim : np.ndarray
            Preprocessed and aligned stimulus.
        trace : np.ndarray
            Preprocessed, aligned, and normalized trace.
        dt : float
            Time step in seconds.
        t0 : float
            Start time offset.
        dt_rel_error : float
            Relative error of the mean dt computation.

    Raises
    ------
    NotImplementedError
        If ref_time is not 'trace' or 'stim'.
    """
    assert triggertimes.ndim == 1, triggertimes.shape
    assert trace.ndim == 1, trace.shape
    assert tracetime.ndim == 1, tracetime.shape
    assert trace.size == tracetime.size

    # Resampling and filtering
    stimtime, stim = preprocess_stimulus(
        stim, triggertimes, ntrigger_per_frame, fupsample_stim)
    tracetime, trace = preprocess_trace(
        tracetime, trace, fupsample_trace, fit_kind, lowpass_cutoff, pre_blur_sigma_s, dt_rtol)

    # Aligning stimulus and trace
    if ref_time == 'trace':
        stim, trace, dt, t0, dt_rel_error = align_stim_to_trace(
            stim=stim, stimtime=stimtime, trace=trace, tracetime=tracetime)
    elif ref_time == 'stim':
        trace, dt, t0, dt_rel_error = align_trace_to_stim(stimtime=stimtime, trace=trace, tracetime=tracetime)
    else:
        raise NotImplementedError

    # Smoothing and normalization
    trace = finalize_trace(trace, dt, post_blur_sigma_s)
    stim = finalize_stim(stim)
    assert stim.shape[0] == trace.shape[0], (stim.shape[0], trace.shape[0])

    return stim, trace, dt, t0, dt_rel_error


def preprocess_stimulus(stim: np.ndarray, triggertimes: np.ndarray,
                        ntrigger_per_frame: int,
                        fupsample_stim: int = None) -> tuple[np.ndarray, np.ndarray]:
    """Preprocess the stimulus by aligning to trigger times and optionally upsampling.

    Parameters
    ----------
    stim : np.ndarray
        Stimulus array, shape (n_frames, ...).
    triggertimes : np.ndarray
        Trigger timestamps in seconds, shape (n_triggers,).
    ntrigger_per_frame : int
        Number of triggers per stimulus frame.
    fupsample_stim : int, optional
        Upsampling factor for the stimulus. Default is None.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        stimtime : np.ndarray
            Stimulus timestamps.
        stim : np.ndarray
            Possibly truncated and upsampled stimulus.

    Raises
    ------
    ValueError
        If there are more triggertimes than stimulus frames.
    """
    if ntrigger_per_frame > 1:
        stimtime = np.repeat(triggertimes, ntrigger_per_frame)
        trigger_dt, _ = get_mean_dt(triggertimes, rtol_error=np.inf, rtol_warning=0.5)
        stimtime += np.tile(np.arange(ntrigger_per_frame) * trigger_dt / ntrigger_per_frame,
                            stimtime.size // ntrigger_per_frame)
    else:
        stimtime = triggertimes

    if stim.shape[0] < stimtime.size:
        raise ValueError(
            f"More triggertimes than expected: stim-len {stim.shape[0]} != stimtime-len {stimtime.size}")
    elif stim.shape[0] > stimtime.size:
        warnings.warn(f"Less triggertimes than expected: stim-len {stim.shape[0]} != stimtime-len {stimtime.size}")
        stim = stim[:stimtime.size].copy()

    if (fupsample_stim is not None) and (fupsample_stim > 1):
        stimtime, stim = upsample_stim(stimtime, stim, fupsample=fupsample_stim)

    return stimtime, stim


def preprocess_trace(tracetime: np.ndarray, trace: np.ndarray,
                     fupsample_trace: int = None,
                     fit_kind: str = 'trace',
                     lowpass_cutoff: float = 0,
                     pre_blur_sigma_s: float = 0,
                     dt_rtol: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Preprocess the neural trace by resampling, filtering, and transforming.

    Parameters
    ----------
    tracetime : np.ndarray
        Timestamps for the trace, shape (n_timepoints,).
    trace : np.ndarray
        Neural response trace, shape (n_timepoints,).
    fupsample_trace : int, optional
        Upsampling factor for the trace. Default is None.
    fit_kind : str, optional
        Transformation applied to the trace: 'trace', 'gradient', or 'events'. Default is 'trace'.
    lowpass_cutoff : float, optional
        Low-pass filter cutoff in Hz. 0 disables filtering. Default is 0.
    pre_blur_sigma_s : float, optional
        Gaussian blur sigma (seconds) applied before upsampling. Default is 0.
    dt_rtol : float, optional
        Relative tolerance for step-size consistency; triggers resampling if exceeded. Default is 0.1.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        tracetime : np.ndarray
            Preprocessed timestamps.
        trace : np.ndarray
            Preprocessed trace.
    """
    dt, dt_rel_error = get_mean_dt(tracetime)

    if dt_rel_error > dt_rtol:
        warnings.warn('Inconsistent step-sizes in trace, resample trace.')
        tracetime, trace = filter_utils.resample_trace(tracetime=tracetime, trace=trace, dt=dt)

    if lowpass_cutoff > 0:
        trace = lowpass_filter_trace(trace=trace, fs=1. / dt, f_cutoff=lowpass_cutoff)

    if pre_blur_sigma_s > 0:
        trace = gaussian_filter(trace, sigma=pre_blur_sigma_s / dt, mode='nearest')

    tracetime, trace = transform_trace(tracetime, trace, kind=fit_kind, fupsample=fupsample_trace, dt=dt)
    return tracetime, trace


def transform_trace(tracetime: np.ndarray, trace: np.ndarray,
                    kind: str = 'trace',
                    fupsample: int = None,
                    dt: float = None) -> tuple[np.ndarray, np.ndarray]:
    """Transform the neural trace according to the specified method.

    Parameters
    ----------
    tracetime : np.ndarray
        Timestamps for the trace, shape (n_timepoints,).
    trace : np.ndarray
        Neural response trace, shape (n_timepoints,).
    kind : str, optional
        Transformation method: 'trace', 'gradient', or 'events'. Default is 'trace'.
    fupsample : int, optional
        Upsampling factor. Default is None (no upsampling).
    dt : float, optional
        Time step in seconds; required if fupsample > 1.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        fit_tracetime : np.ndarray
            Transformed (and possibly upsampled) timestamps.
        fit_trace : np.ndarray
            Transformed (and possibly upsampled) trace.

    Raises
    ------
    NotImplementedError
        If kind is not 'trace', 'gradient', or 'events'.
    """
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


def finalize_trace(trace: np.ndarray, dt: float,
                   post_blur_sigma_s: float) -> np.ndarray:
    """Apply post-processing to the trace: optional blur, z-score, and clipping.

    Parameters
    ----------
    trace : np.ndarray
        Neural response trace, shape (n_timepoints,).
    dt : float
        Time step in seconds.
    post_blur_sigma_s : float
        Gaussian blur sigma in seconds. 0 disables blurring.

    Returns
    -------
    np.ndarray
        Finalized trace, clipped to [-7, 7] after z-score normalization.
    """
    if post_blur_sigma_s > 0:
        trace = gaussian_filter(trace, sigma=post_blur_sigma_s / dt, mode='nearest')

    trace = trace / np.std(trace)
    trace = np.clip(trace, -7, 7)  # Clip extreme outliers

    return trace


def finalize_stim(stim: np.ndarray) -> np.ndarray:
    """Normalize and convert stimulus to a canonical format.

    Converts binary stimuli to {-1, 1}, z-scores continuous stimuli,
    or passes through index-based stimuli unchanged.

    Parameters
    ----------
    stim : np.ndarray
        Stimulus array, shape (n_frames, ...).

    Returns
    -------
    np.ndarray
        Stimulus converted to a standard format suitable for RF estimation.
    """
    if stim.ndim > 1:  # Otherwise assume stimulus indexes have been passed
        if 'bool' in str(stim.dtype) or set(np.unique(stim).astype(float)) == {0., 1.}:
            stim = 2 * stim.astype(np.int8) - 1
        elif set(np.unique(stim).astype(float)) == {-1, 1}:
            pass
        elif set(np.unique(stim).astype(float)) == {-1., 1.}:
            stim = stim.astype(np.int8)
        else:
            stim = math_utils.normalize_zscore(stim)
    return stim
