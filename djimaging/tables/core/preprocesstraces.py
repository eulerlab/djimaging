import warnings
from abc import abstractmethod

import datajoint as dj
import numpy as np
from djimaging.utils import filter_utils, math_utils, plot_utils, trace_utils
from matplotlib import pyplot as plt
from scipy import signal

from djimaging.utils.dj_utils import get_primary_key
from djimaging.utils.filter_utils import lowpass_filter_trace
from djimaging.utils.plot_utils import plot_trace_and_trigger
from djimaging.utils.trace_utils import get_mean_dt


class PreprocessParamsTemplate(dj.Lookup):
    database = ""

    @property
    def definition(self):
        definition = """
        preprocess_id:       tinyint unsigned    # unique param set id
        ---
        window_length:       int       # window length for SavGol filter in seconds
        poly_order:          int       # order of polynomial for savgol filter
        non_negative:        tinyint unsigned  # Clip negative values of trace
        subtract_baseline:   tinyint unsigned  # Subtract baseline
        standardize:         tinyint unsigned  # standardize (1: with sd of baseline, 2: sd of trace, 0: nothing)
        f_cutoff = 0 : float  # Cutoff frequency for low pass filter, only applied when > 0.
        fs_resample = 0 : float  # Resampling frequency, only applied when > 0.
        """
        return definition

    def add_default(self, preprocess_id=1, window_length=60, poly_order=3, non_negative=False,
                    subtract_baseline=True, standardize=1, f_cutoff=None, fs_resample=None,
                    skip_duplicates=False):
        key = dict(
            preprocess_id=preprocess_id,
            window_length=window_length,
            poly_order=int(poly_order),
            non_negative=int(non_negative),
            subtract_baseline=int(subtract_baseline),
            standardize=int(standardize),
            f_cutoff=f_cutoff if f_cutoff is not None else 0,
            fs_resample=fs_resample if fs_resample is not None else 0,
        )
        """Add default preprocess parameter to table"""
        self.insert1(key, skip_duplicates=skip_duplicates)


class PreprocessTracesTemplate(dj.Computed):
    database = ""
    _baseline_max_dt = np.inf

    @property
    def definition(self):
        definition = """
        # performs basic preprocessing on raw traces
        -> self.traces_table
        -> self.preprocessparams_table
        ---
        preprocess_trace_times: longblob   # Time of preprocessed trace, if not resampled same as in trace.
        preprocess_trace:      longblob    # preprocessed trace
        smoothed_trace:        longblob    # output of savgol filter which is subtracted from the raw trace
        """
        return definition

    @property
    @abstractmethod
    def traces_table(self):
        pass

    @property
    @abstractmethod
    def preprocessparams_table(self):
        pass

    @property
    @abstractmethod
    def presentation_table(self):
        pass

    @property
    def key_source(self):
        try:
            return ((self.traces_table() & 'trace_flag=1' & 'trigger_valid=1') * self.preprocessparams_table()).proj()
        except (AttributeError, TypeError):
            pass

    def make(self, key):
        window_len_seconds, poly_order, subtract_baseline, non_negative, standardize, f_cutoff, fs_resample = \
            (self.preprocessparams_table() & key).fetch1(
                'window_length', 'poly_order', 'subtract_baseline', 'non_negative', 'standardize',
                'f_cutoff', 'fs_resample')

        trace_times, trace = (self.traces_table() & key).fetch1('trace_times', 'trace')
        stim_start = (self.presentation_table() & key).fetch1('triggertimes')[0]

        preprocess_trace_times, preprocess_trace, smoothed_trace = process_trace(
            trace_times=trace_times, trace=trace, stim_start=stim_start,
            poly_order=poly_order, window_len_seconds=window_len_seconds,
            subtract_baseline=subtract_baseline, standardize=standardize, non_negative=non_negative,
            f_cutoff=f_cutoff, fs_resample=fs_resample, baseline_max_dt=self._baseline_max_dt)

        self.insert1(dict(key, preprocess_trace_times=preprocess_trace_times,
                          preprocess_trace=preprocess_trace, smoothed_trace=smoothed_trace))

    def plot1(self, key=None, xlim=None, ylim=None):
        key = get_primary_key(self, key)

        preprocess_trace_times, preprocess_trace, smoothed_trace = (self & key).fetch1(
            "preprocess_trace_times", "preprocess_trace", "smoothed_trace")
        trace_times, trace = (self.traces_table() & key).fetch1("trace_times", "trace")
        triggertimes = (self.presentation_table() & key).fetch1("triggertimes")

        fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex='all')

        ax = axs[0]
        plot_trace_and_trigger(time=trace_times, trace=trace - np.mean(smoothed_trace),
                               triggertimes=triggertimes, ax=ax, title=str(key))
        ax.set(ylabel='mean subtracted\nraw trace')

        ax = axs[1]
        plot_trace_and_trigger(time=trace_times, trace=smoothed_trace - np.mean(smoothed_trace),
                               triggertimes=triggertimes, ax=ax)
        ax.set(ylabel='mean subtracted\ntrend')

        ax = axs[2]
        plot_trace_and_trigger(time=preprocess_trace_times, trace=preprocess_trace, triggertimes=triggertimes, ax=ax)
        ax.set(ylabel='preprocessed\ntrace')

        ax.set(xlim=xlim, ylim=ylim)

        plt.show()

    def plot(self, restriction=None, sort=True):
        if restriction is None:
            restriction = dict()

        preprocess_traces = (self & restriction).fetch("preprocess_trace")
        preprocess_traces = math_utils.padded_vstack(preprocess_traces, cval=np.nan)
        n = preprocess_traces.shape[0]

        fig, ax = plt.subplots(1, 1, figsize=(10, 1 + np.minimum(n * 0.1, 10)))
        if len(restriction) > 0:
            plot_utils.set_long_title(fig=fig, title=restriction)

        sort_idxs = trace_utils.argsort_traces(preprocess_traces, ignore_nan=True) if sort else np.arange(n)

        ax.set_title('preprocess_traces')
        plot_utils.plot_signals_heatmap(ax=ax, signals=preprocess_traces[sort_idxs, :], symmetric=False)
        plt.show()


def drop_left_and_right(trace, drop_nmin_lr=(0, 0), drop_nmax_lr=(3, 3), inplace: bool = False):
    """Drop left and right most values if they are out of limits of the rest of the trace.
    This is necessary because the first few and last few points may be affected by artifacts.
    This is relatively conservative.
    """
    if not inplace:
        trace = trace.copy()

    if (drop_nmax_lr[0] > 0) or (drop_nmax_lr[1] > 0):
        l_idx = drop_nmax_lr[0]
        r_idx = -drop_nmax_lr[1] if drop_nmax_lr[1] > 0 else trace.size
        trace_min = np.min(trace[l_idx:r_idx])
        trace_max = np.max(trace[l_idx:r_idx])
    else:
        l_idx, r_idx, trace_min, trace_max = None, None, None, None

    if drop_nmax_lr[0] > 0:
        l_oor = (trace[:l_idx] < trace_min) | (trace[:l_idx] > trace_max)
        if np.all(l_oor):
            l_ndrop = drop_nmax_lr[0]
        else:
            l_ndrop = np.argmin(l_oor)
        if l_ndrop > 0:
            trace[:l_ndrop] = trace[l_ndrop]

    if drop_nmax_lr[1] > 0:
        r_oor = ((trace[r_idx:] < trace_min) | (trace[r_idx:] > trace_max))
        if np.all(r_oor):
            r_ndrop = drop_nmax_lr[1]
        else:
            r_ndrop = np.argmin(r_oor[::-1])
        if r_ndrop > 0:
            trace[-r_ndrop:] = trace[-(r_ndrop + 1)]

    if drop_nmin_lr[0] > 0:
        trace[:drop_nmin_lr[0]] = trace[drop_nmin_lr[0]]
    if drop_nmin_lr[1] > 0:
        trace[-drop_nmin_lr[1]:] = trace[-drop_nmin_lr[1]]

    return trace


def detrend_trace(trace, window_len_seconds, fs, poly_order):
    window_len_frames = np.ceil(window_len_seconds * fs)

    if window_len_frames % 2 == 0:
        window_len_frames -= 1

    window_len_frames = int(window_len_frames)
    smoothed_trace = signal.savgol_filter(trace, window_length=window_len_frames, polyorder=poly_order, mode='nearest')
    detrended_trace = trace - smoothed_trace
    return detrended_trace, smoothed_trace


def extract_baseline(trace, trace_times, stim_start: float, max_dt=np.inf):
    """Compute baseline for trace"""
    if not np.any(trace_times < stim_start):
        raise ValueError(f"stim_start={stim_start:.1g}, trace_start={trace_times.min():.1g}")

    baseline_end = np.nonzero(trace_times[trace_times < stim_start])[0][-1]
    baseline_start = np.nonzero(trace_times >= trace_times[baseline_end] - max_dt)[0][0]
    baseline = trace[baseline_start:baseline_end]
    return baseline


def non_negative_trace(trace, inplace: bool = False):
    """Make trace positive, remove lowest 2.5 percentile (and standardize by STD of baseline)"""
    if not inplace:
        trace = trace.copy()

    clip_value = np.percentile(trace, q=2.5)
    trace[trace < clip_value] = clip_value
    trace -= clip_value

    return trace


def subtract_baseline_trace(trace, trace_times, stim_start: float, inplace: bool = False, max_dt=np.inf):
    """Subtract baseline of trace (and standardize by STD of baseline)"""
    if not inplace:
        trace = trace.copy()

    baseline = extract_baseline(trace, trace_times, stim_start, max_dt=max_dt)
    trace -= np.median(baseline)

    return trace


def process_trace(trace_times, trace, poly_order, window_len_seconds,
                  subtract_baseline: bool, standardize: int, non_negative: bool, stim_start: float = None,
                  f_cutoff: float = None, fs_resample: float = None, drop_nmin_lr=(0, 0), drop_nmax_lr=(3, 3),
                  dt_rtol=0.1, baseline_max_dt=np.inf):
    """Detrend and preprocess trace"""
    assert standardize in [0, 1, 2], standardize

    trace = np.asarray(trace).copy()
    dt, dt_rel_error = get_mean_dt(tracetime=trace_times)
    fs = 1. / dt

    if dt_rel_error > dt_rtol:
        warnings.warn('Inconsistent step-sizes in trace, resample trace.')
        tracetime, trace = filter_utils.resample_trace(tracetime=trace_times, trace=trace, dt=dt)

    trace = drop_left_and_right(trace, drop_nmin_lr=drop_nmin_lr, drop_nmax_lr=drop_nmax_lr, inplace=True)

    if (f_cutoff is not None) and (f_cutoff > 0):
        trace = lowpass_filter_trace(trace=trace, fs=fs, f_cutoff=f_cutoff)

    trace, smoothed_trace = detrend_trace(trace, window_len_seconds, fs, poly_order)

    if stim_start is not None:
        if not np.any(trace_times < stim_start):
            raise ValueError(f"stim_start={stim_start:.1g}, trace_start={trace_times.min():.1g}")

    if subtract_baseline:
        trace = subtract_baseline_trace(trace, trace_times, stim_start, max_dt=baseline_max_dt, inplace=True)

    if non_negative:
        trace = non_negative_trace(trace, inplace=True)

    if standardize == 1:
        baseline = extract_baseline(trace, trace_times, stim_start)
        trace /= np.std(baseline)
    elif standardize == 2:
        trace /= np.std(trace)

    if (fs_resample is not None) and (fs_resample != 0) and (fs != fs_resample):
        if (fs_resample < fs) and np.isclose((fs / fs_resample), np.round(fs / fs_resample)):
            trace_times, trace = filter_utils.downsample_trace(
                tracetime=trace_times, trace=trace, fdownsample=int(np.round(fs / fs_resample)))
        else:
            trace_times, trace = filter_utils.resample_trace(tracetime=trace_times, trace=trace, dt=1. / fs_resample)

    return trace_times, trace, smoothed_trace
