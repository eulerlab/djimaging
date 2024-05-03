from abc import abstractmethod

import datajoint as dj
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

from djimaging.utils import filter_utils, math_utils, plot_utils, trace_utils
from djimaging.utils.dj_utils import get_primary_key
from djimaging.utils.filter_utils import lowpass_filter_trace
from djimaging.utils.plot_utils import plot_trace_and_trigger


class PreprocessParamsTemplate(dj.Lookup):
    database = ""

    @property
    def definition(self):
        definition = """
        -> self.stimulus_table
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

    @property
    @abstractmethod
    def stimulus_table(self):
        pass

    def add_default(self, preprocess_id=1, stim_names=None, window_length=60, poly_order=3, non_negative=False,
                    subtract_baseline=True, standardize=1, f_cutoff=None, fs_resample=None,
                    skip_duplicates=False):
        if stim_names is None:
            stim_names = (self.stimulus_table()).fetch('stim_name')

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

        for stim_name in stim_names:
            """Add default preprocess parameter to table"""
            stim_key = key.copy()
            stim_key['stim_name'] = stim_name
            self.insert1(stim_key, skip_duplicates=skip_duplicates)


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
        pp_trace: longblob    # preprocessed trace
        smoothed_trace:   longblob    # output of savgol filter which is subtracted from the raw trace
        pp_trace_t0:         float       # numerical array of trace times
        pp_trace_dt:         float       # time between frames
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
            return ((self.traces_table() & 'trace_valid=1' & 'trigger_valid=1') * self.preprocessparams_table()).proj()
        except (AttributeError, TypeError):
            pass

    def make(self, key):
        window_len_seconds, poly_order, subtract_baseline, non_negative, standardize, f_cutoff, fs_resample = \
            (self.preprocessparams_table() & key).fetch1(
                'window_length', 'poly_order', 'subtract_baseline', 'non_negative', 'standardize',
                'f_cutoff', 'fs_resample')

        trace, trace_t0, trace_dt = (self.traces_table() & key).fetch1('trace', 'trace_t0', 'trace_dt')
        triggertimes = (self.presentation_table() & key).fetch1('triggertimes')

        stim_start = triggertimes[0] if len(triggertimes) > 0 else None

        pp_trace, smoothed_trace, pp_trace_dt = process_trace(
            trace=trace, trace_t0=trace_t0, trace_dt=trace_dt,
            stim_start=stim_start, poly_order=poly_order, window_len_seconds=window_len_seconds,
            subtract_baseline=subtract_baseline, standardize=standardize, non_negative=non_negative,
            f_cutoff=f_cutoff, fs_resample=fs_resample, baseline_max_dt=self._baseline_max_dt)

        self.insert1(dict(
            key,
            pp_trace_t0=trace_t0,
            pp_trace_dt=pp_trace_dt,
            pp_trace=pp_trace.astype(np.float32),
            smoothed_trace=smoothed_trace.astype(np.float32)
        ))

    def plot1(self, key=None, xlim=None, ylim=None):
        key = get_primary_key(self, key)

        pp_trace_t0, pp_trace_dt, pp_trace, smoothed_trace = (self & key).fetch1(
            "pp_trace_t0", "pp_trace_dt", "pp_trace", "smoothed_trace")
        trace_t0, trace_dt, trace = (self.traces_table() & key).fetch1("trace_t0", "trace_dt", "trace")
        triggertimes = (self.presentation_table() & key).fetch1("triggertimes")

        trace_times = np.arange(trace.size) * trace_dt + pp_trace_t0
        pp_trace_times = np.arange(pp_trace.size) * pp_trace_dt + pp_trace_t0

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
        plot_trace_and_trigger(time=pp_trace_times, trace=pp_trace, triggertimes=triggertimes, ax=ax)
        ax.set(ylabel='preprocessed\ntrace')

        ax.set(xlim=xlim, ylim=ylim)

        plt.show()

    def plot(self, restriction=None, sort=True):
        if restriction is None:
            restriction = dict()

        preprocess_traces = (self & restriction).fetch("pp_trace")
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


def process_trace(trace, trace_t0, trace_dt,
                  poly_order=3, window_len_seconds=0,
                  subtract_baseline: bool = False, standardize: int = 0,
                  non_negative: bool = False, stim_start: float = None,
                  f_cutoff: float = None, fs_resample: float = None,
                  drop_nmin_lr=(0, 0), drop_nmax_lr=(3, 3),
                  baseline_max_dt=np.inf):
    """Detrend and preprocess trace"""
    assert standardize in [0, 1, 2], standardize

    trace_times = np.arange(trace.size) * trace_dt + trace_t0
    fs = 1. / trace_dt

    trace = np.asarray(trace).copy()

    trace = drop_left_and_right(trace, drop_nmin_lr=drop_nmin_lr, drop_nmax_lr=drop_nmax_lr, inplace=True)

    if (f_cutoff is not None) and (f_cutoff > 0):
        trace = lowpass_filter_trace(trace=trace, fs=fs, f_cutoff=f_cutoff)

    if window_len_seconds > 0:
        trace, smoothed_trace = detrend_trace(trace, window_len_seconds, fs, poly_order)
    else:
        smoothed_trace = np.zeros_like(trace)

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
            fdownsample = int(np.round(fs / fs_resample))
            trace_dt_new = trace_dt * fdownsample
            trace_times, trace = filter_utils.downsample_trace(
                tracetime=trace_times, trace=trace, fdownsample=fdownsample)
        else:
            trace_dt_new = 1. / fs_resample
            trace_times, trace = filter_utils.resample_trace(tracetime=trace_times, trace=trace, dt=trace_dt_new)
    else:
        trace_dt_new = trace_dt

    return trace, smoothed_trace, trace_dt_new
