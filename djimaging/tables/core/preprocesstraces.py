import warnings
from abc import abstractmethod

import datajoint as dj
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

from djimaging.utils.dj_utils import get_primary_key
from djimaging.utils.plot_utils import plot_trace_and_trigger


def drop_left_and_right(trace, drop_nmin_lr=(0, 0), drop_nmax_lr=(3, 3), inplace=False):
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


def detrend_trace(trace_times, trace, poly_order, window_len_seconds, fs,
                  subtract_baseline: bool, standardize: bool, non_negative: bool, stim_start: float = None,
                  drop_nmin_lr=(0, 0), drop_nmax_lr=(3, 3)):
    """Detrend trace"""
    raw_trace = drop_left_and_right(trace, drop_nmin_lr=drop_nmin_lr, drop_nmax_lr=drop_nmax_lr, inplace=False)

    window_len_frames = np.ceil(window_len_seconds * fs)
    if window_len_frames % 2 == 0:
        window_len_frames -= 1
    window_len_frames = int(window_len_frames)
    if len(raw_trace) >= window_len_frames:
        smoothed_trace = signal.savgol_filter(raw_trace, window_length=window_len_frames, polyorder=poly_order)
    else:
        smoothed_trace = signal.savgol_filter(raw_trace, window_length=window_len_frames, polyorder=poly_order,
                                              mode='nearest')

    preprocess_trace = raw_trace - smoothed_trace

    if standardize or subtract_baseline:
        # heuristic to find out whether triggers are in time base or in frame base
        if stim_start > 1000:
            print("Converting triggers from frame base to time base")
            stim_start /= 500
        if not np.any(trace_times < stim_start):
            raise ValueError(f"stim_start={stim_start:.1g}, trace_start={trace_times.min():.1g}")

    if non_negative:
        clip_value = np.percentile(preprocess_trace, q=2.5)
        preprocess_trace[preprocess_trace < clip_value] = clip_value
        preprocess_trace = preprocess_trace - clip_value
        if standardize:
            # find last frame recorded before stimulus started
            baseline_end = np.nonzero(trace_times[trace_times < stim_start])[0][-1]
            baseline = preprocess_trace[:baseline_end]
            preprocess_trace = preprocess_trace / np.std(baseline)
    elif subtract_baseline:
        # find last frame recorded before stimulus started
        baseline_end = np.nonzero(trace_times[trace_times < stim_start])[0][-1]
        baseline = preprocess_trace[:baseline_end]
        preprocess_trace = preprocess_trace - np.median(baseline)

        if standardize:
            preprocess_trace = preprocess_trace / np.std(baseline)

    return preprocess_trace, smoothed_trace


class PreprocessParamsTemplate(dj.Lookup):
    database = ""

    @property
    def definition(self):
        definition = """
        preprocess_id:       int       # unique param set id
        ---
        window_length:       int       # window length for SavGol filter in seconds
        poly_order:          int       # order of polynomial for savgol filter
        non_negative:        tinyint unsigned
        subtract_baseline:   tinyint unsigned
        standardize:         tinyint unsigned  # whether to standardize (divide by sd)
        """
        return definition

    def add_default(self, skip_duplicates=False):
        """Add default preprocess parameter to table"""
        key = {
            'preprocess_id': 1,
            'window_length': 60,
            'poly_order': 3,
            'non_negative': 0,
            'subtract_baseline': 1,
            'standardize': 1,
        }
        self.insert1(key, skip_duplicates=skip_duplicates)


class PreprocessTracesTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = """
        # performs basic preprocessing on raw traces
        -> self.traces_table
        -> self.preprocessparams_table
        ---
        preprocess_trace_times: longblob    # Time of preprocessed trace, if not resampled same as in trace.
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
            return (self.traces_table() & 'trace_flag=1' & 'trigger_flag=1') * self.preprocessparams_table()
        except TypeError:
            pass

    def make(self, key):
        window_len_seconds = (self.preprocessparams_table() & key).fetch1('window_length')
        poly_order = (self.preprocessparams_table() & key).fetch1('poly_order')
        subtract_baseline = (self.preprocessparams_table() & key).fetch1('subtract_baseline')
        non_negative = (self.preprocessparams_table() & key).fetch1('non_negative')
        standardize = (self.preprocessparams_table() & key).fetch1('standardize')
        fs = (self.presentation_table.ScanInfo() & key).fetch1('scan_frequency')

        assert not (non_negative and subtract_baseline), \
            "You are trying to populate with an invalid parameter set"
        assert (np.logical_or(standardize == non_negative, standardize == subtract_baseline)), \
            "You are trying to populate with an invalid parameter set"

        trace_times = (self.traces_table() & key).fetch1('trace_times')
        trace = (self.traces_table() & key).fetch1('trace')
        stim_start = (self.presentation_table() & key).fetch1('triggertimes')[0]

        try:
            preprocess_trace, smoothed_trace = detrend_trace(
                trace_times=trace_times, trace=trace, stim_start=stim_start,
                poly_order=poly_order, window_len_seconds=window_len_seconds, fs=fs,
                subtract_baseline=subtract_baseline, standardize=standardize, non_negative=non_negative)

            self.insert1(dict(key, preprocess_trace_times=trace_times,
                              preprocess_trace=preprocess_trace, smoothed_trace=smoothed_trace))
        except ValueError as e:
            warnings.warn(f"{e}\nfor key\n{key}")

    def plot1(self, key=None):
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

        plt.show()
