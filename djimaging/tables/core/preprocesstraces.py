import datajoint as dj
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

from djimaging.utils.dj_utils import PlaceholderTable


def detrend_trace(trace_times, raw_trace, poly_order, window_len_seconds, fs,
                  subtract_baseline: bool, standardize: bool, non_negative: bool, stim_start: float = None):
    """Detrend trace"""
    # TODO: clean!

    raw_trace = raw_trace.copy()
    raw_trace[0] = raw_trace[1]  # Drop first value

    window_len_frames = np.ceil(window_len_seconds * fs)
    if window_len_frames % 2 == 0:
        window_len_frames -= 1
    window_len_frames = int(window_len_frames)
    smoothed_trace = signal.savgol_filter(raw_trace, window_length=window_len_frames, polyorder=poly_order)
    preprocess_trace = raw_trace - smoothed_trace

    if standardize or subtract_baseline:
        # heuristic to find out whether triggers are in time base or in frame base
        if stim_start > 1000:
            print("Converting triggers from frame base to time base")
            stim_start /= 500
        assert np.any(trace_times < stim_start), f"stim_start={stim_start:.1g}, trace_start={trace_times.min():.1g}"

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
    database = ""  # hack to suppress DJ error

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
    database = ""  # hack to suppress DJ error

    @property
    def definition(self):
        definition = """
        # performs basic preprocessing on raw traces
        -> self.traces_table
        -> self.preprocessparams_table
        ---
        preprocess_trace:      longblob    # preprocessed trace
        smoothed_trace:        longblob    # output of savgol filter which is subtracted from the raw trace
        """
        return definition

    traces_table = PlaceholderTable
    preprocessparams_table = PlaceholderTable
    presentation_table = PlaceholderTable

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
        raw_trace = (self.traces_table() & key).fetch1('trace')
        stim_start = (self.presentation_table() & key).fetch1('triggertimes')[0]

        preprocess_trace, smoothed_trace = detrend_trace(
            trace_times=trace_times, raw_trace=raw_trace, stim_start=stim_start,
            poly_order=poly_order, window_len_seconds=window_len_seconds, fs=fs,
            subtract_baseline=subtract_baseline, standardize=standardize, non_negative=non_negative)

        self.insert1(dict(key, preprocess_trace=preprocess_trace, smoothed_trace=smoothed_trace))

    def plot1(self, key: dict):
        key = {k: v for k, v in key.items() if k in self.primary_key}

        preprocess_trace, smoothed_trace = (self & key).fetch1("preprocess_trace", "smoothed_trace")
        trace_times = (self.traces_table() & key).fetch1("trace_times")
        triggertimes = (self.presentation_table() & key).fetch1("triggertimes")

        fig, axs = plt.subplots(2, 1, figsize=(10, 4), sharex='all')
        ax = axs[0]
        ax.plot(trace_times, preprocess_trace)
        ax.set(ylabel='preprocess_trace')
        ax.vlines(triggertimes, np.min(preprocess_trace), np.max(preprocess_trace), color='r', label='trigger')
        ax.legend(loc='upper right')
        ax = axs[1]
        ax.plot(trace_times, smoothed_trace)
        ax.set(xlabel='trace_times', ylabel='smoothed_trace')
        ax.vlines(triggertimes, np.min(smoothed_trace), np.max(smoothed_trace), color='r', label='trigger')
        ax.legend(loc='upper right')
        plt.show()
