import numpy as np
import datajoint as dj
from matplotlib import pyplot as plt
from scipy import signal

from djimaging.utils.scanm_utils import load_traces_from_h5_file, split_trace_by_reps
from djimaging.utils.dj_utils import PlaceholderTable


class TracesTemplate(dj.Computed):
    database = ""  # hack to suppress DJ error

    @property
    def definition(self):
        definition = """
        # Raw Traces for each roi under a specific presentation
    
        -> self.presentation_table
        -> self.roi_table
    
        ---
        trace          :longblob              # array of raw trace
        trace_times    :longblob              # numerical array of trace times
        trace_flag     :tinyint unsigned      # flag if values in trace are correct(1) or not(0)
        trigger_flag   :tinyint unsigned      # flag if triggertimes aren't outside tracetimes
        """
        return definition

    presentation_table = PlaceholderTable
    field_table = PlaceholderTable
    roi_table = PlaceholderTable

    def make(self, key):

        # get all params we need for creating trace
        filepath = (self.presentation_table() & key).fetch1("h5_header")
        triggertimes = (self.presentation_table() & key).fetch1("triggertimes")
        roi_ids = (self.roi_table() & key).fetch("roi_id")

        roi2trace = load_traces_from_h5_file(filepath, roi_ids)

        for roi_id, roi_data in roi2trace.items():
            trace_key = key.copy()
            trace_key['roi_id'] = roi_id
            trace_key['trace'] = roi_data['trace']
            trace_key['trace_times'] = roi_data['trace_times']
            trace_key['trace_flag'] = roi_data['trace_flag']

            if trace_key['trace_flag']:
                if triggertimes[0] < trace_key['trace_times'][0]:
                    trace_key["trigger_flag"] = 0
                elif trace_key['trace_flag'] and triggertimes[-1] > trace_key['trace_times'][-1]:
                    trace_key["trigger_flag"] = 0
                else:
                    trace_key["trigger_flag"] = 1
            else:
                trace_key["trigger_flag"] = 0

            self.insert1(trace_key)

    def plot1(self, key):
        trace_times, trace = (self & key).fetch1("trace_times", "trace")
        triggertimes = (self.presentation_table() & key).fetch1("triggertimes")

        fig, ax = plt.subplots(1, 1, figsize=(10, 2))
        ax.plot(trace_times, trace)
        ax.set(xlabel='trace_times', ylabel='trace')
        ax.vlines(triggertimes, np.min(trace), np.max(trace), color='r', label='trigger')
        ax.legend(loc='upper right')
        plt.show()


class PreprocessParamsTemplate(dj.Lookup):
    database = ""  # hack to suppress DJ error

    @property
    def definition(self):
        definition = """
        preprocess_id:          int       # unique param set id
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
        # TODO: clean!

        window_len_seconds = (self.preprocessparams_table() & key).fetch1('window_length')
        poly_order = (self.preprocessparams_table() & key).fetch1('poly_order')
        subtract_baseline = (self.preprocessparams_table() & key).fetch1('subtract_baseline')
        non_negative = (self.preprocessparams_table() & key).fetch1('non_negative')
        standardize = (self.preprocessparams_table() & key).fetch1('standardize')
        fs = (self.presentation_table() & key).fetch1('scan_frequency')

        assert not (non_negative and subtract_baseline), \
            "You are trying to populate with an invalid parameter set"
        assert (np.logical_or(standardize == non_negative, standardize == subtract_baseline)), \
            "You are trying to populate with an invalid parameter set"

        trace_times = (self.traces_table() & key).fetch1('trace_times')
        raw_trace = (self.traces_table() & key).fetch1('trace').copy()
        raw_trace[0] = raw_trace[1]  # Drop first value

        window_len_frames = np.ceil(window_len_seconds * fs)
        if window_len_frames % 2 == 0:
            window_len_frames -= 1
        window_len_frames = int(window_len_frames)
        smoothed_trace = signal.savgol_filter(raw_trace, window_length=window_len_frames, polyorder=poly_order)
        preprocess_trace = raw_trace - smoothed_trace

        stim_start = None
        if standardize or subtract_baseline:
            stim_start = (self.presentation_table() & key).fetch1('triggertimes')[0]
            # heuristic to find out whether triggers are in time base or in frame base
            if stim_start > 1000:
                print("Converting triggers from frame base to time base")
                stim_start /= 500

            assert np.any(trace_times < stim_start), \
                f"stim_start={stim_start:.1g}, traces_starts at {trace_times.min():.1g}: key={key}"

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


class SnippetsTemplate(dj.Computed):
    database = ""  # hack to suppress DJ error

    @property
    def definition(self):
        definition = """
        # Snippets created from slicing traces using the triggertimes. 
        -> self.preprocesstraces_table
        ---
        snippets               :longblob          # array of snippets (time x repetitions)
        snippets_times         :longblob          # array of snippet times (time x repetitions)
        triggertimes_snippets  :longblob          # snippeted triggertimes (ntrigger_rep x repetitions)
        droppedlastrep_flag    :tinyint unsigned  # Was the last repetition incomplete and therefore dropped?
        """
        return definition

    preprocesstraces_table = PlaceholderTable
    stimulus_table = PlaceholderTable
    presentation_table = PlaceholderTable
    traces_table = PlaceholderTable

    @property
    def key_source(self):
        return self.preprocesstraces_table() * (self.stimulus_table() & "isrepeated=1")

    def make(self, key):
        ntrigger_rep = (self.stimulus_table() & key).fetch1('ntrigger_rep')
        triggertimes = (self.presentation_table() & key).fetch1('triggertimes')
        trace_times = (self.traces_table() & key).fetch1('trace_times')
        traces = (self.preprocesstraces_table() & key).fetch1('preprocess_trace')

        if triggertimes[-1] > 2 * trace_times[-1]:
            triggertimes = triggertimes / 500.

        snippets, snippets_times, triggertimes_snippets, droppedlastrep_flag = split_trace_by_reps(
            traces, trace_times, triggertimes, ntrigger_rep, allow_drop_last=True)

        self.insert1(dict(
            **key,
            snippets=snippets,
            snippets_times=snippets_times,
            triggertimes_snippets=triggertimes_snippets,
            droppedlastrep_flag=droppedlastrep_flag,
        ))

    def plot1(self, key):
        key = {k: v for k, v in key.items() if k in self.primary_key}

        snippets, snippets_times, triggertimes_snippets = (self & key).fetch1(
            "snippets", "snippets_times", "triggertimes_snippets")

        fig, ax = plt.subplots(1, 1, figsize=(10, 2))
        ax.plot(snippets_times - snippets_times[0, :], snippets)
        ax.set(ylabel='preprocess_trace')
        ax.vlines(triggertimes_snippets - triggertimes_snippets[0, :],
                  np.min(snippets), np.max(snippets), color='r', label='trigger')
        ax.legend(loc='upper right')
        plt.show()


class AveragesTemplate(dj.Computed):
    database = ""  # hack to suppress DJ error

    @property
    def definition(self):
        definition = """
        # Averages of snippets
    
        -> self.snippets_table
        ---
        average             :longblob  # array of snippet average (time)
        average_norm        :longblob  # normalized array of snippet average (time)
        average_times       :longblob  # array of average time, starting at t=0 (time)
        triggertimes_rel    :longblob  # array of relative triggertimes 
        """
        return definition

    snippets_table = PlaceholderTable

    def make(self, key):
        snippets, times = (self.snippets_table() & key).fetch1('snippets', 'snippets_times')
        triggertimes_snippets = (self.snippets_table() & key).fetch1('triggertimes_snippets').copy()

        times = times - times[0, :]

        if np.any(np.std(times, axis=1) > 1e-4):
            print(f'ERROR: Failed to compute average for {key}, tracetimes cannot be aligned without interpolation')
            return

        average_times = np.mean(times, axis=1)
        average = np.mean(snippets, axis=1)
        average_norm = (average - np.mean(average)) / np.std(average)
        triggertimes_rel = np.mean(triggertimes_snippets - triggertimes_snippets[0, :], axis=1)

        self.insert1(dict(
            **key,
            average=average,
            average_norm=average_norm,
            average_times=average_times,
            triggertimes_rel=triggertimes_rel,
        ))

    def plot1(self, key):
        key = {k: v for k, v in key.items() if k in self.primary_key}

        average, average_norm, average_times, triggertimes_rel = \
            (self & key).fetch1('average', 'average_norm', 'average_times', 'triggertimes_rel')

        fig, ax = plt.subplots(1, 1, figsize=(10, 2))
        ax.plot(average_times, average)
        ax.set(xlabel='average_times', ylabel='average')
        ax.vlines(triggertimes_rel, np.min(average), np.max(average), color='r', label='trigger')
        ax.legend(loc='upper right')
        tax = ax.twinx()
        tax.plot(average_times, average_norm)
        tax.set(ylabel='average_norm')
        plt.show()
