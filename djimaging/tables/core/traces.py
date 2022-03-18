import numpy as np
import datajoint as dj
from matplotlib import pyplot as plt
from copy import deepcopy
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
        traces                  :longblob              # array of raw traces
        traces_times            :longblob              # numerical array of trace times
        traces_flag             :tinyint unsigned      # flag if values in traces are correct(1) or not(0)
        trigger_flag = 1        :tinyint unsigned      # flag if triggertimes aren't outside tracetimes
        """
        return definition

    presentation_table = PlaceholderTable
    field_table = PlaceholderTable
    roi_table = PlaceholderTable

    def make(self, key):

        # get all params we need for creating traces
        filepath = (self.presentation_table() & key).fetch1("h5_header")
        triggertimes = (self.presentation_table() & key).fetch1("triggertimes")
        roi_ids = (self.roi_table() & key).fetch("roi_id")

        roi2trace = load_traces_from_h5_file(filepath, roi_ids)

        for roi_id, roi_data in roi2trace.items():
            trace_key = key.copy()
            trace_key['roi_id'] = roi_id
            trace_key['traces'] = roi_data['trace']
            trace_key['traces_times'] = roi_data['trace_times']
            trace_key['traces_flag'] = roi_data['trace_flag']

            if trace_key['traces_flag']:
                if triggertimes[0] < trace_key['traces_times'][0]:
                    trace_key["trigger_flag"] = 0
                elif trace_key['traces_flag'] and triggertimes[-1] > trace_key['traces_times'][-1]:
                    trace_key["trigger_flag"] = 0
            else:
                trace_key["trigger_flag"] = 0

            self.insert1(trace_key)

    def plot1(self, key):
        traces_times, traces = (self & key).fetch1("traces_times", "traces")
        triggertimes = (self.presentation_table() & key).fetch1("triggertimes")

        fig, ax = plt.subplots(1, 1, figsize=(10, 2))
        ax.plot(traces_times, traces)
        ax.set(xlabel='traces_times', ylabel='traces')
        ax.vlines(triggertimes, np.min(traces), np.max(traces), color='r', label='trigger')
        ax.legend(loc='upper right')
        plt.show()


class DetrendParamsTemplate(dj.Lookup):
    database = ""  # hack to suppress DJ error

    @property
    def definition(self):
        definition = """
        detrend_param_set_id:           int       # unique param set id
        ---
        window_length=60:               int       # window length for SavGol filter in seconds
        poly_order=3:                   int       # order of polynomial for savgol filter
        non_negative=0:                 tinyint unsigned
        subtract_baseline=0:            tinyint unsigned
        standardize=1:                  tinyint unsigned  # whether to standardize (divide by sd)
        """
        return definition


class DetrendTracesTemplate(dj.Computed):
    database = ""  # hack to suppress DJ error

    @property
    def definition(self):
        definition = """
        # performs basic preprocessing on raw traces
        -> self.detrendparams_table
        -> self.traces_table
        ---
        detrend_traces:         longblob        # detrended traces
        smoothed_traces:        longblob        # output of savgol filter which is subtracted from the raw traces
        """
        return definition

    presentation_table = PlaceholderTable
    detrendparams_table = PlaceholderTable
    traces_table = PlaceholderTable

    def make(self, key):

        window_len_seconds = (self.detrendparams_table() & key).fetch1('window_length')
        poly_order = (self.detrendparams_table() & key).fetch1('poly_order')
        subtract_baseline = (self.detrendparams_table() & key).fetch1('subtract_baseline')
        non_negative = (self.detrendparams_table() & key).fetch1('non_negative')
        standardize = (self.detrendparams_table() & key).fetch1('standardize')
        fs = (self.presentation_table() & key).fetch1('scan_frequency')

        assert not (non_negative and subtract_baseline), \
            "You are trying to populate DetrendTraces with an invalid parameter set"
        assert (np.logical_or(standardize == non_negative, standardize == subtract_baseline)), \
            "You are trying to populate DetrendTraces with an invalid parameter set"

        raw_traces = (self.traces_table() & key).fetch1('traces')
        temp = deepcopy(raw_traces)  # TODO: what is happening here?
        temp[0] = temp[1]
        raw_traces = temp
        traces_times = (self.traces_table() & key).fetch1('traces_times')
        window_len_frames = np.ceil(window_len_seconds * fs)
        if window_len_frames % 2 == 0:
            window_len_frames -= 1
        window_len_frames = int(window_len_frames)
        smoothed_traces = \
            signal.savgol_filter(raw_traces, window_length=window_len_frames, polyorder=poly_order)
        detrend_traces = raw_traces - smoothed_traces

        stim_start = None
        if standardize or subtract_baseline:
            stim_start = (self.presentation_table() & key).fetch1('triggertimes')[0]
            # heuristic to find out whether triggers are in time base or in frame base
            if stim_start > 1000:
                print("Converting triggers from frame base to time base")
                stim_start /= 500

            assert np.any(traces_times < stim_start), \
                f"stim_start={stim_start:.1g}, traces_starts at {traces_times.min():.1g}: key={key}"

        if non_negative:
            clip_value = np.percentile(detrend_traces, q=2.5)
            detrend_traces[detrend_traces < clip_value] = clip_value
            detrend_traces = detrend_traces - clip_value
            if standardize:
                # find last frame recorded before stimulus started
                baseline_end = np.nonzero(traces_times[traces_times < stim_start])[0][-1]
                baseline = detrend_traces[:baseline_end]
                detrend_traces = detrend_traces / np.std(baseline)
        elif subtract_baseline:
            # find last frame recorded before stimulus started
            baseline_end = np.nonzero(traces_times[traces_times < stim_start])[0][-1]
            baseline = detrend_traces[:baseline_end]
            detrend_traces = detrend_traces - np.median(baseline)

            if standardize:
                detrend_traces = detrend_traces / np.std(baseline)

        self.insert1(dict(key, detrend_traces=detrend_traces, smoothed_traces=smoothed_traces))

    def plot1(self, key):
        detrend_traces, smoothed_traces = (self & key).fetch1("detrend_traces", "smoothed_traces")
        traces_times = (self.traces_table() & key).fetch1("traces_times")
        triggertimes = (self.presentation_table() & key).fetch1("triggertimes")

        fig, axs = plt.subplots(2, 1, figsize=(10, 4), sharex='all')
        ax = axs[0]
        ax.plot(traces_times, detrend_traces)
        ax.set(ylabel='detrend_traces')
        ax.vlines(triggertimes, np.min(detrend_traces), np.max(detrend_traces), color='r', label='trigger')
        ax.legend(loc='upper right')
        ax = axs[1]
        ax.plot(traces_times, smoothed_traces)
        ax.set(xlabel='traces_times', ylabel='smoothed_traces')
        ax.vlines(triggertimes, np.min(smoothed_traces), np.max(smoothed_traces), color='r', label='trigger')
        ax.legend(loc='upper right')
        plt.show()


class DetrendSnippetsTemplate(dj.Computed):
    database = ""  # hack to suppress DJ error

    @property
    def definition(self):
        definition = """
        # Snippets created from slicing filtered traces using the triggertimes. 
        -> self.detrendtraces_table
        ---
        detrend_snippets         :longblob     # array of snippets (time x repetitions)
        snippets_times           :longblob     # array of snippet times (time x repetitions)
        smoothed_snippets        :longblob     # snippeted, smoothed signal (time x repetitions)
        triggertimes_snippets    :longblob     # snippeted triggertimes (ntrigger_rep x repetitions)
        droppedlastrep_flag      :tinyint unsigned 
        """
        return definition

    stimulus_table = PlaceholderTable
    presentation_table = PlaceholderTable
    traces_table = PlaceholderTable
    detrendtraces_table = PlaceholderTable

    @property
    def key_source(self):
        return self.detrendtraces_table() * (self.stimulus_table() & "isrepeated=1")

    def make(self, key):
        ntrigger_rep = (self.stimulus_table() & key).fetch1('ntrigger_rep')
        triggertimes = (self.presentation_table() & key).fetch1('triggertimes')
        traces_times = (self.traces_table() & key).fetch1('traces_times')
        detrend_traces, smoothed_traces = (self.detrendtraces_table() & key).fetch1('detrend_traces', 'smoothed_traces')

        if triggertimes[-1] > 2 * traces_times[-1]:
            triggertimes = triggertimes / 500.

        snippets_times, triggertimes_snippets, snippets_list, droppedlastrep_flag = split_trace_by_reps(
            triggertimes=triggertimes, ntrigger_rep=ntrigger_rep, times=traces_times,
            trace_list=[detrend_traces, smoothed_traces], allow_drop_last=True)

        self.insert1(dict(
            **key,
            detrend_snippets=snippets_list[0],
            smoothed_snippets=snippets_list[1],
            snippets_times=snippets_times,
            triggertimes_snippets=triggertimes_snippets,
            droppedlastrep_flag=droppedlastrep_flag,
        ))

    def plot1(self, key):
        detrend_snippets, smoothed_snippets, snippets_times, triggertimes_snippets = (self & key).fetch1(
            "detrend_snippets", "smoothed_snippets", "snippets_times", "triggertimes_snippets")

        fig, axs = plt.subplots(2, 1, figsize=(10, 4), sharex='all')
        ax = axs[0]
        ax.plot(snippets_times - snippets_times[0, :], detrend_snippets)
        ax.set(ylabel='detrend_traces')
        ax.vlines(triggertimes_snippets - triggertimes_snippets[0, :],
                  np.min(detrend_snippets), np.max(detrend_snippets), color='r', label='trigger')
        ax.legend(loc='upper right')
        ax = axs[1]
        ax.plot(snippets_times - snippets_times[0, :], smoothed_snippets)
        ax.set(xlabel='relative snippets_times', ylabel='smoothed_traces')
        ax.vlines(triggertimes_snippets - triggertimes_snippets[0, :],
                  np.min(smoothed_snippets), np.max(smoothed_snippets), color='r', label='trigger')
        ax.legend(loc='upper right')
        plt.show()


class AveragesTemplate(dj.Computed):
    database = ""  # hack to suppress DJ error

    @property
    def definition(self):
        definition = """
        # Snippets created from slicing filtered traces using the triggertimes. 
    
        -> self.detrendsnippets_table
        ---
        average             :longblob  # array of snippet average (time)
        average_norm        :longblob  # normalized array of snippet average (time)
        average_times       :longblob  # array of average time, starting at t=0 (time)
        triggertimes_rel    :longblob  # array of relative triggertimes 
        """
        return definition

    detrendsnippets_table = PlaceholderTable

    def make(self, key):
        snippets, times = (self.detrendsnippets_table() & key).fetch1('detrend_snippets', 'snippets_times')
        triggertimes_snippets = (self.detrendsnippets_table() & key).fetch1('triggertimes_snippets').copy()

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
