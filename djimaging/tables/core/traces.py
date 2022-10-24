import warnings

import datajoint as dj
import numpy as np
from matplotlib import pyplot as plt

from djimaging.utils.dj_utils import PlaceholderTable, get_plot_key
from djimaging.utils.plot_utils import plot_trace_and_trigger, plot_signals_heatmap
from djimaging.utils.scanm_utils import load_traces_from_h5_file, split_trace_by_reps


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
        trace_flag     :tinyint unsigned      # Are values in trace correct (1) or not (0)?
        trigger_flag   :tinyint unsigned      # Are triggertimes inside trace_times (1) or not (0)?
        """
        return definition

    presentation_table = PlaceholderTable
    field_table = PlaceholderTable
    roi_table = PlaceholderTable

    _include_artifacts = False

    @property
    def key_source(self):
        try:
            if self._include_artifacts:
                return self.roi_table() * self.presentation_table()
            else:
                return (self.roi_table() & 'artifact_flag=0') * self.presentation_table()
        except TypeError:
            pass

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
            trace_key['trace_flag'] = roi_data['valid_flag']

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

    def plot1(self, key=None):
        key = get_plot_key(table=self, key=key)
        trace_times, trace = (self & key).fetch1("trace_times", "trace")
        triggertimes = (self.presentation_table() & key).fetch1("triggertimes")

        plot_trace_and_trigger(
            time=trace_times, trace=trace, triggertimes=triggertimes, title=str(key))


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
        try:
            return self.preprocesstraces_table() * (self.stimulus_table() & "isrepeated=1")
        except TypeError:
            pass

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

    def plot1(self, key=None):
        key = get_plot_key(table=self, key=key)
        snippets, snippets_times, triggertimes_snippets = (self & key).fetch1(
            "snippets", "snippets_times", "triggertimes_snippets")

        plot_trace_and_trigger(
            time=snippets_times, trace=snippets, triggertimes=triggertimes_snippets, title=str(key))


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

    def plot1(self, key=None):
        key = get_plot_key(table=self, key=key)
        average, average_norm, average_times, triggertimes_rel = \
            (self & key).fetch1('average', 'average_norm', 'average_times', 'triggertimes_rel')

        plot_trace_and_trigger(
            time=average_times, trace=average, triggertimes=triggertimes_rel, trace_norm=average_norm, title=str(key))

    def plot(self, restriction=None):
        if restriction is None:
            restriction = dict()

        average = (self & restriction).fetch('average')

        sizes = [a.size for a in average]

        if np.unique(sizes).size > 1:
            warnings.warn('Traces do not have the same size. Are you plotting multiple stimuli?')
        min_size = np.min(sizes)

        ax = plot_signals_heatmap(signals=np.stack([a[:min_size] for a in average]))
        ax.set(title='Averages')
        plt.show()

