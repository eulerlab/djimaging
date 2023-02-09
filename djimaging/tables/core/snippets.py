import warnings
from abc import abstractmethod

import datajoint as dj
import numpy as np
from matplotlib import pyplot as plt

from djimaging.utils import plot_utils
from djimaging.utils.dj_utils import get_primary_key
from djimaging.utils.scanm_utils import split_trace_by_reps


def get_aligned_snippets_times(snippets_times, raise_error=True):
    snippets_times = snippets_times - snippets_times[0, :]

    is_inconsistent = np.any(np.std(snippets_times, axis=1) > 1e-4)
    if is_inconsistent:
        if raise_error:
            raise ValueError(f'Failed to snippet times: max_std={np.max(np.std(snippets_times, axis=1))}')
        else:
            warnings.warn(f'Snippet times are inconsistent: max_std={np.max(np.std(snippets_times, axis=1))}')

    aligned_times = np.mean(snippets_times, axis=1)
    return aligned_times


class SnippetsTemplate(dj.Computed):
    database = ""

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

    @property
    @abstractmethod
    def preprocesstraces_table(self):
        pass

    @property
    @abstractmethod
    def stimulus_table(self):
        pass

    @property
    @abstractmethod
    def presentation_table(self):
        pass

    @property
    @abstractmethod
    def traces_table(self):
        pass

    @property
    def key_source(self):
        try:
            return (self.preprocesstraces_table() & (self.stimulus_table() & "isrepeated=1")).proj()
        except TypeError:
            pass

    def make(self, key):
        ntrigger_rep = (self.stimulus_table() & key).fetch1('ntrigger_rep')
        triggertimes = (self.presentation_table() & key).fetch1('triggertimes')
        trace_times, traces = (self.preprocesstraces_table() & key).fetch1('preprocess_trace_times', 'preprocess_trace')

        snippets, snippets_times, triggertimes_snippets, droppedlastrep_flag = split_trace_by_reps(
            traces, trace_times, triggertimes, ntrigger_rep, allow_drop_last=True)

        self.insert1(dict(
            **key,
            snippets=snippets,
            snippets_times=snippets_times,
            triggertimes_snippets=triggertimes_snippets,
            droppedlastrep_flag=droppedlastrep_flag,
        ))

    def plot1(self, key=None, xlim=None):
        key = get_primary_key(table=self, key=key)
        snippets, snippets_times, triggertimes_snippets = (self & key).fetch1(
            "snippets", "snippets_times", "triggertimes_snippets")

        fig, axs = plt.subplots(2, 1, figsize=(10, 4))

        plot_utils.plot_trace_and_trigger(
            ax=axs[0], time=snippets_times, trace=snippets, triggertimes=triggertimes_snippets, title=str(key))
        axs[0].set(xlim=xlim)

        aligned_times = get_aligned_snippets_times(snippets_times=snippets_times)
        plot_utils.plot_traces(
            ax=axs[1], time=aligned_times, traces=snippets.T)
