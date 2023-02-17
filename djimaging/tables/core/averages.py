import warnings
from abc import abstractmethod

import datajoint as dj
import numpy as np
from matplotlib import pyplot as plt

from djimaging.tables.core.snippets import get_aligned_snippets_times
from djimaging.utils.dj_utils import get_primary_key
from djimaging.utils import plot_utils


class AveragesTemplate(dj.Computed):
    database = ""

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

    @property
    @abstractmethod
    def snippets_table(self):
        pass

    def make(self, key):
        snippets, snippets_times = (self.snippets_table() & key).fetch1('snippets', 'snippets_times')
        triggertimes_snippets = (self.snippets_table() & key).fetch1('triggertimes_snippets').copy()

        average_times = get_aligned_snippets_times(snippets_times=snippets_times)
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
        key = get_primary_key(table=self, key=key)

        snippets, snippets_times, triggertimes_snippets = (self.snippets_table & key).fetch1(
            "snippets", "snippets_times", "triggertimes_snippets")

        average, average_norm, average_times, triggertimes_rel = \
            (self & key).fetch1('average', 'average_norm', 'average_times', 'triggertimes_rel')

        fig, axs = plt.subplots(2, 1, figsize=(10, 4), sharex='all')

        aligned_times = get_aligned_snippets_times(snippets_times=snippets_times)
        plot_utils.plot_traces(
            ax=axs[0], time=aligned_times, traces=snippets.T, title=str(key))

        plot_utils.plot_trace_and_trigger(
            ax=axs[1], time=average_times, trace=average,
            triggertimes=triggertimes_rel, trace_norm=average_norm)

        plt.show()

    def plot(self, restriction=None):
        if restriction is None:
            restriction = dict()

        average = (self & restriction).fetch('average')

        sizes = [a.size for a in average]

        if np.unique(sizes).size > 1:
            warnings.warn('Traces do not have the same size. Are you plotting multiple stimuli?')
        min_size = np.min(sizes)

        ax = plot_utils.plot_signals_heatmap(signals=np.stack([a[:min_size] for a in average]))
        ax.set(title='Averages')
        plt.show()
