import warnings
from abc import abstractmethod

import datajoint as dj
import numpy as np
from matplotlib import pyplot as plt

from djimaging.tables.core.snippets import get_aligned_snippets_times
from djimaging.utils.dj_utils import get_primary_key
from djimaging.utils import plot_utils, math_utils, trace_utils


class AveragesTemplate(dj.Computed):
    database = ""
    _norm_kind = 'amp_one'

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

        if self._norm_kind == 'zscore':
            average_norm = math_utils.normalize_zscore(average)
        elif self._norm_kind == 'zero_one':
            average_norm = math_utils.normalize_zero_one(average)
        elif self._norm_kind == 'amp_one':
            average_norm = math_utils.normalize_amp_one(average)
        elif self._norm_kind == 'amp_std':
            average_norm = math_utils.normalize_amp_std(average)
        else:
            raise NotImplementedError(self._norm_kind)

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

        averages = (self & restriction).fetch('average')
        averages_norm = (self & restriction).fetch('average_norm')

        averages = math_utils.padded_vstack(averages, cval=np.nan)
        averages_norm = math_utils.padded_vstack(averages_norm, cval=np.nan)

        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        plot_utils.set_long_title(fig=fig, title=restriction)

        sort_idxs = trace_utils.argsort_traces(averages, ignore_nan=True)

        plot_utils.plot_signals_heatmap(ax=axs[0], signals=averages[sort_idxs, :])
        plot_utils.plot_signals_heatmap(ax=axs[1], signals=averages_norm[sort_idxs, :])
        plt.show()
