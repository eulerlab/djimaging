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

    @property
    def key_source(self):
        try:
            return self.snippets_table.proj()
        except (AttributeError, TypeError):
            pass

    def normalize_average(self, average):
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

        return average_norm

    def make(self, key):
        snippets, snippets_times = (self.snippets_table() & key).fetch1('snippets', 'snippets_times')
        triggertimes_snippets = (self.snippets_table() & key).fetch1('triggertimes_snippets')

        average_times = get_aligned_snippets_times(snippets_times=snippets_times)
        average = np.mean(snippets, axis=1)

        average_norm = self.normalize_average(average)

        triggertimes_rel = np.mean(triggertimes_snippets - triggertimes_snippets[0, :], axis=1)

        self.insert1(dict(
            **key,
            average=average,
            average_norm=average_norm,
            average_times=average_times,
            triggertimes_rel=triggertimes_rel,
        ))

    def plot1(self, key=None, xlim=None):
        key = get_primary_key(table=self, key=key)

        snippets, snippets_times, triggertimes_snippets = (self.snippets_table & key).fetch1(
            "snippets", "snippets_times", "triggertimes_snippets")

        average, average_norm, average_times, triggertimes_rel = \
            (self & key).fetch1('average', 'average_norm', 'average_times', 'triggertimes_rel')

        fig, axs = plt.subplots(2, 1, figsize=(10, 4), sharex='all')

        aligned_times = get_aligned_snippets_times(snippets_times=snippets_times)
        plot_utils.plot_traces(
            ax=axs[0], time=aligned_times, traces=snippets.T)
        axs[0].set(ylabel='trace', xlabel='aligned time')

        plot_utils.plot_trace_and_trigger(
            ax=axs[1], time=average_times, trace=average,
            triggertimes=triggertimes_rel, trace_norm=average_norm)

        plt.show()

    def plot(self, restriction=None, sort=True):
        if restriction is None:
            restriction = dict()

        averages = (self & restriction).fetch('average')
        averages_norm = (self & restriction).fetch('average_norm')

        averages = math_utils.padded_vstack(averages, cval=np.nan)
        averages_norm = math_utils.padded_vstack(averages_norm, cval=np.nan)

        n = averages.shape[0]

        fig, axs = plt.subplots(1, 2, figsize=(14, 1 + np.minimum(n * 0.1, 10)))
        if len(restriction) > 0:
            plot_utils.set_long_title(fig=fig, title=restriction)

        sort_idxs = trace_utils.argsort_traces(averages_norm, ignore_nan=True) if sort else np.arange(n)

        axs[0].set_title('average')
        plot_utils.plot_signals_heatmap(ax=axs[0], signals=averages[sort_idxs, :])
        axs[1].set_title('average_norm')
        plot_utils.plot_signals_heatmap(ax=axs[1], signals=averages_norm[sort_idxs, :])
        plt.show()


class ResampledAveragesTemplate(AveragesTemplate):
    """Averages of resampled snippets

    Example usage:

    @schema
    class ResampledAverages(core.ResampledAveragesTemplate):
        _norm_kind = 'amp_one'
        _f_resample = 500
        snippets_table = Snippets
    """

    database = ""
    _norm_kind = 'amp_one'
    _f_resample = 500

    @property
    @abstractmethod
    def snippets_table(self):
        pass

    def make(self, key):
        snippets, snippets_times = (self.snippets_table() & key).fetch1('snippets', 'snippets_times')
        triggertimes_snippets = (self.snippets_table() & key).fetch1('triggertimes_snippets')

        dt = 1 / self._f_resample
        stim_dur = np.median(np.diff(triggertimes_snippets[0]))
        resampled_n = int(np.ceil(stim_dur * self._f_resample))
        n_reps = snippets.shape[1]

        average_times = np.arange(0, resampled_n) * dt

        snippets_resampled = np.zeros((resampled_n, n_reps))
        for rep_idx in range(n_reps):
            snippets_resampled[:, rep_idx] = np.interp(
                x=average_times,
                xp=snippets_times[:, rep_idx] - triggertimes_snippets[0, rep_idx],
                fp=snippets[:, rep_idx])

        average = np.mean(snippets_resampled, axis=1)
        average_norm = self.normalize_average(average)
        triggertimes_rel = np.mean(triggertimes_snippets - triggertimes_snippets[0, :], axis=1)

        self.insert1(dict(
            **key,
            average=average,
            average_norm=average_norm,
            average_times=average_times,
            triggertimes_rel=triggertimes_rel,
        ))

    def plot1(self, key=None, xlim=None):
        key = get_primary_key(table=self, key=key)

        snippets, snippets_times, triggertimes_snippets = (self.snippets_table & key).fetch1(
            "snippets", "snippets_times", "triggertimes_snippets")

        average, average_norm, average_times, triggertimes_rel = \
            (self & key).fetch1('average', 'average_norm', 'average_times', 'triggertimes_rel')

        fig, axs = plt.subplots(2, 1, figsize=(10, 4), sharex='all')

        axs[0].plot(snippets_times - triggertimes_snippets[0], snippets, alpha=0.5)
        axs[0].set(ylabel='trace', xlabel='rel. to trigger', xlim=xlim)

        plot_utils.plot_trace_and_trigger(
            ax=axs[1], time=average_times, trace=average,
            triggertimes=triggertimes_rel, trace_norm=average_norm)

        plt.show()
