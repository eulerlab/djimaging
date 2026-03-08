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
        average_t0          :float     # time of the first sample of the average
        average_dt          :float     # time between samples of the average
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

    def normalize_average(self, average: np.ndarray) -> np.ndarray:
        """Normalize the average trace according to the configured normalization kind.

        Parameters
        ----------
        average : np.ndarray
            The average trace to normalize.

        Returns
        -------
        np.ndarray
            The normalized average trace.

        Raises
        ------
        NotImplementedError
            If `_norm_kind` is not one of 'zscore', 'zero_one', 'amp_one', or 'amp_std'.
        """
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

    def make(self, key: dict) -> None:
        """Compute and store the average of snippets for a given key.

        Fetches snippets from the snippets table, aligns them in time,
        computes the mean across repetitions, normalizes the average, and
        inserts the result into this table.

        Parameters
        ----------
        key : dict
            The primary key identifying the entry to populate.
        """
        snippets_t0, snippets_dt, snippets = (self.snippets_table() & key).fetch1(
            'snippets_t0', 'snippets_dt', 'snippets')
        triggertimes_snippets = (self.snippets_table() & key).fetch1('triggertimes_snippets')

        if snippets.shape[1] <= 1:
            warnings.warn(f"Skipping {key} because it has only one repetition.")
            return

        snippets_times = (np.tile(np.arange(snippets.shape[0]) * snippets_dt, (len(snippets_t0), 1)).T
                          + snippets_t0)
        average_times = get_aligned_snippets_times(snippets_times=snippets_times)

        average = np.mean(snippets, axis=1)

        average_norm = self.normalize_average(average)

        triggertimes_rel = np.mean(triggertimes_snippets - triggertimes_snippets[0, :], axis=1)

        self.insert1(dict(
            **key,
            average=average.astype(np.float32),
            average_norm=average_norm.astype(np.float32),
            average_t0=average_times[0],
            average_dt=snippets_dt,
            triggertimes_rel=triggertimes_rel.astype(np.float32),
        ))

    def plot1(self, key: dict | None = None, xlim: tuple | None = None) -> None:
        """Plot snippets and average trace for a single entry.

        Parameters
        ----------
        key : dict | None, optional
            Primary key identifying the entry to plot. If None, the first
            available key is used.
        xlim : tuple | None, optional
            x-axis limits for the plot. Default is None (auto).
        """
        key = get_primary_key(table=self, key=key)

        snippets_t0, snippets_dt, snippets = (self.snippets_table & key).fetch1(
            'snippets_t0', 'snippets_dt', 'snippets')

        average, average_norm, average_t0, average_dt, triggertimes_rel = \
            (self & key).fetch1('average', 'average_norm', 'average_t0', 'average_dt', 'triggertimes_rel')

        snippets_times = (np.tile(np.arange(snippets.shape[0]) * snippets_dt, (len(snippets_t0), 1)).T
                          + snippets_t0)

        average_times = get_aligned_snippets_times(snippets_times=snippets_times)

        fig, axs = plt.subplots(2, 1, figsize=(10, 4), sharex='all')

        aligned_times = get_aligned_snippets_times(snippets_times=snippets_times)
        plot_utils.plot_traces(
            ax=axs[0], time=aligned_times, traces=snippets.T)
        axs[0].set(ylabel='trace', xlabel='aligned time')

        plot_utils.plot_trace_and_trigger(
            ax=axs[1], time=average_times, trace=average,
            triggertimes=triggertimes_rel, trace_norm=average_norm)

        plt.show()

    def plot(self, restriction: dict | None = None, sort: bool = True) -> None:
        """Plot heatmaps of all averages matching the given restriction.

        Parameters
        ----------
        restriction : dict | None, optional
            Restriction to apply to the table before fetching. Default is None
            (all entries).
        sort : bool, optional
            Whether to sort traces before plotting. Default is True.
        """
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
    _f_resample = 60

    @property
    @abstractmethod
    def snippets_table(self):
        pass

    def make(self, key: dict) -> None:
        """Compute and store the resampled average of snippets for a given key.

        Fetches snippets, resamples them to `_f_resample` Hz, computes the
        mean across repetitions after temporal alignment, normalizes the
        average, and inserts the result into this table.

        Parameters
        ----------
        key : dict
            The primary key identifying the entry to populate.
        """
        snippets_t0, snippets_dt, snippets = (self.snippets_table() & key).fetch1(
            'snippets_t0', 'snippets_dt', 'snippets')

        if snippets.shape[1] <= 1:
            warnings.warn(f"Skipping {key} because it has only one repetition.")
            return

        triggertimes_snippets = (self.snippets_table() & key).fetch1('triggertimes_snippets')

        snippets_times = (np.tile(np.arange(snippets.shape[0]) * snippets_dt, (len(snippets_t0), 1)).T
                          + snippets_t0)

        average, average_times, _ = compute_upsampled_average(
            snippets, snippets_times, triggertimes_snippets, f_resample=self._f_resample)
        average_norm = self.normalize_average(average)
        triggertimes_rel = np.mean(triggertimes_snippets - triggertimes_snippets[0, :], axis=1)

        self.insert1(dict(
            **key,
            average=average,
            average_norm=average_norm,
            average_t0=average_times[0],
            average_dt=1 / self._f_resample,
            triggertimes_rel=triggertimes_rel,
        ))

    def plot1(self, key: dict | None = None, xlim: tuple | None = None) -> None:
        """Plot resampled snippets and average trace for a single entry.

        Parameters
        ----------
        key : dict | None, optional
            Primary key identifying the entry to plot. If None, the first
            available key is used.
        xlim : tuple | None, optional
            x-axis limits for the plot. Default is None (auto).
        """
        key = get_primary_key(table=self, key=key)

        snippets_t0, snippets_dt, snippets, triggertimes_snippets = (self.snippets_table & key).fetch1(
            'snippets_t0', 'snippets_dt', 'snippets', 'triggertimes_snippets')

        average, average_norm, average_t0, average_dt, triggertimes_rel = \
            (self & key).fetch1('average', 'average_norm', 'average_t0', 'average_dt', 'triggertimes_rel')

        snippets_times = (np.tile(np.arange(snippets.shape[0]) * snippets_dt, (len(snippets_t0), 1)).T
                          + snippets_t0)
        average_times = np.arange(len(average)) * average_dt + average_t0

        fig, axs = plt.subplots(2, 1, figsize=(10, 4), sharex='all')

        axs[0].plot(snippets_times - triggertimes_snippets[0], snippets, alpha=0.5)
        axs[0].set(ylabel='trace', xlabel='rel. to trigger', xlim=xlim)

        plot_utils.plot_trace_and_trigger(
            ax=axs[1], time=average_times, trace=average,
            triggertimes=triggertimes_rel, trace_norm=average_norm)

        plt.show()


def compute_upsampled_average(
        snippets: np.ndarray,
        snippets_times: np.ndarray,
        triggertimes_snippets: np.ndarray,
        f_resample: float = 500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a temporally resampled average across snippet repetitions.

    Each repetition is interpolated onto a common time grid defined by
    `f_resample` and the median stimulus duration, then averaged.

    Parameters
    ----------
    snippets : np.ndarray
        2-D array of snippet traces with shape (n_time, n_reps).
    snippets_times : np.ndarray
        2-D array of absolute time stamps with shape (n_time, n_reps).
    triggertimes_snippets : np.ndarray
        2-D array of trigger times with shape (n_triggers, n_reps).
    f_resample : float, optional
        Target sampling frequency in Hz. Default is 500.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        average : np.ndarray
            Mean trace across repetitions on the resampled grid, shape (n_resampled,).
        average_times : np.ndarray
            Time axis of the resampled grid, shape (n_resampled,).
        snippets_resampled : np.ndarray
            Resampled individual snippets, shape (n_resampled, n_reps).
    """
    dt = 1 / f_resample
    stim_dur = np.median(np.diff(triggertimes_snippets[0]))
    resampled_n = int(np.ceil(stim_dur * f_resample))
    n_reps = snippets.shape[1]

    average_times = np.arange(0, resampled_n) * dt

    snippets_resampled = np.zeros((resampled_n, n_reps))
    for rep_idx in range(n_reps):
        snippets_resampled[:, rep_idx] = np.interp(
            x=average_times,
            xp=snippets_times[:, rep_idx] - triggertimes_snippets[0, rep_idx],
            fp=snippets[:, rep_idx])

    average = np.mean(snippets_resampled, axis=1)

    return average, average_times, snippets_resampled
