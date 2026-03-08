"""
Example usage:
@schema
class ChirpSurround(response.ChirpSurroundTemplate):
    snippets_table = Snippets
"""

from abc import abstractmethod

import datajoint as dj
import numpy as np

from matplotlib import pyplot as plt

from djimaging.utils.dj_utils import get_primary_key, get_secondary_keys
from djimaging.tables.core.averages import compute_upsampled_average


class ChirpSurroundTemplate(dj.Computed):
    database = ""

    _normalize_amp = False
    _fs_resample = 500
    _t_on_step = 2
    _t_off_step = 5
    _t_max = 6

    @property
    def definition(self):
        definition = f"""
        # Correlation between two response for two stimuli like local and global chirp
        -> self.snippets_table.proj(lChirp='stim_name')
        -> self.snippets_table.proj(gChirp='stim_name')
        ---
        l_polarity_index : float
        g_polarity_index : float
        surround_strength = NULL : float
        """
        return definition

    @property
    @abstractmethod
    def snippets_table(self):
        pass

    def _l_table(self):
        """Return the local chirp (lChirp) snippet table projection.

        Returns
        -------
        datajoint.expression.QueryExpression
            Projected snippets table restricted to lChirp stimulus.
        """
        secondary_keys = get_secondary_keys(self.snippets_table)
        return (self.snippets_table & dict(stim_name="lChirp")).proj(*secondary_keys, lChirp='stim_name')

    def _g_table(self):
        """Return the global chirp (gChirp) snippet table projection.

        Returns
        -------
        datajoint.expression.QueryExpression
            Projected snippets table restricted to gChirp stimulus.
        """
        secondary_keys = get_secondary_keys(self.snippets_table)
        return (self.snippets_table & dict(stim_name="gChirp")).proj(*secondary_keys, gChirp='stim_name')

    @property
    def key_source(self):
        try:
            return self._l_table().proj() * self._g_table().proj()
        except (AttributeError, TypeError):
            pass

    def compute_entry(self, key: dict, plot: bool = False) -> tuple:
        """Compute surround strength and polarity indices for lChirp and gChirp.

        Parameters
        ----------
        key : dict
            DataJoint primary key identifying the entry to compute.
        plot : bool, optional
            If True, create diagnostic plots. Default is False.

        Returns
        -------
        tuple
            Tuple of (surround_strength, l_polarity_index, g_polarity_index).
        """
        l_snippets_t0, l_snippets_dt, l_snippets, l_triggertimes_snippets = (self._l_table() & key).fetch1(
            "snippets_t0", "snippets_dt", 'snippets', 'triggertimes_snippets')
        g_snippets_t0, g_snippets_dt, g_snippets, g_triggertimes_snippets = (self._g_table() & key).fetch1(
            "snippets_t0", "snippets_dt", 'snippets', 'triggertimes_snippets')

        l_snippets_times = (np.tile(np.arange(l_snippets.shape[0]) * l_snippets_dt, (len(l_snippets_t0), 1)).T
                            + l_snippets_t0)
        g_snippets_times = (np.tile(np.arange(g_snippets.shape[0]) * g_snippets_dt, (len(g_snippets_t0), 1)).T
                            + g_snippets_t0)

        l_average, l_average_times, _ = compute_upsampled_average(
            l_snippets, l_snippets_times, l_triggertimes_snippets, f_resample=self._fs_resample)
        g_average, g_average_times, _ = compute_upsampled_average(
            g_snippets, g_snippets_times, g_triggertimes_snippets, f_resample=self._fs_resample)

        if plot:
            fig, axs = plt.subplots(1, 4, figsize=(12, 2))
            ax = axs[0]
            ax.set_title(f"Average trace; fs={self._fs_resample} Hz")
            ax.plot(l_average)
            ax.plot(g_average)
            ax.xaxis.set_major_formatter(lambda x, pos: f"{x:.1g}\n{x / self._fs_resample}s")
        else:
            axs = [None] * 4

        # Normalize as in Franke et al. 2017
        if self._normalize_amp:
            l_average /= np.max(np.abs(l_average))
        l_average = l_average - np.median(l_average[:int(self._fs_resample * self._t_on_step)])

        if self._normalize_amp:
            g_average /= np.max(np.abs(g_average))
        g_average = g_average - np.median(g_average[:int(self._fs_resample * self._t_on_step)])

        l_avg_on, l_avg_off, l_polarity_index = compute_step_response_index(
            average=l_average, fs=self._fs_resample, t_on_step=self._t_on_step, t_off_step=self._t_off_step,
            plot=axs[1], color='C0')

        g_avg_on, g_avg_off, g_polarity_index = compute_step_response_index(
            average=g_average, fs=self._fs_resample, t_on_step=self._t_on_step, t_off_step=self._t_off_step,
            plot=axs[2], color='C1')

        surround_strength = compute_surround_strength(
            lchirp_on=l_avg_on, gchirp_on=g_avg_on, lchirp_off=l_avg_off, gchirp_off=g_avg_off,
            lchirp_polarity_index=l_polarity_index, gchirp_polarity_index=g_polarity_index, plot=axs[3])

        if plot:
            plt.tight_layout()
            plt.show()

        return surround_strength, l_polarity_index, g_polarity_index

    def make(self, key: dict, plot: bool = False) -> None:
        """Compute and insert surround strength and polarity indices into the table.

        Fetches lChirp and gChirp snippet data, computes the local and global polarity
        indices and the surround strength, and inserts the result into the table.

        Parameters
        ----------
        key : dict
            DataJoint primary key identifying the entry to populate.
        plot : bool, optional
            If True, create diagnostic plots during computation. Default is False.
        """
        surround_strength, l_polarity_index, g_polarity_index = self.compute_entry(key, plot=plot)

        self.insert1(dict(
            key,
            surround_strength=surround_strength,
            l_polarity_index=l_polarity_index,
            g_polarity_index=g_polarity_index,
        ))

    def plot1(self, key: dict | None = None) -> None:
        """Plot diagnostic figures and verify against stored values.

        Parameters
        ----------
        key : dict or None, optional
            DataJoint primary key. If None, uses the first available key.

        Raises
        ------
        ValueError
            If computed values do not match the stored values in the table.
        """
        key = get_primary_key(table=self, key=key)
        surround_strength, l_polarity_index, g_polarity_index = \
            (self & key).fetch1('surround_strength', 'l_polarity_index', 'g_polarity_index')

        plot_values = np.array(self.compute_entry(key, plot=True))
        db_values = np.array([surround_strength, l_polarity_index, g_polarity_index])

        if not np.all(plot_values == db_values):
            raise ValueError("Computed values do not match stored values")


def compute_step_response_index(
        average: np.ndarray,
        fs: float,
        alpha: float = 2,
        t_on_step: float = 2,
        t_off_step: float = 5,
        color: str | None = None,
        plot: plt.Axes | bool | None = None,
) -> tuple:
    """Calculate step response and the polarity index (POi) from Franke et al. 2017.

    Note: In the paper it's not clear how they treated negative values, we clip them to zero.

    # TODO: Merge with compute_polarity_index

    Parameters
    ----------
    average : np.ndarray
        A 1D array of the average response trace.
    fs : float
        Sampling rate of the data in Hz.
    alpha : float, optional
        Duration of the response window in seconds. Default is 2.
    t_on_step : float, optional
        Time of the on-step of the stimulus in seconds. Default is 2.
    t_off_step : float, optional
        Time of the off-step of the stimulus in seconds. Default is 5.
    color : str or None, optional
        Matplotlib color for the trace plot. Default is None.
    plot : matplotlib.axes.Axes or bool or None, optional
        If a matplotlib Axes instance, plot into it. If True, create a new figure.
        If None or False, no plot is produced. Default is None.

    Returns
    -------
    tuple
        Tuple of (avg_on, avg_off, polarity_index).
    """

    idx_on_start = int(np.floor(t_on_step * fs))
    idx_on_end = int(np.floor((t_on_step + alpha) * fs))

    idx_off_start = int(np.floor(t_off_step * fs))
    idx_off_end = int(np.floor((t_off_step + alpha) * fs))

    avg_on = np.clip(np.mean(average[idx_on_start:idx_on_end]), 0, None)
    avg_off = np.clip(np.mean(average[idx_off_start:idx_off_end]), 0, None)

    scale = avg_on + avg_off
    if scale < 1e-9:
        polarity_index = 0.
    else:
        polarity_index = (avg_on - avg_off) / scale

    if plot:
        if isinstance(plot, plt.Axes):
            ax = plot
        else:
            fig, ax = plt.subplots(1, 1, figsize=(4, 2))
        ax.set_title(f"POi: {polarity_index:.2f}")
        ax.plot(average, color=color)
        ax.plot([idx_on_start, idx_on_end], [avg_on, avg_on], color='r')
        ax.plot([idx_off_start, idx_off_end], [avg_off, avg_off], color='r')
        ax.set_xlim(0, fs * (t_off_step + 2 * alpha))
        ax.xaxis.set_major_formatter(lambda x, pos: f"{x:.1g}\n{x / fs}s")

    return avg_on, avg_off, polarity_index


def compute_surround_strength(
        lchirp_on: float,
        gchirp_on: float,
        lchirp_off: float,
        gchirp_off: float,
        lchirp_polarity_index: float,
        gchirp_polarity_index: float,
        plot: plt.Axes | bool | None = None,
) -> float:
    """Calculate the surround strength (SSt) from Hsiang et al. 2024.

    Note: In the paper they only calculated the surround strength for On bipolar cells.

    Parameters
    ----------
    lchirp_on : float
        Light increment response of the local chirp stimulus.
    gchirp_on : float
        Light increment response of the global chirp stimulus.
    lchirp_off : float
        Light decrement response of the local chirp stimulus.
    gchirp_off : float
        Light decrement response of the global chirp stimulus.
    lchirp_polarity_index : float
        Polarity index of the local chirp stimulus.
    gchirp_polarity_index : float
        Polarity index of the global chirp stimulus.
    plot : matplotlib.axes.Axes or bool or None, optional
        If a matplotlib Axes instance, plot into it. If True, create a new figure.
        If None or False, no plot is produced. Default is None.

    Returns
    -------
    float
        Surround strength (SSt), or np.nan if polarity indices have opposite signs
        or the local response is too small.
    """

    if (lchirp_polarity_index > 0) & (gchirp_polarity_index > 0):
        if lchirp_on < 1e-5:
            surround_strength = np.nan
        else:
            surround_strength = (lchirp_on - gchirp_on) / lchirp_on
    elif (lchirp_polarity_index < 0) & (gchirp_polarity_index < 0):
        if lchirp_off < 1e-5:
            surround_strength = np.nan
        else:
            surround_strength = (lchirp_off - gchirp_off) / np.maximum(lchirp_off, 1e-9)
    else:
        surround_strength = np.nan

    if plot:
        if isinstance(plot, plt.Axes):
            ax = plot
        else:
            fig, ax = plt.subplots(1, 1, figsize=(4, 2))
        ax.set_title(f"SSt: {surround_strength:.2f}")
        ax.plot(0, lchirp_on, marker='^', color='C0', label='lChirp On')
        ax.plot(1, lchirp_off, marker='v', color='C0', label='lChirp Off')
        ax.plot(2, gchirp_on, marker='^', color='C1', label='gChirp On')
        ax.plot(3, gchirp_off, marker='v', color='C1', label='gChirp Off')
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xlim(-0.5, 10)
        ax.legend()

    return surround_strength
