"""
Tables for Chirp response quality index (QI) and features.

Example usage:

from djimaging.tables import response

@schema
class ChirpQI(response.ChirpQITemplate):
    _stim_family = "chirp"
    _stim_name = "chirp"

    stimulus_table = Stimulus
    snippets_table = Snippets


@schema
class ChirpFeatures(response.ChirpFeaturesTemplate):
    stimulus_table = Stimulus
    snippets_table = Snippets
    presentation_table = Presentation
"""

from abc import abstractmethod

import datajoint as dj
import numpy as np
from matplotlib import pyplot as plt

from djimaging.tables.response.response_quality import RepeatQITemplate
from djimaging.utils.dj_utils import get_primary_key
from djimaging.tables.core.averages import compute_upsampled_average


class ChirpQITemplate(RepeatQITemplate):
    _stim_family = "chirp"
    _stim_name = "chirp"

    @property
    @abstractmethod
    def stimulus_table(self):
        pass

    @property
    @abstractmethod
    def snippets_table(self):
        pass


class ChirpFeaturesTemplate(dj.Computed):
    database = ""

    _fs_resample = 500
    _t_on_step = 2
    _t_off_step = 5
    _t_max = 6
    _t_flicker_start = 10
    _t_flicker_end = 29

    _hfi_constant = 1e4  # See Baden et al 2013

    @property
    def definition(self):
        definition = '''
        # Computes Chirp features for local and global chirps.
        # See Franke et al. 2017 for details, for HFi see also Baden et al. 2013
        -> self.snippets_table
        ---
        polarity_index: float # Polarity index (POi) from Franke et al. 2017
        high_frequency_index: float # High Frequency Index (HFi) from Baden et al. 2013 and Franke et al. 2017
        transience_index: float # Response transience index (RTi) from Franke et al. 2017
        plateau_index: float # Response plateau index (RPi) from Franke et al. 2017
        tonic_release_index: float # Tonic release index (TRi) from Franke et al. 2017
        '''
        return definition

    @property
    @abstractmethod
    def stimulus_table(self):
        pass

    @property
    @abstractmethod
    def snippets_table(self):
        pass

    @property
    @abstractmethod
    def presentation_table(self):
        pass

    @property
    def key_source(self):
        try:
            return self.snippets_table().proj() & \
                (self.stimulus_table() & "stim_name = 'chirp' or stim_family = 'chirp'")
        except (AttributeError, TypeError):
            pass

    def compute_entry(self, key, plot=False):
        snippets_t0, snippets_dt, snippets, triggertimes_snippets = (self.snippets_table() & key).fetch1(
            "snippets_t0", "snippets_dt", 'snippets', 'triggertimes_snippets')

        snippets_times = (np.tile(np.arange(snippets.shape[0]) * snippets_dt, (len(snippets_t0), 1)).T
                          + snippets_t0)

        average, average_times, _ = compute_upsampled_average(
            snippets, snippets_times, triggertimes_snippets, f_resample=self._fs_resample)

        if plot:
            fig, axs = plt.subplots(1, 6, figsize=(15, 2))
            ax = axs[0]
            ax.set_title(f"Average trace; fs={self._fs_resample} Hz")
            ax.plot(average)
            ax.xaxis.set_major_formatter(lambda x, pos: f"{x:.1g}\n{x / self._fs_resample}s")
        else:
            axs = [None] * 6

        mean_dt = np.mean(np.diff(snippets_times, axis=0))
        high_frequency_index = compute_high_frequency_index(
            snippets=snippets, fs=1 / mean_dt, constant=self._hfi_constant, plot=axs[1])

        # Normalize as in Franke et al. 2017
        average /= np.max(np.abs(average))
        average = average - np.median(average[:int(self._fs_resample * self._t_on_step)])

        polarity_index = compute_polarity_index(
            average=average, fs=self._fs_resample, t_on_step=self._t_on_step, t_off_step=self._t_off_step, plot=axs[2])

        transience_index = compute_transience_index(
            average=average, fs=self._fs_resample, t_on_step=self._t_on_step, t_max=self._t_max, plot=axs[3])

        plateau_index = compute_plateau_index(
            average=average, fs=self._fs_resample, t_on_step=self._t_on_step, t_max=self._t_max, plot=axs[4])

        tonic_release_index = compute_tonic_release_index(
            average=average, fs=self._fs_resample,
            t_flicker_start=self._t_flicker_start, t_flicker_end=self._t_flicker_end, plot=axs[5])

        if plot:
            plt.tight_layout()
            plt.show()

        return polarity_index, high_frequency_index, transience_index, plateau_index, tonic_release_index

    def make(self, key, plot=False):
        polarity_index, high_frequency_index, transience_index, plateau_index, tonic_release_index = \
            self.compute_entry(key, plot=plot)

        self.insert1(dict(
            key,
            polarity_index=polarity_index,
            high_frequency_index=high_frequency_index,
            transience_index=transience_index,
            plateau_index=plateau_index,
            tonic_release_index=tonic_release_index,
        ))

    def plot1(self, key=None):
        key = get_primary_key(table=self, key=key)
        polarity_index, high_frequency_index, transience_index, plateau_index, tonic_release_index = \
            (self & key).fetch1(
                'polarity_index', 'high_frequency_index', 'transience_index', 'plateau_index', 'tonic_release_index')

        plot_values = np.array(self.compute_entry(key, plot=True))
        db_values = np.array(
            [polarity_index, high_frequency_index, transience_index, plateau_index, tonic_release_index])

        if not np.all(plot_values == db_values):
            raise ValueError("Computed values do not match stored values")


def compute_high_frequency_index(snippets, fs, constant=1e4, plot=None):
    """
    Calculate the High Frequency Index (HFi) from Baden et al. 2013 and Franke et al. 2017.
    This metric is invariant to linear transformations of the signal.

    Note: In the paper the ratio is flipped in the equation which is probably a typo.


    :param snippets: A 2d (time X trial) numpy array
    :param fs: Sampling rate of the data in Hz.
    :param constant: A constant to scale the HFi before taking the log.
    :param plot: If True, plot the average trace and the frequency bands.

    :return: High Frequency Index (HFi) for each ROI.
    """

    # Determine the number of points corresponding to the first 6 seconds
    n_points = int(6 * fs)
    fft_results = np.fft.rfft(snippets[:n_points, :], axis=0)
    fft_power = np.abs(fft_results) ** 2
    fft_mean_power = np.mean(fft_power, axis=1)

    # Define frequency bands
    freqs = np.fft.rfftfreq(n_points, d=1 / fs)
    band1 = (freqs >= 0.5) & (freqs <= 1)  # 0.5–1 Hz
    band2 = (freqs >= 2) & (freqs <= 16)  # 2–16 Hz

    # Calculate mean power in each band
    f1 = np.mean(fft_mean_power[band1])
    f2 = np.mean(fft_mean_power[band2])

    # Calculate HFi. Note: corrected from np.log10(f1 / f2)
    high_frequency_index = np.log10(constant * f2 / f1)

    if plot:
        if isinstance(plot, plt.Axes):
            ax = plot
        else:
            fig, ax = plt.subplots(1, 1, figsize=(4, 2))
        ax.set_title(f"HFi: {high_frequency_index:.2f}")
        ax.loglog(freqs, fft_mean_power)
        ax.fill_between(x=freqs[band1], y1=np.min(fft_mean_power), y2=np.max(fft_mean_power), alpha=0.5)
        ax.plot([freqs[band1][0], freqs[band1][-1]], [f1, f1], 'r')
        ax.plot([freqs[band2][0], freqs[band2][-1]], [f2, f2], 'r')
        ax.fill_between(x=freqs[band2], y1=np.min(fft_mean_power), y2=np.max(fft_mean_power), alpha=0.5)

    return high_frequency_index


def compute_polarity_index(average, fs, alpha=2, t_on_step=2, t_off_step=5, plot=None):
    """
    Calculate the polarity index (POi) from Franke et al. 2017.

    Note: In the paper it's not clear how they treated negative values, we clip them to zero.

    :param average: A 1d numpy array of trace
    :param fs: Sampling rate of the data in Hz.
    :param alpha: Duration of the response window in seconds.
    :param t_on_step: Time of the on-step of the stimulus in seconds.
    :param t_off_step: Time of the off-step of the stimulus in seconds.
    :param plot: If True, plot the average trace and the response window.

    :return: Polarity index (POi) for each ROI.
    """

    idx_on_start = int(np.floor(t_on_step * fs))
    idx_on_end = int(np.floor((t_on_step + alpha) * fs))

    idx_off_start = int(np.floor(t_off_step * fs))
    idx_off_end = int(np.floor((t_off_step + alpha) * fs))

    avg_on = np.clip(np.mean(average[idx_on_start:idx_on_end]), 0, None)
    avg_off = np.clip(np.mean(average[idx_off_start:idx_off_end]), 0, None)

    scale = avg_on + avg_off
    if scale < 1e-9:
        polarity_index = 0.5
    else:
        polarity_index = (avg_on - avg_off) / scale

    if plot:
        if isinstance(plot, plt.Axes):
            ax = plot
        else:
            fig, ax = plt.subplots(1, 1, figsize=(4, 2))
        ax.set_title(f"POi: {polarity_index:.2f}")
        ax.plot(average)
        ax.plot([idx_on_start, idx_on_end], [avg_on, avg_on], color='r')
        ax.plot([idx_off_start, idx_off_end], [avg_off, avg_off], color='r')
        ax.set_xlim(0, fs * (t_off_step + 2 * alpha))
        ax.xaxis.set_major_formatter(lambda x, pos: f"{x:.1g}\n{x / fs}s")

    return polarity_index


def compute_transience_index(average, fs, alpha=0.4, t_on_step=2, t_max=6, plot=None):
    """
    Calculate the response transience index (RTi) from Franke et al. 2017.

    Note: In the paper it's not clear how they treated negative values.

    :param average: A 1d numpy array of trace
    :param fs: Sampling rate of the data in Hz.
    :param alpha: Duration of the response window in seconds.
    :param t_on_step: Time of the on-step of the stimulus in seconds.
    :param t_max: Time of the offset of the response window in seconds.
    :param plot: If True, plot the average trace and the response window.

    :return: Response transience index (RTi).
    """

    idx_start = int(t_on_step * fs)
    idx_end = int(t_max * fs)

    average_shifted = average - np.median(average[:idx_start])

    idx_max = idx_start + np.argmax(average_shifted[idx_start:idx_end])
    idx_max_alpha = idx_max + int(alpha * fs)
    transience_index = 1 - (average_shifted[idx_max_alpha] / average_shifted[idx_max])

    if plot:
        if isinstance(plot, plt.Axes):
            ax = plot
        else:
            fig, ax = plt.subplots(1, 1, figsize=(4, 2))
        ax.set_title(f"RTi: {transience_index:.2f}")
        ax.plot(average_shifted)
        ax.axvline(idx_max, c='r')
        ax.plot(idx_max, average_shifted[idx_max], 'rX')
        ax.axvline(idx_max_alpha, c='r')
        ax.plot(idx_max_alpha, average_shifted[idx_max_alpha], 'rX')
        ax.set_xlim(0, fs * (t_max + 3))
        ax.xaxis.set_major_formatter(lambda x, pos: f"{x:.1g}\n{x / fs}s")

    return transience_index


def compute_plateau_index(average, fs, alpha=2., t_on_step=2, t_max=6, plot=None):
    """
    Calculate the response plateau index (RPi) from Franke et al. 2017.

    Note: In the paper it's not clear how they treated negative values.

    :param average: A 1d numpy array of trace
    :param fs: Sampling rate of the data in Hz.
    :param alpha: Duration of the response window in seconds.
    :param t_on_step: Time of the on-step of the stimulus in seconds.
    :param t_max: Time of the offset of the response window in seconds.
    :param plot: If True, plot the average trace and the response window.

    :return: Response plateau index (RPi).
    """

    idx_start = int(t_on_step * fs)
    idx_end = int(t_max * fs)

    idx_max = idx_start + np.argmax(average[idx_start:idx_end])
    idx_max_alpha = idx_max + int(alpha * fs)
    plateau_index = average[idx_max_alpha] / average[idx_max]

    if plot:
        if isinstance(plot, plt.Axes):
            ax = plot
        else:
            fig, ax = plt.subplots(1, 1, figsize=(4, 2))
        ax.set_title(f"RPi: {plateau_index:.2f}")
        ax.plot(average)
        ax.axvline(idx_max, c='r')
        ax.plot(idx_max, average[idx_max], 'rX')
        ax.axvline(idx_max_alpha, c='r')
        ax.plot(idx_max_alpha, average[idx_max_alpha], 'rX')
        ax.set_xlim(0, fs * (t_max + 2 * alpha))
        ax.xaxis.set_major_formatter(lambda x, pos: f"{x:.1g}\n{x / fs}s")

    return plateau_index


def compute_tonic_release_index(average, fs, t_flicker_start=10, t_flicker_end=29, dt_baseline=1, plot=None):
    """
    Calculate the tonic release index (TRi) from Franke et al. 2017.

    :param average: A 1d numpy array of trace
    :param fs: Sampling rate of the data in Hz.
    :param t_flicker_start: Time of the onset of the flicker in seconds.
    :param t_flicker_end: Time of the offset of the flicker in seconds.
    :param dt_baseline: Duration of the baseline window in seconds.
    :param plot: If True, plot the average trace and the response window.

    :return: Tonic release index (TRi).
    """

    idx_baseline_flicker_start = int(fs * (t_flicker_start - dt_baseline))
    idx_flicker_start = int(fs * t_flicker_start)
    idx_flicker_end = int(fs * t_flicker_end)

    baseline = np.median(average[idx_baseline_flicker_start:idx_flicker_start])
    average_flicker = average[idx_flicker_start:idx_flicker_end] - baseline

    tonic_release_index = np.sum(np.abs(average_flicker[average_flicker < 0])) / np.sum(np.abs(average_flicker))

    if plot:
        # Test is plot is instance of matplotlib axis
        if isinstance(plot, plt.Axes):
            ax = plot
        else:
            fig, ax = plt.subplots(1, 1, figsize=(4, 2))
        ax.set_title(f"TRi: {tonic_release_index:.2f}")
        ax.plot(average)
        ax.plot([idx_baseline_flicker_start, idx_flicker_start], [baseline, baseline], c='c')

        ax.axvline(idx_baseline_flicker_start, c='c', ls='--')
        ax.axvline(idx_flicker_start, c='r', ls='--')
        ax.axvline(idx_flicker_end, c='r', ls='--')

        baseline_trace = np.ones_like(average_flicker) * baseline

        ax.fill_between(np.arange(idx_flicker_start, idx_flicker_end),
                        baseline_trace, np.clip(average_flicker, None, 0) + baseline,
                        color='r', alpha=0.5, zorder=10)
        ax.fill_between(np.arange(idx_flicker_start, idx_flicker_end),
                        baseline_trace, np.clip(average_flicker, 0, None) + baseline,
                        color='g', alpha=0.5, zorder=10)
        ax.xaxis.set_major_formatter(lambda x, pos: f"{x:.1g}\n{x / fs}s")

    return tonic_release_index
