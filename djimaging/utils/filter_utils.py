import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import butter, lfilter, freqz, filtfilt


class LowPassFilter:
    """Butterworth low-pass filter wrapper with optional forward-backward filtering.

    Parameters
    ----------
    fs : float
        Sampling frequency of the signal in Hz.
    cutoff : float
        Cutoff frequency in Hz.
    order : int, optional
        Filter order. Default is 6.
    direction : str, optional
        Filtering direction: ``'l'`` for causal (``lfilter``) or ``'ff'`` for
        zero-phase forward-backward (``filtfilt``). Default is ``'l'``.

    Raises
    ------
    NotImplementedError
        If `direction` is not ``'l'`` or ``'ff'``.
    """

    def __init__(self, fs: float, cutoff: float, order: int = 6, direction: str = 'l') -> None:
        self.fs = float(fs)
        self.cutoff = float(cutoff)
        self.order = int(order)
        if direction == 'l':
            self.filter = lfilter
        elif direction == 'ff':
            self.filter = filtfilt
        else:
            raise NotImplementedError()

        self.nyq = 0.5 * self.fs
        self.normal_cutoff = self.cutoff / self.nyq
        self.b, self.a = butter(self.order, self.normal_cutoff, btype='low', analog=False)

    def filter_data(self, trace: np.ndarray) -> np.ndarray:
        """Apply the low-pass filter to a 1-D trace.

        Parameters
        ----------
        trace : np.ndarray
            1-D array of signal values to filter.

        Returns
        -------
        np.ndarray
            Filtered signal of the same shape as `trace`.
        """
        y = self.filter(self.b, self.a, trace)
        return y

    def plot(self) -> None:
        """Plot the frequency response and an example filtered signal.

        Returns
        -------
        None
        """
        w, h = freqz(self.b, self.a, worN=8000)
        plt.subplot(2, 1, 1)
        plt.plot(0.5 * self.fs * w / np.pi, np.abs(h), 'b')
        plt.plot(self.cutoff, 0.5 * np.sqrt(2), 'ko')
        plt.axvline(self.cutoff, color='k')
        plt.xlim(0, 0.5 * self.fs)
        plt.title("Lowpass Filter Frequency Response")
        plt.xlabel('Frequency [Hz]')
        plt.grid()

        # First make some data to be filtered.
        T = 30. / self.cutoff  # seconds
        n = int(T * self.fs)  # total number of samples
        t = np.linspace(0, T, n, endpoint=False)

        # "Noisy" data
        data = np.sin(self.cutoff * 0.1 * 2 * np.pi * t) + 0.4 * np.cos(
            self.cutoff * 1.77 * 2 * np.pi * t) + 0.3 * np.sin(self.cutoff * 2.9 * 2 * np.pi * t)

        # Filter the data, and plot both the original and filtered signals.
        y = self.filter_data(data)

        plt.subplot(2, 1, 2)
        plt.plot(t, data, 'b-', label='data')
        plt.plot(t, y, 'g-', linewidth=2, label='filtered data')
        plt.xlabel('Time [sec]')
        plt.grid()
        plt.legend()

        plt.subplots_adjust(hspace=0.35)
        plt.show()


def lowpass_filter_trace(trace: np.ndarray, fs: float, f_cutoff: float) -> np.ndarray:
    """Apply a zero-phase Butterworth low-pass filter to a 1-D trace.

    Parameters
    ----------
    trace : np.ndarray
        1-D input signal to filter.
    fs : float
        Sampling frequency of `trace` in Hz.
    f_cutoff : float
        Cutoff frequency in Hz.

    Returns
    -------
    np.ndarray
        Low-pass filtered signal of the same shape as `trace`.
    """
    trace_filtered = LowPassFilter(fs=fs, cutoff=f_cutoff, order=6, direction='ff').filter_data(trace)
    return trace_filtered


def resample_trace(tracetime: np.ndarray, trace: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Resample a trace to a uniform time grid via linear interpolation.

    Parameters
    ----------
    tracetime : np.ndarray
        1-D array of original sample times.
    trace : np.ndarray
        1-D array of signal values at `tracetime`.
    dt : float
        Desired time step of the resampled grid.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(tracetime_resampled, trace_resampled)`` – uniformly spaced time
        array and the corresponding interpolated signal values.
    """
    tracetime_resampled = np.arange(tracetime[0], np.nextafter(tracetime[-1], tracetime[-1] + dt), dt)
    trace_resampled = np.interp(x=tracetime_resampled, xp=tracetime, fp=trace)
    return tracetime_resampled, trace_resampled


def upsample_stim(stimtime: np.ndarray, stim: np.ndarray, fupsample: int) -> tuple[np.ndarray, np.ndarray]:
    """Upsample a stimulus by repeating each frame `fupsample` times.

    Parameters
    ----------
    stimtime : np.ndarray
        1-D array of stimulus frame times.
    stim : np.ndarray
        Stimulus array whose first axis corresponds to frames.
    fupsample : int
        Integer upsampling factor.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(stimtime_upsampled, stim_upsampled)`` – expanded time array and
        stimulus array with each frame repeated `fupsample` times.
    """
    dt = np.mean(np.diff(stimtime))
    stimtime = np.sort(np.concatenate([stimtime + dt * float(i / fupsample) for i in range(0, fupsample)]))
    stim = np.repeat(stim, fupsample, axis=0)
    return stimtime, stim


def upsample_trace(tracetime: np.ndarray, trace: np.ndarray,
                   fupsample: int) -> tuple[np.ndarray, np.ndarray]:
    """Upsample a trace by linear interpolation between existing samples.

    Parameters
    ----------
    tracetime : np.ndarray
        1-D array of original sample times.
    trace : np.ndarray
        1-D array of signal values at `tracetime`.
    fupsample : int
        Integer upsampling factor (must be > 1).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(tracetime_upsampled, trace_upsampled)`` – finer time grid and the
        corresponding interpolated signal values.

    Raises
    ------
    AssertionError
        If `fupsample` is not an integer greater than 1, or if the resulting
        time grid is inconsistent.
    """
    assert fupsample == int(fupsample), fupsample
    fupsample = int(fupsample)
    assert fupsample > 1, fupsample

    dt = np.mean(np.diff(tracetime))
    tracetime_upsampled = np.tile(tracetime, (fupsample, 1)).T
    tracetime_upsampled += np.linspace(0, 1, fupsample, endpoint=False) * dt
    tracetime_upsampled = tracetime_upsampled.flatten()

    diffs = np.diff(tracetime_upsampled)
    mu = np.mean(diffs)
    std = np.std(diffs)
    assert np.isclose(mu, dt / fupsample, atol=mu / 10.), f"{mu} {dt} {fupsample}"
    assert mu > 10 * std

    trace_upsampled = np.interp(tracetime_upsampled, tracetime, trace)

    return tracetime_upsampled, trace_upsampled


def downsample_trace(tracetime: np.ndarray, trace: np.ndarray,
                     fdownsample: int) -> tuple[np.ndarray, np.ndarray]:
    """Downsample a trace by averaging blocks of `fdownsample` samples.

    Parameters
    ----------
    tracetime : np.ndarray
        1-D or 2-D array of sample times (last axis = time).
    trace : np.ndarray
        1-D or 2-D array of signal values (last axis = time).
    fdownsample : int
        Integer downsampling factor (must be > 1).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(tracetime_downsampled, trace_downsampled)`` – coarser time grid and
        the block-averaged signal values.

    Raises
    ------
    AssertionError
        If `fdownsample` is not an integer greater than 1, or if the output
        shapes are inconsistent.
    """
    assert isinstance(fdownsample, int)
    assert fdownsample > 1

    tracetime = np.atleast_2d(tracetime.copy())
    trace = np.atleast_2d(trace.copy())

    n_new = trace.shape[1] // fdownsample
    n_max = n_new * fdownsample

    trace = trace[:, :n_max]
    trace_downsampled = np.mean(trace.reshape(trace.shape[0], n_new, fdownsample), axis=2)
    tracetime_downsampled = tracetime[:, ::fdownsample][:, :n_new]

    tracetime_downsampled, trace_downsampled = tracetime_downsampled.squeeze(), trace_downsampled.squeeze()

    assert tracetime_downsampled.shape[-1] == trace_downsampled.shape[-1], \
        f"{tracetime_downsampled.shape} != {trace_downsampled.shape} != {n_new}"

    return tracetime_downsampled, trace_downsampled
