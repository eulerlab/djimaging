import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import butter, lfilter, freqz, filtfilt


class LowPassFilter:

    def __init__(self, fs, cutoff, order=6, direction='l'):
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

    def filter_data(self, trace):
        """Apply lowpass filter to trace"""
        y = self.filter(self.b, self.a, trace)
        return y

    def plot(self):
        """Plot the frequency response."""

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


def lowpass_filter_trace(trace, fs, f_cutoff):
    """Apply low pass filter to trace"""
    trace_filtered = LowPassFilter(fs=fs, cutoff=f_cutoff, order=6, direction='ff').filter_data(trace)
    return trace_filtered


def resample_trace(tracetime, trace, dt):
    """Resample trace through linear interpolation"""
    tracetime_resampled = np.arange(tracetime[0], np.nextafter(tracetime[-1], tracetime[-1] + dt), dt)
    trace_resampled = np.interp(x=tracetime_resampled, xp=tracetime, fp=trace)
    return tracetime_resampled, trace_resampled


def upsample_stim(stimtime: np.ndarray, stim: np.ndarray, fupsample: int) -> (np.ndarray, np.ndarray):
    dt = np.mean(np.diff(stimtime))
    stimtime = np.sort(np.concatenate([stimtime + dt * float(i / fupsample) for i in range(0, fupsample)]))
    stim = np.repeat(stim, fupsample, axis=0)
    return stimtime, stim


def upsample_trace(tracetime, trace, fupsample):
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


def downsample_trace(tracetime, trace, fdownsample):
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
