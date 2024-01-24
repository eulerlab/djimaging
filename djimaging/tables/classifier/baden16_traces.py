"""
Preprocess traces to match Baden et al. 2016 dataset.
Mostly used to match traces for RGC classification.
"""

from abc import abstractmethod

import numpy as np
from matplotlib import pyplot as plt

from djimaging.utils.trace_utils import get_mean_dt

from djimaging.utils.dj_utils import get_primary_key
from scipy import interpolate
import datajoint as dj


class Baden16TracesTemplate(dj.Computed):
    database = ""

    _shift_chirp = 1
    _shift_bar = -4

    _stim_name_chirp = 'gChirp'
    _stim_name_bar = 'movingbar'

    @property
    def definition(self):
        definition = """
        # Process traces to match Baden 2016 format, e.g. for RGC classifier.
        -> self.averages_table().proj(avg_stim_name='stim_name')
        -> self.os_ds_table().proj(os_ds_stim_name='stim_name')
        ---
        preproc_chirp:   blob  # preprocessed chirp trace (averaged, downsampled and normalized)
        preproc_bar:     blob  # preprocessed bar (time component in pref. dir., averaged and rolled)
        """
        return definition

    @property
    def key_source(self):
        try:
            return ((self.averages_table & dict(stim_name=self._stim_name_chirp)).proj(avg_stim_name='stim_name') *
                    (self.os_ds_table & dict(stim_name=self._stim_name_bar)).proj(os_ds_stim_name='stim_name'))
        except (AttributeError, TypeError):
            pass

    @property
    @abstractmethod
    def averages_table(self):
        pass

    @property
    @abstractmethod
    def os_ds_table(self):
        pass

    def make(self, key):
        chirp_average, chirp_average_times = (self.averages_table & dict(stim_name=self._stim_name_chirp) & key).fetch1(
            'average', 'average_times')
        chirp_dt = get_mean_dt(chirp_average_times)[0]

        bar_time_component, bar_dt = (self.os_ds_table & dict(stim_name=self._stim_name_bar) & key).fetch1(
            'time_component', 'time_component_dt')

        preproc_chirp = preprocess_chirp(chirp_average, dt=chirp_dt, shift=self._shift_chirp)
        preproc_bar = preprocess_bar(bar_time_component, dt=bar_dt, shift=self._shift_bar)

        key = key.copy()
        key['preproc_chirp'] = preproc_chirp
        key['preproc_bar'] = preproc_bar
        self.insert1(key)

    def plot1(self, key=None):
        key = get_primary_key(table=self, key=key)

        preproc_chirp = (self & key).fetch1('preproc_chirp')
        preproc_bar = (self & key).fetch1('preproc_bar')

        fig, axs = plt.subplots(1, 2, figsize=(12, 3), gridspec_kw=dict(width_ratios=(3, 1)))

        ax = axs[0]
        ax.plot(preproc_chirp)
        ax.set(xlabel='frames', ylabel='signal', title='chirp')

        ax = axs[1]
        ax.plot(preproc_bar)
        ax.set(xlabel='frames', ylabel='signal', title='bar')

        plt.show()


def preprocess_chirp(chirp_average, dt, shift=1):
    """
    Preprocesses chirp traces by resampling.
    The previous cutting of the last 7 frames has been replaced by an equivalent speedup factor.
    Traces are baseline corrected and normalized to be in the range [-1, 1]
    :param chirp_average: chirp average trace
    :param dt: float, time between two datapoints of chirp_average
    :param shift: int, number of frames to shift chirp trace. Old chirp was different. Shift corrects for this.
    :return: array; shape rois x frames
    """
    if chirp_average.ndim > 1:
        raise ValueError(f"chirp_trace must be 1D, but has shape {chirp_average.shape}")

    baden_dt = 0.128

    # Extract original baseline before shifting
    baden16_baseline = np.mean(chirp_average[:int(np.round(8 * baden_dt / dt))])

    # Shift to re-correct for stimulator delay
    time_avg = np.arange(chirp_average.size) * dt + (shift * baden_dt)

    baden16_time = np.linspace(0, 32, 249)

    # Resample to Baden frequency which was (in stimulus space) slightly different
    baden16_average = interpolate.interp1d(
        time_avg, chirp_average, assume_sorted=True, bounds_error=False,
        fill_value=(chirp_average[0], chirp_average[-1]))(baden16_time)

    # Normalize
    baden16_average -= baden16_baseline
    baden16_average /= np.max(np.abs(baden16_average))

    return baden16_average


def preprocess_bar(bar_average, dt, shift=-4):
    """
    Preprocesses bar time component by resampling if necessary and by rolling to match Baden traces;
    :param bar_average: moving bar average trace in preferred direction
    :param dt: float, time between two frames
    :param shift: int, number of frames to shift bar trace. Old bar has different. Shift corrects for this.
    :return: array; shape rois x frames
    """
    if bar_average.ndim > 1:
        raise ValueError(f"time_component must be 1D, but has shape {bar_average.shape}")

    baden_dt = 0.128

    time = np.arange(bar_average.size) * dt
    baden16_time = np.arange(32) * baden_dt
    baden16_average = interpolate.interp1d(
        time, bar_average, assume_sorted=True, bounds_error=False, fill_value='extrapolate')(baden16_time)
    # Shift
    baden16_average = np.roll(baden16_average, shift)
    return baden16_average
