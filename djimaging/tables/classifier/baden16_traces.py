from abc import abstractmethod

import numpy as np
from matplotlib import pyplot as plt

from djimaging.utils.trace_utils import get_mean_dt

from djimaging.utils.dj_utils import get_secondary_keys, get_primary_key
from scipy import interpolate
import datajoint as dj
from cached_property import cached_property


def preprocess_chirp(chirp_average, dt, shift=2, n_int=249):
    """
    Preprocesses chirp traces by resampling if necessary, averaging across repetitions, cutting off
    last 7 frames and downsampling to 249 frames to match Baden traces;
    subtracting mean of first 8 frames (baseline subtraction); and normalizing
    to be in the range [-1, 1]
    :param chirp_average: chirp average trace
    :param dt: float, time between two frames
    :param shift: int, number of frames to shift chirp trace. Old chirp was different. Shift corrects for this.
    :param n_int: Number of points to interpolate over. For new chirp stimulus, this is 249. For old chirp this is 251.
    :return: array; shape rois x frames
    """
    if chirp_average.ndim > 1:
        raise ValueError(f"chirp_trace must be 1D, but has shape {chirp_average.shape}")

    # extract original baseline before shifting
    baden16_baseline = np.mean(chirp_average[:8]) 
    
    # Shift
    baden16_average = np.roll(chirp_average, shift)
    
    time_avg = np.arange(baden16_average.size) * dt  # Allow different frequencies
    baden16_time = np.arange(249) * (32. / (n_int - 1))

    # Resample to Baden frequency which was (in stimulus space) slightly different
    baden16_average = interpolate.interp1d(
        time_avg, baden16_average, assume_sorted=True, bounds_error=False, fill_value='extrapolate')(baden16_time)

    # Normalize
    baden16_average -= baden16_baseline
    baden16_average /= np.max(np.abs(baden16_average))

    return baden16_average


def preprocess_bar(bar_average, dt, shift=-3):
    """
    Preprocesses bar time component by resampling if necessary and by rolling to match Baden traces;
    :param bar_average: moving bar average trace in preferred direction
    :param dt: float, time between two frames
    :param shift: int, number of frames to shift bar trace. Old bar has different. Shift corrects for this.
    :return: array; shape rois x frames
    """
    if bar_average.ndim > 1:
        raise ValueError(f"time_component must be 1D, but has shape {bar_average.shape}")
    time = np.arange(bar_average.size) * dt
    baden16_time = np.arange(32) * 0.128
    baden16_average = interpolate.interp1d(
        time, bar_average, assume_sorted=True, bounds_error=False, fill_value='extrapolate')(baden16_time)
    # Shift
    baden16_average = np.roll(baden16_average, shift)
    return baden16_average


class Baden16TracesTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = """
        # Process traces to match Baden 2016 format, e.g. for RGC classifier.
        -> self.roi_table
        -> self.preprocessparams_table
        ---
        preproc_chirp:   blob  # preprocessed chirp trace (averaged, downsampled and normalized)
        preproc_bar:     blob  # preprocessed bar
        """
        return definition

    _shift_chirp = 2
    _shift_bar = -3
    _chirp_n_int = 249

    _stim_name_chirp = 'gChirp'
    _stim_name_bar = 'movingbar'
    _restr = dict()  # e.g. dict(condition='control')

    @property
    def key_source(self):
        try:
            return self.roi_table.proj() * self.preprocessparams_table.proj() \
                & (self.chirp_tab.proj() * self.bar_tab.proj() & self._restr)
        except (AttributeError, TypeError):
            pass

    @property
    @abstractmethod
    def roi_table(self):
        pass

    @property
    @abstractmethod
    def preprocessparams_table(self):
        pass

    @property
    @abstractmethod
    def averages_table(self):
        pass

    @property
    @abstractmethod
    def os_ds_table(self):
        pass

    @cached_property
    def chirp_tab(self):
        secondary_keys = get_secondary_keys(self.averages_table)
        return (self.averages_table() & f"stim_name = '{self._stim_name_chirp}'" & self._restr).proj(
            **{f"chirp_{k}": k for k in secondary_keys + ['stim_name']})

    @cached_property
    def bar_tab(self):
        secondary_keys = get_secondary_keys(self.os_ds_table)
        return (self.os_ds_table() & f"stim_name = '{self._stim_name_bar}'" & self._restr).proj(
            **{f"bar_{k}": k for k in secondary_keys + ['stim_name']})

    def make(self, key):
        chirp_average, chirp_average_times = (self.chirp_tab & key).fetch1('chirp_average', 'chirp_average_times')
        chirp_dt = get_mean_dt(chirp_average_times)[0]

        bar_time_component, bar_dt = (self.bar_tab & key).fetch1('bar_time_component', 'bar_time_component_dt')

        preproc_chirp = preprocess_chirp(chirp_average, dt=chirp_dt, shift=self._shift_chirp, n_int=self._chirp_n_int)
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
