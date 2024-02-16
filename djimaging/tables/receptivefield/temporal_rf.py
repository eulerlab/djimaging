from abc import abstractmethod

import datajoint as dj
import numpy as np
from matplotlib import pyplot as plt

from djimaging.utils.receptive_fields.temporal_rf_utils import compute_trf_transience_index, compute_half_amp_width, \
    compute_main_peak_lag
from djimaging.utils.dj_utils import get_primary_key
from djimaging.utils.plot_utils import plot_trf, set_long_title


class TempRFPropertiesTemplate(dj.Computed):
    database = ""
    _max_dt_future = np.inf

    @property
    def definition(self):
        definition = """
        -> self.split_rf_table
        ---
        transience_idx : float
        half_amp_width : float
        main_peak_lag : float
        """
        return definition

    @property
    def key_source(self):
        try:
            return self.split_rf_table.proj()
        except (AttributeError, TypeError):
            pass

    @property
    @abstractmethod
    def rf_table(self):
        pass

    @property
    @abstractmethod
    def split_rf_table(self):
        pass

    def fetch1_rf_time(self, key):
        try:
            rf_time = (self.rf_table & key).fetch1('rf_time')
        except dj.DataJointError:
            rf_time = (self.rf_table & key).fetch1('model_dict')['rf_time']
        return rf_time

    def make(self, key):
        rf_time = self.fetch1_rf_time(key=key)
        trf, trf_peak_idxs = (self.split_rf_table & key).fetch1('trf', 'trf_peak_idxs')

        transience_idx = compute_trf_transience_index(
            rf_time, trf, trf_peak_idxs, max_dt_future=self._max_dt_future)
        half_amp_width = compute_half_amp_width(
            rf_time, trf, trf_peak_idxs, plot=False, max_dt_future=self._max_dt_future)
        main_peak_lag = compute_main_peak_lag(
            rf_time, trf, trf_peak_idxs, plot=False, max_dt_future=self._max_dt_future)

        key = key.copy()
        key['transience_idx'] = transience_idx if transience_idx is not None else -1.
        key['half_amp_width'] = half_amp_width if half_amp_width is not None else -1.
        key['main_peak_lag'] = main_peak_lag if main_peak_lag is not None else -1.

        self.insert1(key)

    def plot(self):
        fig, axs = plt.subplots(1, 3, figsize=(12, 3))
        for ax, name in zip(axs, ['transience_idx', 'half_amp_width', 'main_peak_lag']):
            ax.hist(self.fetch(name))
            ax.set_title(name)
        plt.tight_layout()
        return fig, axs

    def plot1(self, key=None):
        key = get_primary_key(table=self, key=key)

        rf_time = self.fetch1_rf_time(key=key)
        trf, peak_idxs = (self.split_rf_table & key).fetch1('trf', 'trf_peak_idxs')
        transience_idx, half_amp_width, main_peak_lag = (self & key).fetch1(
            'transience_idx', 'half_amp_width', 'main_peak_lag')

        fig, ax = plt.subplots(1, 1, figsize=(6, 2.5))
        plot_trf(trf, t_trf=rf_time, peak_idxs=peak_idxs, ax=ax)
        ax.axvline(-main_peak_lag, color='k', ls='--', label='main_peak')
        ax.axvline(-main_peak_lag - half_amp_width / 2, color='gray', ls=':', label='width')
        ax.axvline(-main_peak_lag + half_amp_width / 2, color='gray', ls=':', label='_')
        ax.axvline(0, color='gray', ls='-', label='_', zorder=-2)
        set_long_title(ax=ax, title=key, fontsize=8)
        plt.legend()
        plt.tight_layout()
        return fig, ax
