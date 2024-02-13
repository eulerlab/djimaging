from abc import abstractmethod

import numpy as np
import datajoint as dj
from matplotlib import pyplot as plt

from djimaging.tables.receptivefield.rf_properties_utils import compute_trf_transience_index, compute_half_amp_width, \
    compute_main_peak_lag
from djimaging.tables.receptivefield.rf_utils import compute_polarity_and_peak_idxs
from djimaging.utils.dj_utils import get_primary_key
from djimaging.utils.plot_utils import set_long_title, plot_trf, prep_long_title


class CenterSurroundParamsTemplate(dj.Lookup):
    database = ""

    @property
    def definition(self):
        definition = """
        cs_params_id: int # unique param set id
        ---
        peak_nstd : float  # How many standard deviations does a peak need to be considered peak?
        npeaks_max : int unsigned # Maximum number of peaks, ignored if zero
        """
        return definition

    def add_default(self, skip_duplicates=False, **params):
        """Add default preprocess parameter to table"""
        key = dict(
            cs_params_id=1,
            peak_nstd=1,
            npeaks_max=0,
        )
        key.update(**params)
        self.insert1(key, skip_duplicates=skip_duplicates)


class CenterSurroundTemplate(dj.Computed):
    database = ""
    _max_dt_future = np.inf

    @property
    def definition(self):
        definition = """
        -> self.color_rf_table
        -> self.cs_params_table
        ---
        c_polarity_idx : float  # Polarity index of center RF
        c_transience_idx : float  # Transience index of center RF
        c_half_amp_width : float  # Half amplitude width of center RF
        c_main_peak_lag : float  # Main peak lag of center RF
        s_polarity_idx : float  # Polarity index of surround RF
        s_transience_idx : float  # Transience index of surround RF
        s_half_amp_width : float  # Half amplitude width of surround RF
        s_main_peak_lag : float  # Main peak lag of surround RF
        """
        return definition

    @property
    def key_source(self):
        try:
            return self.color_rf_table.proj() * self.cs_params_table.proj()
        except (AttributeError, TypeError):
            pass

    @property
    @abstractmethod
    def color_rf_table(self):
        pass

    @property
    @abstractmethod
    def cs_params_table(self):
        pass

    def fetch1_rf_time(self, key):
        try:
            rf_time = (self.color_rf_table & key).fetch1('rf_time')
        except dj.DataJointError:
            rf_time = (self.color_rf_table & key).fetch1('model_dict')['rf_time']
        return rf_time

    def make(self, key, plot=False):
        rf_time = self.fetch1_rf_time(key=key)
        rf = (self.color_rf_table & key).fetch1('rf')
        peak_nstd, npeaks_max = (self.cs_params_table & key).fetch1('peak_nstd', 'npeaks_max')

        polarity_idxs = []
        transience_idxs = []
        half_amp_widths = []
        main_peak_lags = []

        for kernel in rf.T:
            polarity, peak_idxs = compute_polarity_and_peak_idxs(
                rf_time=rf_time, trf=kernel, nstd=peak_nstd, npeaks_max=npeaks_max if npeaks_max > 0 else None,
                max_dt_future=self._max_dt_future)

            polarity_idxs.append(polarity)
            transience_idxs.append(compute_trf_transience_index(
                rf_time, polarity * kernel, peak_idxs, max_dt_future=self._max_dt_future))
            half_amp_widths.append(compute_half_amp_width(
                rf_time, polarity * kernel, peak_idxs, plot=plot, max_dt_future=self._max_dt_future))
            main_peak_lags.append(compute_main_peak_lag(
                rf_time, polarity * kernel, peak_idxs, plot=plot, max_dt_future=self._max_dt_future))

        key = key.copy()

        for i, prefix in enumerate(['c', 's']):
            key[f'{prefix}_polarity_idx'] = polarity_idxs[i] if polarity_idxs[i] is not None else -1.
            key[f'{prefix}_transience_idx'] = transience_idxs[i] if transience_idxs[i] is not None else -1.
            key[f'{prefix}_half_amp_width'] = half_amp_widths[i] if half_amp_widths[i] is not None else -1.
            key[f'{prefix}_main_peak_lag'] = main_peak_lags[i] if main_peak_lags[i] is not None else -1.

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
        rf = (self.color_rf_table & key).fetch1('rf')

        peak_nstd, npeaks_max = (self.cs_params_table & key).fetch1('peak_nstd', 'npeaks_max')

        fig, axs = plt.subplots(1, 2, figsize=(6, 4), sharey=True)
        fig.suptitle(prep_long_title(key))
        prefixes = ['c', 's']
        for i, kernel in enumerate(rf.T):
            polarity, peak_idxs = compute_polarity_and_peak_idxs(
                rf_time=rf_time, trf=kernel, nstd=peak_nstd, npeaks_max=npeaks_max if npeaks_max > 0 else None,
                max_dt_future=self._max_dt_future)

            prefix = prefixes[i]
            transience_idx, half_amp_width, main_peak_lag = (self & key).fetch1(
                f'{prefix}_transience_idx', f'{prefix}_half_amp_width', f'{prefix}_main_peak_lag')

            ax = axs[i]
            plot_trf(kernel, t_trf=rf_time, peak_idxs=peak_idxs, ax=ax, lim_y=False)
            ax.set_title('')
            ax.set_title(
                f'{prefix} RF\npolarity={polarity:.1f}\n'
                f'transience={transience_idx:.1f}\n'
                f'half_amp_width={half_amp_width:.1f}\n'
                f'main_peak_lag={main_peak_lag:.1f}', loc='left', ha='left')
            ax.axvline(-main_peak_lag, color='k', ls='--', label='main_peak')
            ax.axvline(-main_peak_lag - half_amp_width / 2, color='gray', ls=':', label='width')
            ax.axvline(-main_peak_lag + half_amp_width / 2, color='gray', ls=':', label='_')
            ax.axvline(0, color='gray', ls='-', label='_', zorder=-2)

        plt.legend()
        plt.tight_layout()
        return fig, axs
