from abc import abstractmethod
from copy import deepcopy

import datajoint as dj
import numpy as np
from matplotlib import pyplot as plt

from djimaging.utils.receptive_fields.temporal_rf_utils import compute_polarity_and_peak_idxs
from djimaging.utils.receptive_fields.split_rf_utils import compute_explained_rf, resize_srf, merge_strf, split_strf
from djimaging.utils import math_utils
from djimaging.utils.dj_utils import get_primary_key
from djimaging.utils.plot_utils import plot_srf, plot_trf, plot_signals_heatmap
from djimaging.utils.trace_utils import sort_traces


class SplitRFParamsTemplate(dj.Lookup):
    database = ""

    @property
    def definition(self):
        definition = """
        split_rf_params_id: tinyint unsigned # unique param set id
        ---
        method : varchar(63)  # Method used to split RF, currently available are SVD, STD, and MAX
        blur_std : float
        blur_npix : int unsigned
        upsample_srf_scale : int unsigned
        peak_nstd : float  # How many standard deviations does a peak need to be considered peak?
        npeaks_max : int unsigned # Maximum number of peaks, ignored if zero
        """
        return definition

    def add_default(self, skip_duplicates=False, **params):
        """Add default preprocess parameter to table"""
        key = dict(
            split_rf_params_id=1,
            blur_std=1.,
            blur_npix=1,
            method='SVD',
            upsample_srf_scale=0,
            peak_nstd=1,
            npeaks_max=0,
        )
        key.update(**params)
        self.insert1(key, skip_duplicates=skip_duplicates)


class SplitRFTemplate(dj.Computed):
    database = ""
    _max_dt_future = np.inf

    @property
    def definition(self):
        definition = '''
        # Compute basic receptive fields
        -> self.rf_table
        -> self.split_rf_params_table
        ---
        srf: longblob  # spatio receptive field
        trf: longblob  # temporal receptive field
        polarity : tinyint  # Polarity of the RF, 1 for positive, -1 for negative
        split_qidx : float  # Quality index as explained variance of the sRF tRF split between 0 and 1
        trf_peak_idxs : blob  # Indexes of peaks in tRF
        '''
        return definition

    @property
    def key_source(self):
        try:
            return self.rf_table.proj() * self.split_rf_params_table.proj()
        except (AttributeError, TypeError):
            pass

    @property
    @abstractmethod
    def rf_table(self):
        pass

    @property
    @abstractmethod
    def split_rf_params_table(self):
        pass

    def make(self, key):
        # Get data
        strf = (self.rf_table() & key).fetch1("rf")
        rf_time = self.fetch1_rf_time(key=key)

        # Get preprocess params
        method, blur_std, blur_npix, upsample_srf_scale, peak_nstd, npeaks_max = (
                self.split_rf_params_table & key).fetch1(
            'method', 'blur_std', 'blur_npix', 'upsample_srf_scale', 'peak_nstd', 'npeaks_max')

        # Get tRF and sRF
        srf, trf, split_qidx = split_strf(
            strf, method=method, blur_std=blur_std, blur_npix=blur_npix, upsample_srf_scale=upsample_srf_scale)

        # Make tRF always positive, so that sRF reflects the polarity of the RF
        polarity, peak_idxs = compute_polarity_and_peak_idxs(
            rf_time=rf_time, trf=trf, nstd=peak_nstd, npeaks_max=npeaks_max if npeaks_max > 0 else None,
            max_dt_future=self._max_dt_future)

        if method.lower() in ['svd']:
            if polarity == -1:
                srf *= -1
                trf *= -1
                polarity = 1

        if split_qidx is None:
            strf_fit = merge_strf(srf=resize_srf(srf, output_shape=strf.shape[1:]), trf=trf)
            split_qidx = compute_explained_rf(strf, strf_fit)

        # Save
        rf_key = deepcopy(key)
        rf_key['srf'] = srf.astype(np.float32)
        rf_key['trf'] = trf.astype(np.float32)
        rf_key['polarity'] = polarity
        rf_key['trf_peak_idxs'] = peak_idxs
        rf_key['split_qidx'] = split_qidx
        self.insert1(rf_key)

    def fetch1_rf_time(self, key):
        try:
            rf_time = (self.rf_table & key).fetch1('rf_time')
        except dj.DataJointError:
            try:
                rf_time = (self.rf_table & key).fetch1('model_dict')['rf_time']
            except dj.DataJointError:
                rf_time = (self.rf_table.params_table & key).fetch1('rf_time')
        return rf_time

    def plot1(self, key=None):
        key = get_primary_key(table=self, key=key)

        rf_time = self.fetch1_rf_time(key=key)
        srf, trf, peak_idxs = (self & key).fetch1("srf", "trf", "trf_peak_idxs")

        fig, axs = plt.subplots(1, 2, figsize=(8, 3))
        ax = axs[0]
        plot_srf(srf, ax=ax)

        ax = axs[1]
        plot_trf(trf, t_trf=rf_time, peak_idxs=peak_idxs, ax=ax)

        plt.tight_layout()
        plt.show()

    def plot(self, restriction=None, sort=False):
        if restriction is None:
            restriction = dict()

        trf = math_utils.padded_vstack((self & restriction).fetch('trf'))

        if sort:
            trf = sort_traces(trf)

        ax = plot_signals_heatmap(signals=trf)
        ax.set(title='tRF')
        plt.show()
