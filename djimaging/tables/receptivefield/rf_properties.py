import warnings
from abc import abstractmethod
from copy import deepcopy

import datajoint as dj
import numpy as np
from matplotlib import pyplot as plt

from djimaging.tables.receptivefield.rf_properties_utils import compute_gauss_srf_area, compute_surround_index, \
    fit_rf_model
from djimaging.tables.receptivefield.rf_utils import compute_explained_rf, resize_srf, split_strf, \
    compute_polarity_and_peak_idxs, merge_strf
from djimaging.utils.dj_utils import get_primary_key
from djimaging.utils.plot_utils import plot_srf, plot_trf, plot_signals_heatmap


class SplitRFParamsTemplate(dj.Lookup):
    database = ""

    @property
    def definition(self):
        definition = """
        split_rf_params_id: int # unique param set id
        ---
        blur_std : float
        blur_npix : int unsigned
        upsample_srf_scale : int unsigned
        peak_nstd : float  # How many standard deviations does a peak need to be considered peak?
        """
        return definition

    def add_default(self, skip_duplicates=False):
        """Add default preprocess parameter to table"""
        key = {
            'split_rf_params_id': 1,
            'blur_std': 1.,
            'blur_npix': 1,
            'upsample_srf_scale': 0,
            'peak_nstd': 1,
        }
        self.insert1(key, skip_duplicates=skip_duplicates)


class SplitRFTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = '''
        # Compute basic receptive fields
        -> self.rf_table
        -> self.split_rf_params_table
        ---
        srf: longblob  # spatio receptive field
        trf: longblob  # temporal receptive field
        split_qidx : float  # Quality index as explained variance of the sRF tRF split between 0 and 1
        trf_peak_idxs : blob  # Indexes of peaks in tRF
        '''
        return definition

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

        # Get preprocess params
        blur_std, blur_npix, upsample_srf_scale, peak_nstd = \
            self.split_rf_params_table().fetch1('blur_std', 'blur_npix', 'upsample_srf_scale', 'peak_nstd')

        # Get tRF and sRF
        srf, trf = split_strf(strf, blur_std=blur_std, blur_npix=blur_npix, upsample_srf_scale=upsample_srf_scale)

        # Make tRF always positive, so that sRF reflects the polarity of the RF
        polarity, peak_idxs = compute_polarity_and_peak_idxs(trf, nstd=peak_nstd)
        if polarity == -1:
            srf *= -1
            trf *= -1

        strf_fit = merge_strf(srf=resize_srf(srf, output_shape=strf.shape[1:]), trf=trf)
        split_qidx = compute_explained_rf(strf, strf_fit)

        # Save
        rf_key = deepcopy(key)
        rf_key['srf'] = srf
        rf_key['trf'] = trf
        rf_key['trf_peak_idxs'] = peak_idxs
        rf_key['split_qidx'] = split_qidx
        self.insert1(rf_key)

    def plot1(self, key=None):
        key = get_primary_key(table=self, key=key)

        dt = (self.rf_table() & key).fetch1("dt")
        srf, trf, peak_idxs = (self & key).fetch1("srf", "trf", "trf_peak_idxs")

        fig, axs = plt.subplots(1, 2, figsize=(8, 3))
        ax = axs[0]
        plot_srf(srf, ax=ax)

        ax = axs[1]
        plot_trf(trf, dt=dt, peak_idxs=peak_idxs, ax=ax)

        plt.tight_layout()
        plt.show()

    def plot(self, restriction=None):
        if restriction is None:
            restriction = dict()

        trf = np.stack((self & restriction).fetch('trf'))
        ax = plot_signals_heatmap(signals=trf)
        ax.set(title='tRF')
        plt.show()


class RFContoursParamsTemplate(dj.Lookup):
    database = ""

    @property
    def definition(self):
        definition = """
        rf_contours_params_id: int # unique param set id
        ---
        normalize_srf : enum('none', 'zeroone', 'zscore')
        levels : blob
        """
        return definition

    def add_default(self, skip_duplicates=False):
        """Add default preprocess parameter to table"""
        key = {
            'normalize_srf': 'none',
            'blur_std': 1.,
            'blur_npix': 1,
            'upsample_srf_scale': 0,
            'peak_nstd': 1,
        }
        self.insert1(key, skip_duplicates=skip_duplicates)


class RFContoursTemplate(dj.Computed):
    database = ""
    # TODO: Implement


class FitGauss2DRFTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = """
        -> self.split_rf_table
        ---
        srf_fit: longblob
        srf_params: longblob
        rf_area_um2: float # Area covered by 2 standard deviations
        rf_cdia_um: float # Circle equivalent diameter
        surround_index: float # Strength of surround in sRF
        rf_qidx: float # Quality index as explained variance of the sRF estimation between 0 and 1
        """
        return definition

    _polarity = None

    @property
    @abstractmethod
    def split_rf_table(self): pass

    @property
    @abstractmethod
    def stimulus_table(self): pass

    def make(self, key):
        srf = (self.split_rf_table() & key).fetch1("srf")
        stim_dict = (self.stimulus_table() & key).fetch1("stim_dict")

        # Fit RF model
        srf_fit, srf_params = fit_rf_model(srf, kind='gauss', polarity=self._polarity)

        # Compute properties
        area, diameter = compute_gauss_srf_area(srf_params, stim_dict["pix_scale_x_um"], stim_dict["pix_scale_y_um"])
        qi = compute_explained_rf(srf, srf_fit)
        surround_index = compute_surround_index(srf, srf_fit)

        # Save
        fit_key = deepcopy(key)
        fit_key['srf_fit'] = srf_fit
        fit_key['srf_params'] = srf_params
        fit_key['rf_cdia_um'] = diameter
        fit_key['rf_area_um2'] = area
        fit_key['surround_index'] = surround_index
        fit_key['rf_qidx'] = qi

        self.insert1(fit_key)

    def plot1(self, key=None):
        key = get_primary_key(table=self, key=key)
        srf = (self.split_rf_table() & key).fetch1("srf")
        srf_fit, rf_qidx = (self & key).fetch1("srf_fit", 'rf_qidx')

        vabsmax = np.maximum(np.max(np.abs(srf)), np.max(np.abs(srf_fit)))

        fig, axs = plt.subplots(1, 3, figsize=(10, 3))

        ax = axs[0]
        plot_srf(srf, ax=ax, vabsmax=vabsmax)
        ax.set_title('sRF')

        ax = axs[1]
        plot_srf(srf_fit, ax=ax, vabsmax=vabsmax)
        ax.set_title('sRF fit')

        ax = axs[2]
        plot_srf(srf - srf_fit, ax=ax, vabsmax=vabsmax)
        ax.set_title(f'Difference: QI={rf_qidx:.2f}')

        plt.tight_layout()
        plt.show()


class FitDoG2DRFTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = """
        -> self.split_rf_table
        ---
        srf_fit: longblob
        srf_center_fit: longblob
        srf_surround_fit: longblob
        srf_params: longblob
        srf_eff_center: longblob
        rf_qidx: float
        rf_area_um2: float # Area covered by 2 standard deviations
        rf_cdia_um: float # Circle equivalent diameter
        surround_index: float # Strength of surround in sRF
        """
        return definition

    _polarity = None

    @property
    @abstractmethod
    def split_rf_table(self):
        pass

    @property
    @abstractmethod
    def stimulus_table(self):
        pass

    def make(self, key):
        srf = (self.split_rf_table() & key).fetch1("srf")
        stim_dict = (self.stimulus_table() & key).fetch1("stim_dict")

        srf_fit, srf_center_fit, srf_surround_fit, srf_params, eff_polarity = fit_rf_model(
            srf, kind='dog', polarity=self._polarity)

        qi = compute_explained_rf(srf, srf_fit)

        if (self._polarity is None) or (eff_polarity == self._polarity):
            srf_eff_center = eff_polarity * np.clip(eff_polarity * srf_fit, 0, None)
        else:
            srf_eff_center = np.zeros_like(srf_fit)

        if np.allclose(srf_eff_center, 0.):
            warnings.warn(f'Failed to fit DoG to key={key}')
            return

        surround_index = compute_surround_index(srf=srf, srf_center=srf_eff_center)
        # Compute area from gaussian fit to effective center? TODO: is this really necessary?
        _, srf_gauss_params = fit_rf_model(srf_eff_center, kind='gauss', polarity=self._polarity)
        area, diameter = compute_gauss_srf_area(
            srf_gauss_params, stim_dict["pix_scale_x_um"], stim_dict["pix_scale_y_um"])

        # Save
        fit_key = deepcopy(key)
        fit_key['srf_fit'] = srf_fit
        fit_key['srf_center_fit'] = srf_center_fit
        fit_key['srf_surround_fit'] = srf_surround_fit
        fit_key['srf_params'] = srf_params
        fit_key['srf_eff_center'] = srf_eff_center
        fit_key['rf_cdia_um'] = diameter
        fit_key['rf_area_um2'] = area
        fit_key['surround_index'] = surround_index
        fit_key['rf_qidx'] = qi

        self.insert1(fit_key)

    def plot1(self, key=None):
        key = get_primary_key(table=self, key=key)
        srf = (self.split_rf_table() & key).fetch1("srf")
        srf_fit, srf_center_fit, srf_surround_fit, srf_eff_center, rf_qidx = (self & key).fetch1(
            "srf_fit", 'srf_center_fit', 'srf_surround_fit', 'srf_eff_center', 'rf_qidx')

        vabsmax = np.maximum(np.max(np.abs(srf)), np.max(np.abs(srf_fit)))

        fig, axs = plt.subplots(2, 3, figsize=(10, 6))

        ax = axs[0, 0]
        plot_srf(srf, ax=ax, vabsmax=vabsmax)
        ax.set_title('sRF')

        ax = axs[0, 1]
        plot_srf(srf_fit, ax=ax, vabsmax=vabsmax)
        ax.set_title('sRF fit')

        ax = axs[0, 2]
        plot_srf(srf - srf_fit, ax=ax, vabsmax=vabsmax)
        ax.set_title(f'Difference: QI={rf_qidx:.2f}')

        ax = axs[1, 0]
        plot_srf(srf_center_fit, ax=ax, vabsmax=vabsmax)
        ax.set_title('sRF center fit')

        ax = axs[1, 1]
        plot_srf(srf_surround_fit, ax=ax, vabsmax=vabsmax)
        ax.set_title('sRF surround fit')

        ax = axs[1, 2]
        plot_srf(srf_eff_center, ax=ax, vabsmax=vabsmax)
        ax.set_title('sRF effective center')

        plt.tight_layout()
        plt.show()
