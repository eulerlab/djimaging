"""
Tables for receptive field feature extraction.
"""

import warnings
from abc import abstractmethod
from copy import deepcopy

import datajoint as dj
import numpy as np

from matplotlib import pyplot as plt

from djimaging.utils.receptive_fields.spatial_rf_utils import compute_gauss_srf_area, compute_surround_index, \
    compute_center_index, fit_rf_model
from djimaging.utils.dj_utils import get_primary_key
from djimaging.utils.plot_utils import plot_srf


class FitGauss2DRFTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = """
        -> self.split_rf_table
        ---
        srf_fit: mediumblob
        srf_params: blob
        rf_area_um2: float # Area covered by 2 standard deviations
        rf_cdia_um: float # Circle equivalent diameter
        center_index: float # Weight and sign of center in sRF
        surround_index: float # Weight and sign of surround in sRF
        rf_qidx: float # Quality index as explained variance of the sRF estimation between 0 and 1
        """
        return definition

    _polarity = None

    @property
    def key_source(self):
        try:
            return self.split_rf_table.proj()
        except (AttributeError, TypeError):
            pass

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

        # Fit RF model
        srf_fit, srf_params, qi = fit_rf_model(srf, kind='gauss', polarity=self._polarity)

        # Compute properties
        area, diameter = compute_gauss_srf_area(srf_params, stim_dict["pix_scale_x_um"], stim_dict["pix_scale_y_um"])
        center_index = compute_center_index(srf=srf, srf_center=srf_fit)
        surround_index = compute_surround_index(srf=srf, srf_center=srf_fit)

        # Save
        fit_key = deepcopy(key)
        fit_key['srf_fit'] = srf_fit.astype(np.float32)
        fit_key['srf_params'] = srf_params
        fit_key['rf_cdia_um'] = diameter
        fit_key['rf_area_um2'] = area
        fit_key['center_index'] = center_index
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

    def plot(self, by=("preprocess_id",)):
        columns = ["rf_qidx", "rf_cdia_um", "center_index", "surround_index"]
        df = self.proj(*columns).fetch(format='frame').reset_index()
        for params, group in df.groupby(list(by)):
            print(by)
            print(params)
            group.hist(column=columns)
            plt.show()


class FitDoG2DRFTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = """
        -> self.split_rf_table
        ---
        srf_fit: mediumblob
        srf_center_fit: mediumblob
        srf_surround_fit: mediumblob
        srf_params: blob
        srf_eff_center: mediumblob
        srf_eff_center_params: blob
        rf_qidx: float
        rf_area_um2: float # Area covered by 2 standard deviations
        rf_cdia_um: float # Circle equivalent diameter
        center_index: float # Weight and sign of center in sRF
        surround_index: float # Weight and sign of surround in sRF
        """
        return definition

    _polarity = None

    @property
    def key_source(self):
        try:
            return self.split_rf_table.proj()
        except (AttributeError, TypeError):
            pass

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

        srf_fit, srf_center_fit, srf_surround_fit, srf_params, eff_polarity, qi = fit_rf_model(
            srf, kind='dog', polarity=self._polarity)

        if (self._polarity is None) or (eff_polarity == self._polarity):
            srf_eff_center = eff_polarity * np.clip(eff_polarity * srf_fit, 0, None)
        else:
            srf_eff_center = np.zeros_like(srf_fit)

        if np.allclose(srf_eff_center, 0.):
            warnings.warn(f'Failed to fit DoG for key={key}')
            return

        center_index = compute_center_index(srf=srf, srf_center=srf_eff_center)
        surround_index = compute_surround_index(srf=srf, srf_center=srf_eff_center)

        # Compute area from gaussian fit to effective center
        srf_gauss_params = fit_rf_model(srf_eff_center, kind='gauss', polarity=eff_polarity,
                                        center=(srf_params['x_mean_0'], srf_params['y_mean_0']))[1]

        if srf_gauss_params is None:
            warnings.warn(f'Failed to fit DoG center for key={key}')
            return

        area, diameter = compute_gauss_srf_area(
            srf_gauss_params, stim_dict["pix_scale_x_um"], stim_dict["pix_scale_y_um"])

        # Save
        fit_key = deepcopy(key)
        fit_key['srf_fit'] = srf_fit.astype(np.float32)
        fit_key['srf_center_fit'] = srf_center_fit.astype(np.float32)
        fit_key['srf_surround_fit'] = srf_surround_fit.astype(np.float32)
        fit_key['srf_params'] = srf_params
        fit_key['srf_eff_center'] = srf_eff_center.astype(np.float32)
        fit_key['srf_eff_center_params'] = srf_gauss_params
        fit_key['rf_cdia_um'] = diameter
        fit_key['rf_area_um2'] = area
        fit_key['center_index'] = center_index
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

    def plot(self, by=("preprocess_id",)):
        columns = ["rf_qidx", "rf_cdia_um", "center_index", "surround_index"]
        df = self.proj(*columns).fetch(format='frame').reset_index()
        for params, group in df.groupby(list(by)):
            print(by)
            print(params)
            group.hist(column=columns)
            plt.show()
