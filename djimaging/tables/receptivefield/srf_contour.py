"""
Estimate contours of receptive fields and compute metrics

Example usage:

from djimaging.tables import receptivefield

@schema
class RfContoursParams(receptivefield.RfContoursParamsTemplate):
    pass


@schema
class RfContours(receptivefield.RfContoursTemplate):
    split_rf_table = SplitRF
    rf_contours_params_table = RfContoursParams
    stimulus_table = Stimulus


@schema
class RfContourMetrics(receptivefield.RfContourMetricsTemplate):
    _d_inner = 20  # Border around sRF center in um
    _d_outer = 120  # Border around sRF center in um

    rf_contour_table = RfContours
"""

from abc import abstractmethod

import datajoint as dj
import numpy as np
from matplotlib import pyplot as plt

from djimaging.utils.dj_utils import get_primary_key
from djimaging.utils.math_utils import normalize
from djimaging.utils.plot_utils import plot_srf, set_long_title
from djimaging.utils.receptive_fields.spatial_rf_utils import compute_center_index, compute_surround_index
from djimaging.utils.receptive_fields.split_rf_utils import smooth_rf, resize_srf
from djimaging.utils.receptive_fields.srf_contour_utils import (
    compute_contour, compute_irregular_index, get_center_and_surround_masks, plot_center_and_surround_masks)


class RfContoursParamsTemplate(dj.Lookup):
    database = ""

    @property
    def definition(self):
        definition = """
        rf_contours_params_id: int # unique param set id
        ---
        blur_std : float
        blur_npix : int unsigned
        upsample_srf_scale : int unsigned
        norm_kind : enum('zscore', 'zero_one', 'amp_one', 'std_one', 'none')
        levels : blob  # First level is used for area and cdia
        """
        return definition

    def add_default(self, skip_duplicates=False, **params):
        """Add default preprocess parameter to table"""
        key = dict(
            rf_contours_params_id=1,
            blur_std=0,
            blur_npix=0,
            upsample_srf_scale=5,
            norm_kind='amp_one',
            levels=(0.6, 0.65, 0.7),
        )
        key.update(**params)
        self.insert1(key, skip_duplicates=skip_duplicates)


class RfContoursTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = """
        -> self.split_rf_table
        -> self.rf_contours_params_table
        ---
        srf_contours : blob
        srf_contours_um2 : blob
        srf_contours_cdia_um : blob
        is_single_contour : tinyint unsigned # True if all levels have a single contour
        largest_contour_ratio : float # Ratio of largest contour relative to full area (for level where it's lowest)
        rf_area_um2 = NULL : float # Area of largest contour at first level
        rf_cdia_um = NULL : float # Circle equivalent diameter of largest contour at first level
        irregular_index = NULL : float # Irregularity index of largest contour at first level
        """
        return definition

    @property
    @abstractmethod
    def split_rf_table(self):
        pass

    @property
    @abstractmethod
    def rf_contours_params_table(self):
        pass

    @property
    @abstractmethod
    def stimulus_table(self):
        pass

    @property
    def key_source(self):
        try:
            return self.split_rf_table.proj() * self.rf_contours_params_table.proj()
        except (AttributeError, TypeError):
            pass

    def fetch1_pixel_size(self, key):
        stim_dict = (self.stimulus_table() & key).fetch1("stim_dict")
        stim_pixel_size_x_um = stim_dict["pix_scale_x_um"]
        stim_pixel_size_y_um = stim_dict["pix_scale_y_um"]
        upsample_srf_scale = (self.split_rf_table.split_rf_params_table() & key).fetch1('upsample_srf_scale')
        pixel_size_x_um = stim_pixel_size_x_um / upsample_srf_scale if upsample_srf_scale > 1 else stim_pixel_size_x_um
        pixel_size_y_um = stim_pixel_size_y_um / upsample_srf_scale if upsample_srf_scale > 1 else stim_pixel_size_y_um

        return pixel_size_x_um, pixel_size_y_um

    def _fetch_and_compute_contours(self, key, plot=False):
        norm_kind, blur_std, blur_npix, upsample_srf_scale, levels = (
                self.rf_contours_params_table() & key).fetch1(
            "norm_kind", "blur_std", "blur_npix", "upsample_srf_scale", "levels")
        srf = (self.split_rf_table() & key).fetch1("srf")

        pixel_size_x_um, pixel_size_y_um = self.fetch1_pixel_size(key)
        if not np.isclose(pixel_size_x_um, pixel_size_y_um):
            raise ValueError("Pixel size is not isotropic")

        polarity = 1 if np.abs(np.max(srf)) >= np.abs(np.min(srf)) else -1

        srf_contours, srf_contours_um2, srf_contours_cdia_um = _compute_contours(
            srf, pixel_size_x_um, norm_kind, blur_std, blur_npix, upsample_srf_scale,
            [lvl * polarity for lvl in levels], plot=plot, title=str(key))

        return srf_contours, srf_contours_um2, srf_contours_cdia_um

    def make(self, key, plot=False, verbose=False):
        srf_contours, srf_contours_um2, srf_contours_cdia_um = self._fetch_and_compute_contours(key, plot=plot)

        levels = list(srf_contours.keys())
        level0 = levels[0]

        is_single_contour = np.all([len(srf_contours[level]) == 1 for level in levels])
        no_contours = np.any([len(srf_contours[level]) == 0 for level in levels])
        largest_contour_ratio = 0. if no_contours else np.min([
            srf_contours_um2[level][0] / np.sum(srf_contours_um2[level]) for level in levels])

        if verbose:
            print(f"levels={levels}")
            print(f"is_single_contour={is_single_contour}")
            print(f"no_contours={no_contours}")
            print(f"largest_contour_ratio={largest_contour_ratio}")

        self.insert1(dict(
            **key,
            srf_contours=srf_contours,
            srf_contours_um2=srf_contours_um2,
            srf_contours_cdia_um=srf_contours_cdia_um,
            is_single_contour=int(is_single_contour),
            largest_contour_ratio=largest_contour_ratio,
            rf_area_um2=np.nan if no_contours else srf_contours_um2[level0][0],
            rf_cdia_um=np.nan if no_contours else srf_contours_cdia_um[level0][0],
            irregular_index=np.nan if no_contours else compute_irregular_index(srf_contours[level0][0]),
        ))

    def plot1(self, key=None):
        key = get_primary_key(self, key)

        srf_contours, srf_contours_um2, srf_contours_cdia_um = (self & key).fetch1(
            "srf_contours", "srf_contours_um2", "srf_contours_cdia_um")

        srf_contours_, srf_contours_um2_, srf_contours_cdia_um_ = self._fetch_and_compute_contours(key, plot=True)

        levels = list(srf_contours.keys())

        for level in levels:
            if not np.allclose(srf_contours_um2_[level], srf_contours_um2[level]):
                raise ValueError("srf_contours do not match. Plots are likely incorrect.")


def _compute_contours(srf, pixel_size, norm_kind, blur_std, blur_npix, upsample_srf_scale, levels,
                      plot=False, title=None):
    srf_raw = srf.copy()
    if blur_npix > 0:
        srf = smooth_rf(rf=srf, blur_std=blur_std, blur_npix=blur_npix)

    if upsample_srf_scale > 1:
        srf = resize_srf(srf, scale=upsample_srf_scale)

    if norm_kind != 'none':
        srf = normalize(srf, norm_kind=norm_kind)

    srf_contours = dict()
    srf_contours_um2 = dict()
    srf_contours_cdia_um = dict()

    if plot:
        fig, axs = plt.subplots(1, len(levels) + 2, figsize=(15, 4))

        if title is not None:
            set_long_title(fig=fig, title=title, y=1, va='bottom')

        plot_srf(srf_raw, ax=axs[0], pixelsize=pixel_size)
        plot_srf(srf, ax=axs[1], pixelsize=pixel_size / upsample_srf_scale)

    for i, level in enumerate(levels):
        cntrs, cntrs_size = compute_contour(
            srf, level, pixel_size_um=pixel_size / upsample_srf_scale, plot=plot, ax=axs[i + 2] if plot else None)

        # Sort by size, largest first
        sort_index = np.argsort(cntrs_size)[::-1]

        cntrs = [cntrs[i] for i in sort_index]
        cntrs_size = [cntrs_size[i] for i in sort_index]

        srf_contours[level] = cntrs
        srf_contours_um2[level] = cntrs_size
        srf_contours_cdia_um[level] = [np.sqrt(cntr_size / np.pi) * 2 for cntr_size in cntrs_size]

    if plot:
        plt.tight_layout()
        plt.show()

    return srf_contours, srf_contours_um2, srf_contours_cdia_um


class RfContourMetricsTemplate(dj.Computed):
    database = ""
    _d_inner = 20  # Border around sRF center in um
    _d_outer = 120  # Border around sRF center in um

    @property
    def definition(self):
        definition = """
        -> self.rf_contour_table
        ---
        center_index: float # Weight and sign of center in sRF
        surround_index: float # Weight and sign of surround in sRF (with space to center)
        full_surround_index : float # Weight and sign of surround in sRF (with no border to center)
        """
        return definition

    @property
    @abstractmethod
    def rf_contour_table(self):
        pass

    @property
    def key_source(self):
        try:
            return (self.rf_contour_table & "largest_contour_ratio>0").proj()
        except (AttributeError, TypeError):
            pass

    def fetch1_pixel_size(self, key):
        pixel_size_x_um, pixel_size_y_um = self.rf_contour_table().fetch1_pixel_size(key)
        upsample_srf_scale = (self.rf_contour_table.rf_contours_params_table() & key).fetch1("upsample_srf_scale")

        pixel_size_x_um = pixel_size_x_um / upsample_srf_scale if upsample_srf_scale > 1 else pixel_size_x_um
        pixel_size_y_um = pixel_size_y_um / upsample_srf_scale if upsample_srf_scale > 1 else pixel_size_y_um

        return pixel_size_x_um, pixel_size_y_um

    def _fetch_and_compute(self, key, plot=False):
        srf_contours, srf_contours_um2, srf_contours_cdia_um = (self.rf_contour_table & key).fetch1(
            "srf_contours", "srf_contours_um2", "srf_contours_cdia_um")

        upsample_srf_scale, levels = (self.rf_contour_table.rf_contours_params_table() & key).fetch1(
            "upsample_srf_scale", "levels")

        srf = (self.rf_contour_table.split_rf_table() & key).fetch1("srf")

        if upsample_srf_scale > 1:
            srf = resize_srf(srf, scale=upsample_srf_scale)

        pixel_size_x_um, pixel_size_y_um = self.fetch1_pixel_size(key)
        if not np.isclose(pixel_size_x_um, pixel_size_y_um):
            raise ValueError("Pixel size is not isotropic")

        srf_contour = srf_contours[levels[0]][0]

        mask_center, mask_surround, mask_full_surround = get_center_and_surround_masks(
            cntr=srf_contour, srf_shape=srf.shape, pixelsize=pixel_size_x_um,
            d_inner=self._d_inner, d_outer=self._d_outer)

        srf_center = srf.copy()
        srf_center[~mask_center] = 0
        center_index = compute_center_index(srf, srf_center=srf_center)

        srf_surround = srf.copy()
        srf_surround[~mask_surround] = 0
        surround_index = compute_surround_index(srf, srf_surround=srf_surround)

        srf_full_surround = srf.copy()
        srf_full_surround[~mask_full_surround] = 0
        full_surround_index = compute_surround_index(srf, srf_surround=srf_full_surround)

        if plot:
            fig, axs = plt.subplots(1, 5, figsize=(12, 3), squeeze=True)

            set_long_title(fig=fig, title=str(key))

            plot_srf(ax=axs[0], srf=srf, pixelsize=pixel_size_x_um)
            plot_center_and_surround_masks(
                mask_center, mask_surround, mask_full_surround, pixelsize=pixel_size_x_um, axs=axs[1:5])

            axs[1].set(xlabel=f"center_index={center_index:.2f}")
            axs[2].set(xlabel=f"full_surround_index={full_surround_index:.2f}")
            axs[3].set(xlabel=f"surround_index={surround_index:.2f}")

            plt.tight_layout()
            plt.show()

        return center_index, surround_index, full_surround_index

    def make(self, key, plot=False):
        center_index, surround_index, full_surround_index = self._fetch_and_compute(key, plot=plot)

        self.insert1(dict(
            **key,
            center_index=center_index,
            surround_index=surround_index,
            full_surround_index=full_surround_index,
        ))

    def plot1(self, key=None):
        key = get_primary_key(self, key)

        center_index, surround_index, full_surround_index = (self & key).fetch1(
            "center_index", "surround_index", "full_surround_index")

        center_index_, surround_index_, full_surround_index_ = self._fetch_and_compute(key, plot=True)

        if not np.isclose(center_index_, center_index) or \
                not np.isclose(surround_index_, surround_index) or \
                not np.isclose(full_surround_index_, full_surround_index):
            raise ValueError("indexes do not match. Plots are likely incorrect.")
