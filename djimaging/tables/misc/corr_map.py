"""
Table for Correlation Mask.

Example usage:

from djimaging.tables import misc

@schema
class CorrMap(misc.CorrMapTemplate):
    userinfo_table = UserInfo
    presentation_table = Presentation
    raw_params_table = RawDataParams
"""
from abc import abstractmethod

import datajoint as dj
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from djimaging.autorois.autoshift_utils import shift_img
from djimaging.autorois.corr_roi_mask_utils import stack_corr_image
from djimaging.utils.dj_utils import get_primary_key
from djimaging.utils.math_utils import normalize_zscore
from djimaging.utils.plot_utils import prep_long_title
from djimaging.utils.scanm import read_utils


class CorrMapTemplate(dj.Computed):
    database = ""
    _cut_x = (1, 1)
    _cut_z = (1, 1)
    _include_prestim = False

    def __init__(self):
        super().__init__()
        if "corr_map_pre_stim" in self.definition:
            self._include_prestim = True

    @property
    def definition(self):
        definition = """
        -> self.presentation_table
        ---
        corr_map : longblob  # Correlation mask for stack, after stimulus onset
        corr_map_max : float  # Maximum correlation
        corr_map_mean : float  # Mean correlation
        """

        if self._include_prestim:
            definition += """
            corr_map_pre_stim : longblob  # Correlation mask for stack before stimulus
            """

        return definition

    @property
    def key_source(self):
        try:
            return self.presentation_table.proj()
        except (AttributeError, TypeError):
            pass

    @property
    @abstractmethod
    def presentation_table(self):
        pass

    @property
    @abstractmethod
    def userinfo_table(self):
        pass

    @property
    @abstractmethod
    def raw_params_table(self):
        pass

    def get_cut_x(self):
        return self._cut_x

    def get_cut_z(self):
        return self._cut_z

    def make(self, key):
        filepath = (self.presentation_table & key).fetch1('pres_data_file')
        from_raw_data = (self.raw_params_table & key).fetch1('from_raw_data')
        data_name = (self.userinfo_table & key).fetch1('data_stack_name')

        triggertimes = (self.presentation_table & key).fetch1('triggertimes')
        fs = (self.presentation_table.ScanInfo & key).fetch1('scan_frequency')
        stack = read_utils.load_stacks(filepath, from_raw_data, ch_names=(data_name,))[0][data_name]

        if len(triggertimes) > 0:
            idx_start = int(triggertimes[0] * fs)
            idx_end = int(triggertimes[-1] * fs)
        else:
            idx_start = 0
            idx_end = stack.shape[2]

        corr_map = stack_corr_image(stack[:, :, idx_start:idx_end], cut_x=self._cut_x, cut_z=self._cut_z[::-1])

        entry = dict(key, corr_map=corr_map, corr_map_max=np.max(corr_map), corr_map_mean=np.mean(corr_map))

        if self._include_prestim:
            if idx_start < 2:
                corr_map_pre_stim = np.full_like(corr_map, np.nan)
            else:
                corr_map_pre_stim = stack_corr_image(
                    stack[:, :, :idx_start], cut_x=self._cut_x, cut_z=self._cut_z[::-1])
            self.insert1({**entry, 'corr_map_pre_stim': corr_map_pre_stim})
        else:
            self.insert1(entry)

    def plot1(self, key=None, gamma=0.7):
        key = get_primary_key(self, key=key)

        data_name = (self.userinfo_table & key).fetch1('data_stack_name')
        main_ch_average = (self.presentation_table.StackAverages & key & f'ch_name="{data_name}"').fetch1('ch_average')

        corr_map = (self & key).fetch1('corr_map')
        if self._include_prestim:
            corr_map_pre_stim = (self & key).fetch1('corr_map_pre_stim')

        vabsmax = np.max(np.abs(corr_map))

        extent = (0, main_ch_average.shape[0], 0, main_ch_average.shape[1])

        from matplotlib import pyplot as plt

        fig, axs = plt.subplots(1, 4, figsize=(16, 3))
        fig.suptitle(prep_long_title(key))

        for ax in axs:
            ax.grid(True)

        for ax in [axs[0], axs[3]]:
            im = ax.imshow(main_ch_average.T ** gamma, cmap='viridis', origin='lower', extent=extent)
            plt.colorbar(im, ax=ax, label='Brightness')

        for ax in [axs[1]]:
            im = ax.imshow(corr_map.T, vmin=-vabsmax, vmax=+vabsmax, cmap='bwr', origin='lower', extent=extent)
            plt.colorbar(im, ax=ax, label='Correlation')

        if self._include_prestim:
            for ax in [axs[2]]:
                im = ax.imshow(corr_map_pre_stim.T, vmin=-vabsmax, vmax=+vabsmax, cmap='bwr', origin='lower',
                               extent=extent)
                plt.colorbar(im, ax=ax, label='Pre-Stim-Correlation')

        for ax in [axs[1], axs[2], axs[3]]:
            ax.contour(corr_map.T, levels=[np.percentile(corr_map, 90), np.percentile(corr_map, 95)],
                       cmap='Reds', origin='lower', extent=extent)

        plt.tight_layout()
        plt.show()


class CrossCondCorrMapTemplate(dj.Computed):
    database = ""
    _split_cond = 'cond1'
    _ref_cond = 'C1'
    _max_shift = 5

    @property
    def definition(self):
        definition = f"""
        -> self.corr_map_table.proj(cond1_A='{self._split_cond}')
        -> self.corr_map_table.proj(cond1_B='{self._split_cond}')
        ---
        cross_corr_map : blob  # Cross correlation between stacks of same stimulus but different conditions
        shift_x : int  # Shift in pixels
        shift_z : int  # Shift in pixels
        max_corr : float  # Maximum correlation
        """
        return definition

    @property
    def key_source(self):
        try:
            return (
                    (self.corr_map_table & f"{self._split_cond}='{self._ref_cond}'").proj(
                        cond1_A=f'{self._split_cond}') *
                    (self.corr_map_table & f"{self._split_cond}!='{self._ref_cond}'").proj(
                        cond1_B=f'{self._split_cond}')
            )
        except (AttributeError, TypeError):
            pass

    @property
    @abstractmethod
    def corr_map_table(self):
        pass

    def make(self, key, plot=False):
        correlation, x_shift, z_shift, max_corr = self._make_compute(key)
        self.insert1(dict(key, cross_corr_map=correlation, shift_x=x_shift, shift_z=z_shift, max_corr=max_corr))

    def _make_compute(self, key, plot=False):
        key_a = {**key, 'cond1': key[f'{self._split_cond}_A']}
        key_b = {**key, 'cond1': key[f'{self._split_cond}_B']}

        corr_map_a = (self.corr_map_table & key_a).fetch1('corr_map')
        corr_map_b = (self.corr_map_table & key_b).fetch1('corr_map')

        cut_x = self.corr_map_table().get_cut_x()
        cut_z = self.corr_map_table().get_cut_z()

        image1 = corr_map_a[cut_x[0]:-cut_x[1], cut_z[1]:-cut_z[0]]
        image2 = corr_map_b[cut_x[0]:-cut_x[1], cut_z[1]:-cut_z[0]]

        correlation, x_shift, z_shift, max_corr = cross_correlate_images(
            image1, image2, plot=plot, max_shift=self._max_shift)

        return correlation, x_shift, z_shift, max_corr

    def plot1(self, key=None):
        key = get_primary_key(self, key=key)
        self._make_compute(key, plot=True)


class CrossStimCorrMapTemplate(dj.Computed):
    database = ""
    _ref_stim = 'noise'
    _max_shift = 5

    @property
    def definition(self):
        definition = f"""
        -> self.corr_map_table.proj(stim_A='stim_name')
        -> self.corr_map_table.proj(stim_B='stim_name')
        ---
        cross_corr_map : blob  # Cross correlation between stacks of same stimulus but different conditions
        shift_x : int  # Shift in pixels
        shift_z : int  # Shift in pixels
        max_corr : float  # Maximum correlation
        """
        return definition

    @property
    def key_source(self):
        try:
            return (
                    (self.corr_map_table & f"stim_name='{self._ref_stim}'").proj(stim_A='stim_name') *
                    (self.corr_map_table & f"stim_name!='{self._ref_stim}'").proj(stim_B='stim_name')
            )
        except (AttributeError, TypeError):
            pass

    @property
    @abstractmethod
    def corr_map_table(self):
        pass

    def make(self, key, plot=False):
        correlation, x_shift, z_shift, max_corr = self._make_compute(key)
        self.insert1(dict(key, cross_corr_map=correlation, shift_x=x_shift, shift_z=z_shift, max_corr=max_corr))

    def _make_compute(self, key, plot=False):
        key_a = {**key, 'stim_name': key['stim_A']}
        key_b = {**key, 'stim_name': key['stim_B']}

        corr_map_a = (self.corr_map_table & key_a).fetch1('corr_map')
        corr_map_b = (self.corr_map_table & key_b).fetch1('corr_map')

        cut_x = self.corr_map_table().get_cut_x()
        cut_z = self.corr_map_table().get_cut_z()

        image1 = corr_map_a[cut_x[0]:-cut_x[1], cut_z[1]:-cut_z[0]]
        image2 = corr_map_b[cut_x[0]:-cut_x[1], cut_z[1]:-cut_z[0]]

        correlation, x_shift, z_shift, max_corr = cross_correlate_images(
            image1, image2, plot=plot, max_shift=self._max_shift)

        return correlation, x_shift, z_shift, max_corr

    def plot1(self, key=None):
        key = get_primary_key(self, key=key)
        self._make_compute(key, plot=True)


def normalized_cross_correlation(image1, image2):
    image1_norm = normalize_zscore(image1)
    image2_norm = normalize_zscore(image2)

    correlation = signal.correlate2d(image1_norm, image2_norm, mode='same')
    correlation /= np.sqrt(np.sum(image1_norm ** 2) * np.sum(image2_norm ** 2))
    return correlation


def cross_correlate_images(image1, image2, plot=False, max_shift=None):
    # Compute the normalized cross-correlation
    correlation = normalized_cross_correlation(image1, image2)

    if max_shift is not None and max_shift > 0:
        mask_inside = np.zeros(correlation.shape, dtype=bool)
        mask_inside[correlation.shape[0] // 2 - max_shift - 1:correlation.shape[0] // 2 + max_shift,
        correlation.shape[1] // 2 - max_shift - 1:correlation.shape[1] // 2 + max_shift] = True
        correlation[~mask_inside] = 0

    # Find the peak in the correlation
    x, z = np.unravel_index(np.argmax(correlation), correlation.shape)

    # Calculate the shift
    x_shift = x - image1.shape[0] // 2 + 1
    z_shift = z - image1.shape[1] // 2 + 1

    max_corr = np.max(correlation)

    # Visualize the correlation
    if plot:
        fig, axs = plt.subplots(1, 4, figsize=(16, 4), sharex=True, sharey=True)

        for ax in axs:
            ax.grid(True)

        ax = axs[0]
        im = ax.imshow(correlation.T, cmap='viridis', origin='lower')
        ax.plot(x, z, 'rx')
        plt.colorbar(im)
        ax.set_title(f'Norm. Cross-corr:\nshift=({x_shift}, {z_shift})')

        ax = axs[1]
        im = ax.imshow(image1.T, cmap='viridis', origin='lower')
        plt.colorbar(im)
        ax.set_title(f'Original image1')

        ax = axs[2]
        im = ax.imshow(shift_img(image2, x_shift, z_shift, cval=np.nan).T, cmap='viridis', origin='lower')
        plt.colorbar(im)
        ax.set_title(f'Shifted image2')

        ax = axs[3]
        im = ax.imshow(image2.T, cmap='viridis', origin='lower')
        plt.colorbar(im)
        ax.set_title(f'Original image2')

        for ax in axs:
            ax.axhline(image1.shape[1] // 2, color='r', linestyle='--')
            ax.axvline(image1.shape[0] // 2, color='r', linestyle='--')

        plt.tight_layout()
        plt.show()

    return correlation, x_shift, z_shift, max_corr
