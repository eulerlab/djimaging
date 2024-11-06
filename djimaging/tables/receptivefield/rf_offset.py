"""
Compute RF offset w.r.t. stimulus center.

Example usage:
@schema
class RfOffset(receptivefield.RfOffsetTemplate):
    stimulus_tab = Stimulus
    rf_split_tab = SplitRF
    rf_fit_tab = FitGauss2DRF

@schema
class RfRoiOffset(receptivefield.RfRoiOffsetTemplate):
    rf_offset_tab = RfOffset
    roi_pos_wrt_field_tab = RelativeRoiLocationWrtField
    userinfo_tab = UserInfo
    experiment_tab = Experiment
    stimulus_tab = Stimulus
    pres_tab = Presentation
    roimask_tab = RoiMask.RoiMaskPresentation
    rf_fit_tab = FitGauss2DRF
"""

import warnings
from abc import abstractmethod

import datajoint as dj
import numpy as np
from matplotlib import pyplot as plt
from sympy.physics.vector.tests.test_printing import alpha

from djimaging.tables.receptivefield import FitGauss2DRFTemplate, FitDoG2DRFTemplate, RfContoursTemplate
from djimaging.utils.receptive_fields.srf_contour_utils import compute_cntr_center


class RfOffsetTemplate(dj.Computed):
    database = ''

    @property
    def definition(self):
        definition = """
        # Computes distance to center of stimulus
        -> self.rf_fit_tab
        ---
        rf_dx_um: float  # Offset in x direction in microns (x as in the field coordinates)
        rf_dy_um: float  # Offset in x direction in microns (y as in the field coordinates)
        rf_d_um: float  # Euclidean distance to the center in microns
        """
        return definition

    @property
    def key_source(self):
        try:
            return self.rf_fit_tab.proj()
        except (AttributeError, TypeError):
            pass

    @property
    @abstractmethod
    def stimulus_tab(self):
        pass

    @property
    @abstractmethod
    def rf_split_tab(self):
        pass

    @property
    @abstractmethod
    def rf_fit_tab(self):
        pass

    def fetch_and_compute(self, key):
        """Compute offset w.r.t. stimulus center in microns"""
        stim_dict = (self.stimulus_tab & key).fetch1('stim_dict')

        if ('pix_scale_x_um' not in stim_dict) or ('pix_scale_x_um' not in stim_dict):
            raise ValueError("Pixel size not found in stimulus dict")

        pix_scale_x_um, pix_scale_y_um = stim_dict['pix_scale_x_um'], stim_dict['pix_scale_y_um']

        if isinstance(self.rf_fit_tab(), RfContoursTemplate):
            srf_contours = (self.rf_fit_tab & key).fetch1('srf_contours')
            levels = (self.rf_fit_tab.rf_contours_params_table & key).fetch1('levels')
            if len(srf_contours) == 0:
                rf_dx_um, rf_dy_um = np.nan, np.nan
            elif len(srf_contours[levels[0]]) == 0:
                rf_dx_um, rf_dy_um = np.nan, np.nan
            else:
                rf_dx_um, rf_dy_um = compute_cntr_center(srf_contours[levels[0]][0])

        elif isinstance(self.rf_fit_tab(), (FitDoG2DRFTemplate, FitGauss2DRFTemplate)):
            srf = (self.rf_split_tab & key).fetch1('srf')
            if isinstance(self.rf_fit_tab(), FitDoG2DRFTemplate):
                srf_params = (self.rf_fit_tab & key).fetch1('srf_eff_center_params')
            else:
                srf_params = (self.rf_fit_tab & key).fetch1('srf_params')

            # Compute offset w.r.t. stimulus center.
            # Plus one half to get pixel center, e.g. if the RF is centered on bottom left pixel,
            # the fit will be 0, 0. For a 2x2 stimulus the offset is half a pixel: 0.5 - 2 / 2 = -0.5
            rf_dx_um = ((srf_params['x_mean'] + 0.5) - (srf.shape[1] / 2)) * pix_scale_x_um
            rf_dy_um = ((srf_params['y_mean'] + 0.5) - (srf.shape[0] / 2)) * pix_scale_y_um
        else:
            raise NotImplementedError

        stimulus_offset_dx_um = stim_dict.get('offset_x_um', 0)
        stimulus_offset_dy_um = stim_dict.get('offset_y_um', 0)

        rf_dx_um += stimulus_offset_dx_um
        rf_dy_um += stimulus_offset_dy_um

        return rf_dx_um, rf_dy_um

    def make(self, key):
        # Position of sRF center in stimulus coordinates
        rf_dx_um, rf_dy_um = self.fetch_and_compute(key)

        if not np.isfinite(rf_dx_um) or not np.isfinite(rf_dy_um):
            return

        rf_d_um = (rf_dx_um ** 2 + rf_dy_um ** 2) ** 0.5
        self.insert1(dict(**key, rf_dx_um=rf_dx_um, rf_dy_um=rf_dy_um, rf_d_um=rf_d_um))

    def plot(self, exp_key):
        rf_dx_um, rf_dy_um = (self & exp_key).fetch('rf_dx_um', 'rf_dy_um')

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

        for i, (rf_dx_um_i, rf_dy_um_i) in enumerate(zip(rf_dx_um, rf_dy_um)):
            ax.plot([0, rf_dx_um_i], [0, rf_dy_um_i], '-', c=f'C{i % 10}', alpha=0.4)

        return fig, ax


class RfRoiOffsetTemplate(dj.Computed):
    database = ''

    @property
    def definition(self):
        definition = """
        -> self.rf_offset_tab
        -> self.roi_pos_wrt_field_tab
        ---
        rf_roi_dx_um: float
        rf_roi_dy_um: float
        """
        return definition

    @property
    def key_source(self):
        try:
            return self.rf_offset_tab.proj() * self.roi_pos_wrt_field_tab.proj()
        except (AttributeError, TypeError):
            pass

    @property
    @abstractmethod
    def rf_offset_tab(self):
        pass

    @property
    @abstractmethod
    def roi_pos_wrt_field_tab(self):
        pass

    @property
    @abstractmethod
    def userinfo_tab(self):
        pass

    @property
    @abstractmethod
    def experiment_tab(self):
        pass

    @property
    @abstractmethod
    def stimulus_tab(self):
        pass

    @property
    @abstractmethod
    def pres_tab(self):
        pass

    @property
    @abstractmethod
    def roimask_tab(self):
        pass

    @property
    @abstractmethod
    def rf_fit_tab(self):
        pass

    def make(self, key):
        rf_dx_um, rf_dy_um = (self.rf_offset_tab & key).fetch1('rf_dx_um', 'rf_dy_um')
        relx_wrt_field, rely_wrt_field = (self.roi_pos_wrt_field_tab & key).fetch1(
            'relx_wrt_field', 'rely_wrt_field')

        setupid = (self.experiment_tab.ExpInfo & key).fetch1('setupid')

        if setupid == "1":
            rf_roi_dx_um = relx_wrt_field - rf_dy_um
            rf_roi_dy_um = rely_wrt_field + rf_dx_um
        elif setupid == "3":
            rf_roi_dx_um = relx_wrt_field - rf_dy_um
            rf_roi_dy_um = rely_wrt_field + rf_dx_um
        else:
            raise NotImplementedError(setupid)

        self.insert1(dict(**key, rf_roi_dx_um=rf_roi_dx_um, rf_roi_dy_um=rf_roi_dy_um))

    def plot(self, restriction=None):
        import seaborn as sns

        if restriction is None:
            restriction = {}

        setupid = (self.experiment_tab().ExpInfo & restriction).fetch('setupid')

        if np.unique(setupid).size == 1:
            setupid = setupid[0]
        else:
            raise NotImplementedError(f"Multiple setupids found: {np.unique(setupid)}")

        rf_dx_um, rf_dy_um, relx_wrt_field, rely_wrt_field, rf_roi_dx_um, rf_roi_dy_um = (
                self * self.rf_offset_tab() * self.roi_pos_wrt_field_tab() & restriction).fetch(
            'rf_dx_um', 'rf_dy_um', 'relx_wrt_field', 'rely_wrt_field', 'rf_roi_dx_um', 'rf_roi_dy_um')

        if setupid == "1":
            rf_relx = -rf_dy_um
            rf_rely = rf_dx_um
        elif setupid == "3":
            rf_relx = -rf_dy_um
            rf_rely = rf_dx_um
        else:
            raise NotImplementedError(setupid)

        fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

        scatter_kws = {'s': 1, "alpha": 0.5, "color": 'gray', 'zorder': -100}

        ax = axs[0]
        sns.regplot(ax=ax, x=relx_wrt_field, y=rf_relx, scatter_kws=scatter_kws)
        ax.set_xlabel('relx_wrt_field')
        ax.set_ylabel('rf_relx')

        ax = axs[1]
        sns.regplot(ax=ax, x=rely_wrt_field, y=rf_rely, scatter_kws=scatter_kws)
        ax.set_xlabel('rely_wrt_field')
        ax.set_ylabel('rf_rely')

        ax = axs[2]
        sns.regplot(ax=ax, x=rf_roi_dx_um, y=rf_roi_dy_um, scatter_kws=scatter_kws)
        ax.set_xlabel('rf_roi_dx_um')
        ax.set_ylabel('rf_roi_dy_um')

        for ax in axs:
            ax.grid()
            ax.set_aspect('equal')

        return fig, ax

    def plot_field(self, key, rf_qidx_thresh=0.45, n_std=1, gamma=0.7):
        from matplotlib.patches import Ellipse
        import seaborn as sns

        roi_mask = (self.roimask_tab & key).fetch1("roi_mask")
        pixel_size_um, npixartifact = (self.pres_tab & key).fetch('pixel_size_um', 'npixartifact')

        data_name, alt_name = (self.userinfo_tab & key).fetch1('data_stack_name', 'alt_stack_name')
        main_ch_average = (self.pres_tab.StackAverages & key & f'ch_name="{data_name}"').fetch1('ch_average')
        stim_dict = (self.stimulus_tab & key).fetch1('stim_dict')
        pix_scale_x_um, pix_scale_y_um = stim_dict['pix_scale_x_um'], stim_dict['pix_scale_y_um']

        extent = np.array([-main_ch_average.shape[0] / 2, main_ch_average.shape[0] / 2,
                           -main_ch_average.shape[1] / 2, main_ch_average.shape[1] / 2]) * pixel_size_um

        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.imshow(main_ch_average.T ** gamma, extent=extent, origin='lower', cmap='gray')

        _rois = roi_mask.copy().astype(float)
        _rois[_rois >= 0] = 0
        _rois[_rois < 0] = -1
        ax.contour(-_rois.T, cmap='Reds', origin='lower', extent=extent, levels=[0.99], vmin=0, vmax=1)

        roi_keys = (self.rf_fit_tab & f"rf_qidx>{rf_qidx_thresh}" & key).fetch('KEY')
        colors = sns.color_palette('Spectral', len(roi_keys))

        shift_dx, shift_dy = (self.roimask_tab & key).fetch1('shift_dx', 'shift_dy')

        for i, roi_key in enumerate(roi_keys):
            rf_dx_um, rf_dy_um = (self.rf_offset_tab & roi_key).fetch1('rf_dx_um', 'rf_dy_um')
            relx_wrt_field, rely_wrt_field = (self.roi_pos_wrt_field_tab & roi_key).fetch1(
                'relx_wrt_field', 'rely_wrt_field')

            srf_params = (self.rf_fit_tab & roi_key).fetch1("srf_params")

            rf_xy = rf_dx_um, -rf_dy_um
            roi_xy = rely_wrt_field - shift_dx, relx_wrt_field - shift_dy

            ax.plot(*roi_xy, 'X', zorder=100, ms=3, c=colors[i])
            ax.plot([roi_xy[0], rf_xy[0]], [roi_xy[1], rf_xy[1]], '-', zorder=100, ms=3, c=colors[i])

            ax.add_patch(Ellipse(
                xy=rf_xy,
                width=n_std * 2 * srf_params['x_stddev'] * pix_scale_x_um,
                height=n_std * 2 * srf_params['y_stddev'] * pix_scale_y_um,
                angle=np.rad2deg(srf_params['theta']), color=colors[i], fill=False, alpha=0.5))

        ax.set_xlabel('µm')
        ax.set_ylabel('µm')
