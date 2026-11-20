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
    experiment_stimulator_tab = ExperimentStimulator
    stimulus_tab = Stimulus
    pres_tab = Presentation
    roimask_tab = RoiMask.RoiMaskPresentation  # motion mode: MotionRoiMask
    rf_fit_tab = FitGauss2DRF

`roimask_tab` provides the presentation ROI mask (`roi_mask`, Igor format) and its integer shift relative to the field
mask (`shift_dx`, `shift_dy`). In motion mode (RoiMask without presentation masks) use `MotionRoiMask`, which provides
the same attributes (label mask in raw presentation coordinates and the rounded global shift).

The geometry (how the stimulus array is oriented in the recording field) is taken only from the stimulator of the
experiment (`experiment_stimulator_tab`, see djimaging.tables.stimulator) and from `stim_dict['array_to_screen']` of
the noise stimulus (e.g. `NOISE_ARRAY_TO_SCREEN` in djimaging.utils.stimulator). Stimuli without `array_to_screen`
raise an error.

Known gap (not corrected): the stimulus center is assumed to coincide with the center of the cropped field (the ROI
positions in `relx_wrt_field`/`rely_wrt_field`). If the stimulus is instead centered on the full scan line, the
cropped field center is off along the fast-scan (x) axis, because the crop is asymmetric: it removes
`XPixLineOffs` pixels at the start of each line and `PixRetraceLen` pixels at the end. The stimulus center would then
lie at `(PixRetraceLen - XPixLineOffs) / 2` pixels toward the retrace side of the cropped field center, e.g.
(10 - 6) / 2 = 2 px (~3 um at setup 3, zoom 0.76) for an 80x64 XYScan2 field. There is no y offset (no y retrace).
This estimate also assumes uniform pixel spacing over the full line, including the retrace/flyback, which is
unverified.
"""

from abc import abstractmethod

import datajoint as dj
import numpy as np
from matplotlib import pyplot as plt

from djimaging.tables.receptivefield import FitGauss2DRFTemplate, FitDoG2DRFTemplate, RfContoursTemplate
from djimaging.utils.dj_storage import load_array
from djimaging.utils.receptive_fields.srf_contour_utils import compute_cntr_center
from djimaging.utils.stimulator import array_to_field_matrix, ellipse_to_frame

# Map from the field frame (XCoord, YCoord) to the plot frame of plot_field (x = YCoord, y = -XCoord)
_FIELD_TO_PLOT = np.array([[0, 1], [-1, 0]])


def compute_rf_roi_offset(rf_dx_um: float, rf_dy_um: float, relx_wrt_field: float, rely_wrt_field: float,
                          array_to_screen: dict, geometry: dict) -> tuple[float, float]:
    """Offset of the RF center from the ROI in the field frame (XCoord, YCoord).

    `rf_dx_um`, `rf_dy_um` are the RF center relative to the stimulus center in stimulus array coordinates (RfOffset),
    `geometry` holds the screen axes and offsets of the stimulator (ExperimentStimulator.get_geometry).
    """
    m = array_to_field_matrix(array_to_screen, geometry['screen_x_axis'], geometry['screen_y_axis'])
    rf_relx, rf_rely = m @ np.array([rf_dx_um, rf_dy_um], dtype=float)
    rf_relx += geometry.get('offset_relx_um', 0.)
    rf_rely += geometry.get('offset_rely_um', 0.)
    return float(rf_relx - relx_wrt_field), float(rf_rely - rely_wrt_field)


def fit_affine_roi_to_rf(roi_xy: np.ndarray, rf_xy: np.ndarray, groups: np.ndarray | None = None) -> dict:
    """Least-squares fit of `rf_xy ≈ A @ roi_xy + b_group` in the field frame (XCoord, YCoord).

    Each group (e.g. recording field) gets its own intercept, so `A` only describes how RF positions scale and rotate
    with ROI positions within a field. `A` is split by polar decomposition into `A = R @ S`, with `R` a rotation
    (or reflection) and `S` symmetric positive semi-definite.

    Expected for a correct geometry and unbiased RF centers: rotation 0, scales 1, shear 0, no reflection.
    Not robust to outliers (e.g. failed RF fits at the stimulus border): restrict to good RFs first.

    Parameters
    ----------
    roi_xy, rf_xy : np.ndarray
        Arrays of shape (n, 2) with ROI and RF center positions in µm.
    groups : np.ndarray or None
        Group label per row; None = one group.

    Returns
    -------
    dict
        A (2x2), rotation_deg (counterclockwise from XCoord towards YCoord), reflection (bool),
        scale_x, scale_y (diagonal of S), shear (off-diagonal of S), r2 (of the within-group fit), n, n_groups.
    """
    roi_xy = np.asarray(roi_xy, dtype=float)
    rf_xy = np.asarray(rf_xy, dtype=float)
    if groups is None:
        groups = np.zeros(len(roi_xy), dtype=int)
    groups = np.asarray(groups)

    valid = np.all(np.isfinite(roi_xy), axis=1) & np.all(np.isfinite(rf_xy), axis=1)
    roi_xy, rf_xy, groups = roi_xy[valid], rf_xy[valid], groups[valid]

    # Remove the per-group intercept by centering within each group
    roi_c, rf_c = roi_xy.copy(), rf_xy.copy()
    unique_groups = np.unique(groups)
    for g in unique_groups:
        idx = groups == g
        roi_c[idx] -= roi_xy[idx].mean(axis=0)
        rf_c[idx] -= rf_xy[idx].mean(axis=0)

    n_dof = len(roi_c) - len(unique_groups)
    if n_dof < 3:
        nan = float('nan')
        return dict(A=np.full((2, 2), np.nan), rotation_deg=nan, reflection=False, scale_x=nan, scale_y=nan,
                    shear=nan, r2=nan, n=int(len(roi_c)), n_groups=int(len(unique_groups)))

    # rf_c = roi_c @ A.T
    a_t, *_ = np.linalg.lstsq(roi_c, rf_c, rcond=None)
    A = a_t.T

    ss_res = np.sum((rf_c - roi_c @ A.T) ** 2)
    ss_tot = np.sum(rf_c ** 2)
    r2 = 1. - ss_res / ss_tot if ss_tot > 0 else np.nan

    u, s, vt = np.linalg.svd(A)
    R = u @ vt
    S = vt.T @ np.diag(s) @ vt

    return dict(A=A, rotation_deg=float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))),
                reflection=bool(np.linalg.det(R) < 0),
                scale_x=float(S[0, 0]), scale_y=float(S[1, 1]), shear=float(S[0, 1]),
                r2=float(r2), n=int(len(roi_c)), n_groups=int(len(unique_groups)))


def _format_affine(fit: dict) -> str:
    text = (f"rot = {fit['rotation_deg']:+.1f}°\n"
            f"scale x = {fit['scale_x']:.2f}\n"
            f"scale y = {fit['scale_y']:.2f}\n"
            f"shear = {fit['shear']:+.2f}\n"
            f"R² = {fit['r2']:.2f}\n"
            f"n = {fit['n']} ({fit['n_groups']} fields)")
    if fit['reflection']:
        text += "\nREFLECTION"
    return text


class RfOffsetTemplate(dj.Computed):
    database = ''

    @property
    def definition(self):
        definition = """
        # Computes distance to center of stimulus
        -> self.rf_fit_tab
        ---
        rf_dx_um: float32  # Offset in x direction in microns (x as in the field coordinates)
        rf_dy_um: float32  # Offset in x direction in microns (y as in the field coordinates)
        rf_d_um: float32  # Euclidean distance to the center in microns
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

    def fetch_and_compute(self, key: dict) -> tuple:
        """Compute RF offset with respect to stimulus center in microns.

        Parameters
        ----------
        key : dict
            DataJoint primary key identifying the entry to compute.

        Returns
        -------
        tuple
            Tuple of (rf_dx_um, rf_dy_um) representing the x and y offsets
            from the stimulus center in microns.

        Raises
        ------
        ValueError
            If pixel size is not found in the stimulus dictionary.
        NotImplementedError
            If the rf_fit_tab type is not supported.
        """
        stim_dict = (self.stimulus_tab & key).fetch1('stim_dict')

        if ('pix_scale_x_um' not in stim_dict) or ('pix_scale_y_um' not in stim_dict):
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
            srf = load_array((self.rf_split_tab & key).fetch1('srf'))
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

    def make(self, key: dict) -> None:
        """Compute RF offset and Euclidean distance to stimulus center, then insert.

        Parameters
        ----------
        key : dict
            DataJoint primary key identifying the entry to compute.
        """
        # Position of sRF center in stimulus coordinates
        rf_dx_um, rf_dy_um = self.fetch_and_compute(key)

        if not np.isfinite(rf_dx_um) or not np.isfinite(rf_dy_um):
            return

        rf_d_um = (rf_dx_um ** 2 + rf_dy_um ** 2) ** 0.5
        self.insert1(dict(**key, rf_dx_um=rf_dx_um, rf_dy_um=rf_dy_um, rf_d_um=rf_d_um))

    def plot(self, exp_key: dict) -> tuple:
        """Plot RF offset vectors from the stimulus center for a given experiment.

        Parameters
        ----------
        exp_key : dict
            DataJoint restriction key for the experiment.

        Returns
        -------
        tuple
            Tuple of (fig, ax) matplotlib objects.
        """
        rf_dx_um, rf_dy_um = (self & exp_key).to_arrays('rf_dx_um', 'rf_dy_um')

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
        -> self.experiment_stimulator_tab
        ---
        relx_rf_roi_um: float32  # RF center minus ROI position along XCoord_um
        rely_rf_roi_um: float32  # RF center minus ROI position along YCoord_um
        geometry_confidence: enum('high', 'medium', 'low')  # confidence of the stimulator geometry
        geometry_reason: varchar(1023)  # why the confidence is not high
        """
        return definition

    @property
    def key_source(self):
        try:
            return (self.rf_offset_tab.proj() * self.roi_pos_wrt_field_tab.proj()
                    * self.experiment_stimulator_tab.proj())
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
    def experiment_stimulator_tab(self):
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

    def make(self, key: dict) -> None:
        """Compute the RF-to-ROI relative offset and insert.

        Combines the RF offset from the stimulus center with the ROI position
        within the field to compute the relative displacement between the RF
        center and the ROI position.

        Parameters
        ----------
        key : dict
            DataJoint primary key identifying the entry to compute.

        Raises
        ------
        ValueError
            If the stimulus does not declare `array_to_screen` in its `stim_dict`.
        """
        rf_dx_um, rf_dy_um = (self.rf_offset_tab & key).fetch1('rf_dx_um', 'rf_dy_um')
        relx_wrt_field, rely_wrt_field = (self.roi_pos_wrt_field_tab & key).fetch1(
            'relx_wrt_field', 'rely_wrt_field')
        stim_dict = (self.stimulus_tab & key).fetch1('stim_dict')
        if not stim_dict or 'array_to_screen' not in stim_dict:
            raise ValueError(f"Stimulus `{key.get('stim_name')}` has no `array_to_screen` in its stim_dict. Declare "
                             f"how the stimulus array axes map to the QDSpy screen, e.g. "
                             f"djimaging.utils.stimulator.NOISE_ARRAY_TO_SCREEN for the standard dense noise.")
        geometry = self.experiment_stimulator_tab().get_geometry(key)

        relx_rf_roi_um, rely_rf_roi_um = compute_rf_roi_offset(
            rf_dx_um, rf_dy_um, relx_wrt_field, rely_wrt_field, stim_dict['array_to_screen'], geometry)

        self.insert1(dict(**key, relx_rf_roi_um=relx_rf_roi_um, rely_rf_roi_um=rely_rf_roi_um,
                          geometry_confidence=geometry['confidence'], geometry_reason=geometry['reason'][:1023]))

    def plot(self, restriction: dict | None = None) -> tuple:
        """Plot RF-to-ROI offsets and regression lines.

        Parameters
        ----------
        restriction : dict or None, optional
            DataJoint restriction to apply. Default is None (no restriction).

        Returns
        -------
        tuple
            Tuple of (fig, ax) matplotlib objects.
        """
        import seaborn as sns

        if restriction is None:
            restriction = {}

        df = (self * self.roi_pos_wrt_field_tab() & restriction).proj(
            'relx_wrt_field', 'rely_wrt_field', 'relx_rf_roi_um', 'rely_rf_roi_um').to_pandas()
        df['relx_rf_wrt_field'] = df['relx_rf_roi_um'] + df['relx_wrt_field']
        df['rely_rf_wrt_field'] = df['rely_rf_roi_um'] + df['rely_wrt_field']

        fig, axs = plt.subplots(1, 5, figsize=(19, 4))

        ax = axs[0]
        sns.regplot(ax=ax, data=df, x='relx_wrt_field', y='relx_rf_wrt_field', scatter=False, color='black')

        ax = axs[1]
        sns.regplot(ax=ax, data=df, x='rely_wrt_field', y='rely_rf_wrt_field', scatter=False, color='black')

        ax = axs[2]
        sns.regplot(ax=ax, data=df, x='relx_rf_roi_um', y='rely_rf_roi_um', scatter=False, color='black')

        ax = axs[3]
        sns.regplot(ax=ax, data=df, x='rely_rf_roi_um', y='relx_rf_roi_um', scatter=False, color='black')

        s_kws = {'s': 5, "alpha": 0.8, 'zorder': -100}
        groups = df.reset_index().groupby(['experimenter', 'date', 'exp_num'])
        colors = sns.color_palette('brg', len(groups))
        for i, (field_info, df_field) in enumerate(groups):
            ax = axs[0]
            sns.scatterplot(ax=ax, data=df_field, x='relx_wrt_field', y='relx_rf_wrt_field', color=colors[i], **s_kws)

            ax = axs[1]
            sns.scatterplot(ax=ax, data=df_field, x='rely_wrt_field', y='rely_rf_wrt_field', color=colors[i], **s_kws)

            ax = axs[2]
            sns.scatterplot(ax=ax, data=df_field, x='relx_rf_roi_um', y='rely_rf_roi_um', color=colors[i], **s_kws)

            ax = axs[3]
            sns.scatterplot(ax=ax, data=df_field, x='rely_rf_roi_um', y='relx_rf_roi_um', color=colors[i], **s_kws)

        for ax in axs[:4]:
            ax.grid()
            ax.set_aspect('equal', 'datalim')
            ax.axline([-1, -1], [+1, +1], color='dimgray', ls='--', lw=1)

        self._plot_affine(axs[4], df)

        plt.tight_layout()

        return fig, ax

    @staticmethod
    def _plot_affine(ax, df, min_rois_per_field: int = 10) -> dict:
        """Plot the affine map ROI position -> RF position as the image of the unit square.

        Dashed: identity (expected). Thick black: pooled fit with one intercept per field. Thin colored: per-field fits
        (fields with at least `min_rois_per_field` rows). Marker at the image of (1, 0) shows the orientation.
        """
        df = df.reset_index()
        field_cols = [c for c in ['experimenter', 'date', 'exp_num', 'raw_id', 'field', 'region', 'cond1', 'cond2',
                                  'cond3'] if c in df.columns]
        field_ids = df.groupby(field_cols, sort=False).ngroup().values if field_cols else None

        roi_xy = df[['relx_wrt_field', 'rely_wrt_field']].values
        rf_xy = df[['relx_rf_wrt_field', 'rely_rf_wrt_field']].values
        fit = fit_affine_roi_to_rf(roi_xy, rf_xy, groups=field_ids)

        square = np.array([[1, 0], [1, 1], [0, 1], [0, 0], [1, 0]], dtype=float).T

        def draw(A, **kws):
            pts = A @ square
            ax.plot(pts[0], pts[1], **kws)
            ax.plot(*pts[:, 0], 'o', ms=kws.get('lw', 1) * 2, color=kws.get('color'), alpha=kws.get('alpha', 1))

        draw(np.eye(2), color='dimgray', ls='--', lw=1)

        if field_ids is not None and len(np.unique(field_ids)) > 1:
            for i, g in enumerate(np.unique(field_ids)):
                idx = field_ids == g
                if idx.sum() < min_rois_per_field:
                    continue
                fit_g = fit_affine_roi_to_rf(roi_xy[idx], rf_xy[idx])
                if np.all(np.isfinite(fit_g['A'])):
                    draw(fit_g['A'], color=f'C{i % 10}', lw=0.8, alpha=0.6)

        if np.all(np.isfinite(fit['A'])):
            draw(fit['A'], color='black', lw=2)

        ax.text(0.02, 0.98, _format_affine(fit), transform=ax.transAxes, va='top', ha='left', fontsize=8,
                family='monospace', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        ax.axhline(0, color='lightgray', lw=0.5, zorder=-10)
        ax.axvline(0, color='lightgray', lw=0.5, zorder=-10)
        ax.set_aspect('equal', 'datalim')
        ax.set(xlabel='XCoord (field frame)', ylabel='YCoord (field frame)', title='Affine ROI → RF')

        return fit

    def plot_field(self, key: dict, rf_qidx_thresh: float = 0.45, n_std: float = 1,
                   gamma: float = 0.7, adjust_rfs: tuple = (0, 0),
                   plot_offset: bool = True, plot_outline: bool = True) -> None:
        """Plot the field image with RF ellipses and ROI-to-RF offset vectors.

        Parameters
        ----------
        key : dict
            DataJoint restriction key for the field.
        rf_qidx_thresh : float, optional
            Minimum RF quality index to include an ROI. Default is 0.45.
        n_std : float, optional
            Number of standard deviations for the ellipse radius. Default is 1.
        gamma : float, optional
            Gamma correction applied to the field image. Default is 0.7.
        adjust_rfs : tuple, optional
            (dx, dy) adjustment added to all RF center positions. Default is (0, 0).
        plot_offset : bool, optional
            If True, draw line from ROI to RF center. Default is True.
        plot_outline : bool, optional
            If True, draw Gaussian ellipse around RF center. Default is True.
        """
        from matplotlib.patches import Ellipse
        import seaborn as sns

        mask_rows = (self.roimask_tab & key).to_dicts()
        if len(mask_rows) != 1:
            raise ValueError(f'Expected one presentation ROI mask for key={key}, found {len(mask_rows)}. '
                             f'With several motion parameter sets, add motion_params_id to the key.')
        roi_mask = load_array(mask_rows[0]['roi_mask'])
        shift_dx, shift_dy = mask_rows[0]['shift_dx'], mask_rows[0]['shift_dy']
        pixel_size_um, npixartifact, scan_type = (self.pres_tab & key).to_arrays(
            'pixel_size_um', 'npixartifact', 'scan_type')

        data_name, alt_name = self.userinfo_tab().get_stack_names(key)
        main_ch_average = load_array(
            (self.pres_tab.StackAverages & key & dict(ch_name=data_name)).fetch1('ch_average'))
        stim_dict = (self.stimulus_tab & key).fetch1('stim_dict')
        pix_scale_x_um, pix_scale_y_um = stim_dict['pix_scale_x_um'], stim_dict['pix_scale_y_um']
        geometry = self.experiment_stimulator_tab().get_geometry(key)
        array_to_plot = _FIELD_TO_PLOT @ array_to_field_matrix(
            stim_dict['array_to_screen'], geometry['screen_x_axis'], geometry['screen_y_axis'])

        if scan_type == 'xy':
            extent = np.array([
                main_ch_average.shape[0] / 2 - shift_dx, -main_ch_average.shape[0] / 2 - shift_dx,
                main_ch_average.shape[1] / 2 - shift_dy, -main_ch_average.shape[1] / 2 - shift_dy
            ]) * pixel_size_um
        else:
            extent = np.array([
                main_ch_average.shape[0] / 2 - shift_dx, -main_ch_average.shape[0] / 2 - shift_dx,
                main_ch_average.shape[1] / 20, -main_ch_average.shape[1] / 20
            ]) * pixel_size_um

        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.imshow(main_ch_average.T ** gamma, extent=extent, origin='lower', cmap='gray')

        _rois = roi_mask.copy().astype(float)
        _rois[_rois >= 0] = 0
        _rois[_rois < 0] = -1
        ax.contour(-_rois.T, cmap='Reds', origin='lower', extent=extent, levels=[0.99], vmin=0, vmax=1)

        roi_keys = (self.rf_fit_tab & f"rf_qidx>{rf_qidx_thresh}" & key).keys()
        colors = sns.color_palette('Spectral', len(roi_keys))

        for i, roi_key in enumerate(roi_keys):
            relx_wrt_field, rely_wrt_field, relz_wrt_field, relx_rf_roi_um, rely_rf_roi_um = (
                    self * self.roi_pos_wrt_field_tab() & roi_key).fetch1(
                'relx_wrt_field', 'rely_wrt_field', 'relz_wrt_field', 'relx_rf_roi_um', 'rely_rf_roi_um')

            if scan_type == 'xy':
                roi_xy = rely_wrt_field, relx_wrt_field
            elif scan_type == 'xz':
                roi_xy = rely_wrt_field, -relz_wrt_field / 5
            else:
                raise NotImplementedError(scan_type)
            rf_xy = roi_xy[0] + rely_rf_roi_um + adjust_rfs[0], roi_xy[1] - relx_rf_roi_um + adjust_rfs[1]

            ax.plot(*roi_xy, 'X', zorder=100, ms=3, c=colors[i])

            if plot_offset:
                ax.plot([roi_xy[0], rf_xy[0]], [roi_xy[1], rf_xy[1]], '-', zorder=100, ms=3, c=colors[i])

            if plot_outline:
                if isinstance(self.rf_fit_tab(), FitGauss2DRFTemplate):
                    srf_params = (self.rf_fit_tab & roi_key).fetch1("srf_params")
                elif isinstance(self.rf_fit_tab(), FitDoG2DRFTemplate):
                    srf_params = (self.rf_fit_tab & roi_key).fetch1("srf_eff_center_params")
                else:
                    srf_params = None

                if srf_params is not None:
                    # Ellipse in stimulus array coordinates, mapped to the plot frame
                    width, height, angle = ellipse_to_frame(
                        n_std * 2 * srf_params['x_stddev'] * pix_scale_x_um,
                        n_std * 2 * srf_params['y_stddev'] * pix_scale_y_um,
                        np.rad2deg(srf_params['theta']), array_to_plot)
                    ax.add_patch(Ellipse(xy=rf_xy, width=width, height=height, angle=angle,
                                         color=colors[i], fill=False, alpha=0.5))

        if scan_type == 'xy':
            ax.set_xlabel('relY [µm]')
            ax.set_ylabel('relX [µm]')
        else:
            ax.set_xlabel('µm')
            ax.set_ylabel('µm')
