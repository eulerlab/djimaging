"""
Joint (difference of) Gaussian fits of the sRFs of all colors of an RF with a prior on the RF position.

Example usage:
@schema
class FitSRFJointParams(receptivefield.FitSRFJointParamsTemplate):
    pass

@schema
class FitSRFJoint(receptivefield.FitSRFJointTemplate):
    split_rf_table = SplitRF
    stimulus_table = Stimulus
    params_table = FitSRFJointParams
"""

import warnings
from abc import abstractmethod

import datajoint as dj
import numpy as np
from matplotlib import pyplot as plt

from djimaging.utils.dj_storage import load_array
from djimaging.utils.dj_utils import get_primary_key
from djimaging.utils.plot_utils import plot_srf, prep_long_title
from djimaging.utils.receptive_fields.joint_srf_fit_utils import fit_srfs_joint
from djimaging.utils.receptive_fields.plot_rf_utils import plot_srf_gauss_fit
from djimaging.utils.receptive_fields.spatial_rf_utils import compute_gauss_srf_area, compute_surround_index, \
    compute_center_index


class FitSRFJointParamsTemplate(dj.Lookup):
    """Parameters of the joint (D)oG fit of the sRFs of all colors of an RF, see ``fit_srfs_joint``.

    The center prior is a Gaussian centered on the recording field in stimulus coordinates (stimulus center minus the
    stimulus offset ``offset_x_um``, ``offset_y_um`` of the ``stim_dict``, as in ``RfOffsetTemplate``) with SD
    ``center_prior_sd_um``. Both penalties are in units of the fraction of unexplained variance of all colors' sRFs.
    """
    database = ""

    @property
    def definition(self):
        definition = """
        fit_srf_params_id: int32 # unique param set id
        ---
        kind: varchar(8)  # 'gauss' or 'dog'
        center_prior_sd_um: float32  # SD of the Gaussian prior on the RF center position, in microns
        center_prior_weight: float32  # Weight of the center prior, 0: no prior
        coupling_weight: float32  # Weight of the coupling of the centers of the colors, 0: independent colors
        surround_ratio_min: float32  # Lower bound of the surround / center sigma ratio (DoG only)
        surround_ratio_max: float32  # Upper bound of the surround / center sigma ratio (DoG only)
        """
        return definition

    def add_default(self, skip_duplicates=False, **params):
        """Add default parameters to the table"""
        key = dict(
            fit_srf_params_id=1,
            kind='dog',
            center_prior_sd_um=200.,
            center_prior_weight=0.05,
            coupling_weight=0.05,
            surround_ratio_min=2.,
            surround_ratio_max=10.,
        )
        key.update(**params)
        if key['kind'] not in ('gauss', 'dog'):
            raise ValueError(f"kind must be 'gauss' or 'dog', got {key['kind']!r}")
        self.insert1(key, skip_duplicates=skip_duplicates)


class FitSRFJointTemplate(dj.Computed):
    """(Difference of) Gaussian fit of the sRF, fitted jointly for all colors of an RF (``color_idx`` of the split RF
    table) with a soft prior on the center position and a weak coupling of the centers of the colors.
    See ``djimaging.utils.receptive_fields.joint_srf_fit_utils.fit_srfs_joint``.

    One job fits all colors of an RF and inserts one row per color. Without colors, only the center prior applies.
    ``srf_params`` describes the center Gaussian with the parametrization of ``FitGauss2DRF`` (``x_mean``, ``y_mean``,
    ``x_stddev``, ``y_stddev``, ``theta``; sRF pixels and radians).
    """
    database = ""
    _color_attr = 'color_idx'

    @property
    def definition(self):
        definition = """
        -> self.split_rf_table
        -> self.params_table
        ---
        srf_fit: <blob>
        srf_center_fit: <blob>
        srf_params: <blob>  # Parameters of the fit, see fit_srfs_joint
        rf_qidx: float32  # Quality index as explained variance of the sRF by the fit
        rf_area_um2: float32  # Area covered by 2 standard deviations of the center Gaussian
        rf_cdia_um: float32  # Circle equivalent diameter of the center
        rf_dx_um: float32  # x offset of the center from the prior center (recording field), stimulus array coordinates
        rf_dy_um: float32  # y offset of the center from the prior center (recording field), stimulus array coordinates
        center_index: float32  # Weight and sign of center in sRF
        surround_index: float32  # Weight and sign of surround in sRF
        """
        return definition

    @property
    @abstractmethod
    def split_rf_table(self):
        pass

    @property
    @abstractmethod
    def stimulus_table(self):
        pass

    @property
    @abstractmethod
    def params_table(self):
        pass

    @property
    def key_source(self):
        try:
            # One job per RF, i.e. all colors together
            pk = [k for k in self.split_rf_table.primary_key if k != self._color_attr]
            return (dj.U(*pk) & self.split_rf_table) * self.params_table.proj()
        except (AttributeError, TypeError):
            pass

    def _order_by(self, table):
        return self._color_attr if self._color_attr in table.primary_key else None

    def _fetch_srf_pix_scale(self, key: dict, stim_dict: dict) -> tuple[float, float]:
        """Pixel size of the sRF in microns, accounting for an upsampling of the sRF in the split RF table."""
        if ('pix_scale_x_um' not in stim_dict) or ('pix_scale_y_um' not in stim_dict):
            raise ValueError("Pixel size not found in stimulus dict")
        scale = 1
        params_table = getattr(self.split_rf_table, 'split_rf_params_table', None)
        if params_table is not None and 'upsample_srf_scale' in params_table.heading.names:
            scales = np.unique((params_table & key).to_arrays('upsample_srf_scale'))
            if scales.size != 1:
                raise ValueError(f'Expected one upsample_srf_scale for key={key}, got {scales}')
            scale = max(int(scales[0]), 1)
        return stim_dict['pix_scale_x_um'] / scale, stim_dict['pix_scale_y_um'] / scale

    def make(self, key):
        rows = (self.split_rf_table & key).proj('srf').to_dicts(order_by=self._order_by(self.split_rf_table))
        srfs = np.stack([load_array(row.pop('srf')) for row in rows])
        stim_dict = (self.stimulus_table & key).fetch1("stim_dict")
        kind, center_prior_sd_um, center_prior_weight, coupling_weight, surround_ratio_min, surround_ratio_max = (
                self.params_table & key).fetch1(
            'kind', 'center_prior_sd_um', 'center_prior_weight', 'coupling_weight', 'surround_ratio_min',
            'surround_ratio_max')

        pix_x_um, pix_y_um = self._fetch_srf_pix_scale(key, stim_dict)
        ny, nx = srfs.shape[1:]
        # Recording field in sRF pixel coordinates, see RfOffsetTemplate for the convention
        prior_x = (nx - 1) / 2 - stim_dict.get('offset_x_um', 0) / pix_x_um
        prior_y = (ny - 1) / 2 - stim_dict.get('offset_y_um', 0) / pix_y_um

        srf_fits, srf_center_fits, _, srf_params = fit_srfs_joint(
            srfs, kind=kind, prior_center=(prior_x, prior_y),
            center_prior_sd=(center_prior_sd_um / pix_x_um, center_prior_sd_um / pix_y_um),
            center_prior_weight=center_prior_weight, coupling_weight=coupling_weight,
            surround_ratio=(surround_ratio_min, surround_ratio_max))

        if not np.all(np.isfinite(srf_fits)):
            warnings.warn(f'Failed to fit sRFs for key={key}')
            return

        entries = []
        for row, srf, srf_fit, srf_center_fit, params in zip(rows, srfs, srf_fits, srf_center_fits, srf_params):
            area, diameter = compute_gauss_srf_area(params, pix_x_um, pix_y_um)
            entries.append(dict(
                **{**key, **row},  # row holds the full primary key of the split RF table (overlaps with key)
                srf_fit=srf_fit.astype(np.float32),
                srf_center_fit=srf_center_fit.astype(np.float32),
                srf_params=params,
                rf_qidx=params['r2'],
                rf_area_um2=area,
                rf_cdia_um=diameter,
                rf_dx_um=(params['x_mean'] - prior_x) * pix_x_um,
                rf_dy_um=(params['y_mean'] - prior_y) * pix_y_um,
                center_index=compute_center_index(srf=srf, srf_center=srf_center_fit),
                surround_index=compute_surround_index(srf=srf, srf_center=srf_center_fit),
            ))

        self.insert(entries)

    def plot1(self, key=None):
        """Plot the fits of all colors of an RF."""
        key = get_primary_key(table=self, key=key)
        key = {k: v for k, v in key.items() if k != self._color_attr}
        rows = (self * self.split_rf_table.proj('srf') & key).to_dicts(order_by=self._order_by(self))

        fig, axs = plt.subplots(len(rows), 4, figsize=(12, 3 * len(rows)), squeeze=False)
        fig.suptitle(prep_long_title(key))
        for ax_row, row in zip(axs, rows):
            srf, srf_fit = load_array(row['srf']), load_array(row['srf_fit'])
            vabsmax = np.maximum(np.max(np.abs(srf)), np.max(np.abs(srf_fit)))
            color_str = f"{self._color_attr}={row[self._color_attr]}: " if self._color_attr in row else ''

            plot_srf_gauss_fit(ax_row[0], srf=srf, srf_params=row['srf_params'], vabsmax=vabsmax, plot_cb=True)
            ax_row[0].set_title(f'{color_str}sRF')
            plot_srf(srf_fit, ax=ax_row[1], vabsmax=vabsmax)
            ax_row[1].set_title('sRF fit')
            plot_srf(srf - srf_fit, ax=ax_row[2], vabsmax=vabsmax)
            ax_row[2].set_title(f"Difference: QI={row['rf_qidx']:.2f}")
            plot_srf(load_array(row['srf_center_fit']), ax=ax_row[3], vabsmax=vabsmax)
            ax_row[3].set_title('sRF center fit')

        plt.tight_layout()
        plt.show()

    def plot(self, by=("preprocess_id",)):
        columns = ["rf_qidx", "rf_cdia_um", "rf_dx_um", "rf_dy_um", "center_index", "surround_index"]
        df = self.proj(*columns).to_pandas().reset_index()
        for params, group in df.groupby(list(by)):
            print(by)
            print(params)
            group.hist(column=columns)
            plt.show()
