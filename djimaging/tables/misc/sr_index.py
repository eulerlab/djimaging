"""
Table for computing SR index for each ROI. SR index is defined as (ROI_avg - lb) / (ub - lb), where lb is the background
and lb is the light artifact intensity.

Example usage:

from djimaging.tables import misc

@schema
class SrIndex(misc.SrIndexTemplate):
    _stim_name = 'gChirp'

    presentation_table = Presentation
    roimask_table = RoiMask
    roi_table = Roi
    stimulus_table = Stimulus
    raw_params_table = RawDataParams
"""

import warnings
from abc import abstractmethod

import datajoint as dj
import numpy as np
from matplotlib import pyplot as plt

from djimaging.utils import mask_format_utils
from djimaging.utils.scanm import read_utils
from djimaging.utils.dj_utils import get_primary_key


class SrIndexTemplate(dj.Computed):
    """DataJoint computed table template for the SR (Sulforhodamine) index per ROI."""

    database = ""

    _stim_name = 'gChirp'  # Must have a light artifact, developed for global chirp

    @property
    def definition(self) -> str:
        definition = """
        # SR index, can be used to estimate how strongly a cell has be labeled by SR101
        # Defaults to NaN for ROIs with no pixels on the stack
        -> self.presentation_table
        -> self.roimask_table
        -> self.roi_table
        ---
        sr_idx = NULL  : float  # SR index >= 0 of ROI normalized to min of stack average and light_artifact = 1.
        """
        return definition

    @property
    @abstractmethod
    def presentation_table(self):
        pass

    @property
    @abstractmethod
    def roi_table(self):
        pass

    @property
    @abstractmethod
    def roimask_table(self):
        pass

    @property
    @abstractmethod
    def stimulus_table(self):
        pass

    @property
    @abstractmethod
    def raw_params_table(self):
        pass

    @property
    def key_source(self):
        try:
            return ((self.presentation_table.proj() & self.roi_table.proj()) *
                    self.roimask_table.proj() & f"stim_name='{self._stim_name}'")
        except (AttributeError, TypeError):
            pass

    def fetch_and_compute(self, key: dict, plot: bool = False, plot_sr_threshold: float = 0.5) -> tuple:
        """Compute SR indices without inserting. Use for plotting or development.

        Args:
            key: DataJoint primary key dict identifying the presentation and ROI mask.
            plot: If ``True``, generate diagnostic plots.
            plot_sr_threshold: SR index threshold used to colour SR-positive cells
                in the stack plot. Only relevant when ``plot=True``.

        Returns:
            A 2-tuple ``(roi_ids, sr_idxs)`` where ``roi_ids`` is an array of ROI
            identifiers and ``sr_idxs`` is the corresponding array of SR index values.
        """
        roi_mask = (self.roimask_table & key).fetch1('roi_mask')

        try:
            triggertimes = (self.presentation_table & key).fetch1('triggertimes')
            fs = (self.presentation_table.ScanInfo & key).fetch('scan_frequency')
            ntrigger_rep = (self.stimulus_table & key).fetch1('ntrigger_rep')

            # Find stim onset and offset for faster computations
            stim_onset_idx = int(triggertimes[0] * fs)
            stim_offset_idx = int(triggertimes[ntrigger_rep] * fs)
        except:  # There might be cases without triggertimes
            stim_onset_idx = 0
            stim_offset_idx = 100

        filepath = (self.presentation_table & key).fetch1(self.presentation_table().filepath)
        npixartifact = (self.presentation_table & key).fetch1('npixartifact')
        try:
            from_raw_data = (self.raw_params_table & key).fetch1('from_raw_data')
        except dj.DataJointError:
            from_raw_data = False

        ch_stacks, wparams = read_utils.load_stacks(
            filepath, from_raw_data=from_raw_data, ch_names=('wDataCh1',) if not plot else ('wDataCh0', 'wDataCh1'))

        ch1_stack = ch_stacks['wDataCh1'][:, :, stim_onset_idx:stim_offset_idx]

        roi_ids = (self.roi_table & key).fetch('roi_id')

        sr_idxs, roi_avgs, roi_avgs_means, lb, ub = compute_sr_idxs(
            ch1_stack, roi_ids, roi_mask, npixartifact=npixartifact)

        if plot:
            ch0_stack = ch_stacks['wDataCh0'][:, :, stim_onset_idx:stim_offset_idx]
            plot_roi_sr_idxs(roi_ids, roi_avgs, sr_idxs, lb, ub)
            plot_stack_sr_idxs(np.mean(ch0_stack, axis=2), np.mean(ch1_stack, axis=2),
                               roi_mask, roi_ids, sr_idxs, sr_threshold=plot_sr_threshold)

        return roi_ids, sr_idxs

    def make(self, key: dict, plot: bool = False, plot_sr_threshold: float = 0.5) -> None:
        """Compute and insert SR index values for all ROIs in a given presentation.

        Args:
            key: DataJoint primary key dict identifying the presentation and ROI mask.
            plot: If ``True``, generate diagnostic plots during computation.
            plot_sr_threshold: SR index threshold used to colour SR-positive cells.
                Only relevant when ``plot=True``.
        """
        roi_idxs, sr_idxs = self.fetch_and_compute(key, plot=plot, plot_sr_threshold=plot_sr_threshold)
        for roi_idx, sr_idx in zip(roi_idxs, sr_idxs):
            self.insert1(dict(key, roi_id=roi_idx, sr_idx=sr_idx))

    def plot1(self, key: dict = None, sr_threshold: float = 0.5) -> None:
        key = get_primary_key(table=self.key_source, key=key)
        self.fetch_and_compute(key, plot=True, plot_sr_threshold=sr_threshold)


def compute_sr_idxs(
        ch1_stack: np.ndarray,
        roi_ids: np.ndarray,
        roi_mask: np.ndarray,
        npixartifact: int,
) -> tuple:
    """Compute the SR index for each ROI.

    SR index is defined as ``(roi_avg - lb) / (ub - lb)`` where ``lb`` is
    the minimum pixel value outside the light-artifact rows and ``ub`` is
    the 99th percentile of the median light-artifact trace.

    Args:
        ch1_stack: 3-D array of shape ``(x, y, t)`` containing the channel-1
            (SR101) fluorescence stack.
        roi_ids: 1-D array of ROI identifiers to compute indices for.
        roi_mask: 2-D integer mask array where each ROI is identified by its
            positive integer id.
        npixartifact: Number of leading pixel rows that contain the light artifact.

    Returns:
        A 5-tuple ``(sr_idxs, roi_avgs, roi_avgs_means, lb, ub)`` where
        ``sr_idxs`` is an array of SR index values per ROI,
        ``roi_avgs`` is a list of per-ROI pixel value arrays,
        ``roi_avgs_means`` is the mean of each per-ROI array,
        ``lb`` is the lower bound (background), and ``ub`` is the upper bound
        (light artifact intensity).
    """
    roi_mask = mask_format_utils.as_python_format(roi_mask)

    ch1_avg = np.mean(ch1_stack, axis=2)

    roi_avgs = [ch1_avg[roi_mask == roi_id] for roi_id in roi_ids]
    roi_avgs_means = np.array([np.mean(roi_avg) if len(roi_avg) > 0 else np.nan for roi_avg in roi_avgs])

    if np.any(np.isnan(roi_avgs_means)):
        warnings.warn(f'SR-index=NaN: Some ROIs have no pixels on the stack. Shifted presentation ROI mask?')

    lb = np.nanmin(ch1_avg[npixartifact:, :])
    light_artifact = np.nanmedian(ch1_stack[0, :, :], axis=0)
    ub = np.nanpercentile(light_artifact, 99)

    sr_idxs = (roi_avgs_means - lb) / (ub - lb)

    return sr_idxs, roi_avgs, roi_avgs_means, lb, ub


def plot_stack_sr_idxs(
        ch0_avg: np.ndarray,
        ch1_avg: np.ndarray,
        roi_mask: np.ndarray,
        roi_ids: np.ndarray,
        sr_idxs: np.ndarray,
        sr_threshold: float = 0.5,
) -> None:
    """Plot SR indices on top of stack averages.

    Args:
        ch0_avg: 2-D array containing the channel-0 time-averaged image.
        ch1_avg: 2-D array containing the channel-1 time-averaged image.
        roi_mask: 2-D integer mask where each ROI is identified by its positive id.
        roi_ids: 1-D array of ROI identifiers.
        sr_idxs: 1-D array of SR index values corresponding to ``roi_ids``.
        sr_threshold: SR index threshold above which an ROI is considered SR-positive.
    """
    roi_mask = mask_format_utils.as_python_format(roi_mask)

    binary_sr_mask = np.isin(roi_mask, roi_ids[sr_idxs >= sr_threshold])

    sr_roi_map = np.zeros(roi_mask.shape)
    sr_roi_map[roi_mask == 0] = np.nan
    for roi_id, sr_idx in zip(roi_ids, sr_idxs):
        sr_roi_map[roi_mask == roi_id] = sr_idx

    fig, axs = plt.subplots(2, 6, figsize=(15, 3), height_ratios=(8, 1))

    ax = axs[0, 0]
    im = ax.imshow(ch0_avg.T, origin='lower')
    plt.colorbar(im, cax=axs[1, 0], orientation='horizontal')
    ax.set(title='ch0')

    ax = axs[0, 1]
    im = ax.imshow(ch1_avg.T, origin='lower')
    plt.colorbar(im, cax=axs[1, 1], orientation='horizontal')
    ax.set(title='ch1')

    ax = axs[0, 2]
    im = ax.imshow(roi_mask.T, origin='lower', cmap='jet', interpolation='none', vmin=0)
    plt.colorbar(im, cax=axs[1, 2], orientation='horizontal')
    ax.set(title='ROI mask')

    ax = axs[0, 3]
    im = ax.imshow(sr_roi_map.T, origin='lower', cmap='coolwarm', interpolation='none', vmin=0, vmax=1)
    plt.colorbar(im, cax=axs[1, 3], orientation='horizontal')
    ax.set(title='SR idx')

    ax = axs[0, 4]
    ax.imshow(ch0_avg.T, origin='lower')
    if np.any(binary_sr_mask):
        ax.contour(binary_sr_mask.astype(int).T, origin='lower', cmap='Reds', vmin=0, vmax=1, levels=[0.5], zorder=100)
    ax.set(title='ch0 + SR cells')

    ax = axs[0, 5]
    ax.imshow(ch1_avg.T, origin='lower')
    if np.any(binary_sr_mask):
        ax.contour(binary_sr_mask.astype(int).T, origin='lower', cmap='Reds', vmin=0, vmax=1, levels=[0.5], zorder=100)
    ax.set(title='ch1 + SR cells')

    plt.show()


def plot_roi_sr_idxs(
        roi_ids: np.ndarray,
        roi_avgs: list,
        sr_idxs: np.ndarray,
        lb: float,
        ub: float,
) -> None:
    """Plot boxplot of pixel means for each ROI and overlay SR indices.

    Args:
        roi_ids: 1-D array of ROI identifiers used as x-axis positions.
        roi_avgs: List of per-ROI pixel value arrays used to draw the boxplots.
        sr_idxs: 1-D array of SR index values corresponding to ``roi_ids``.
        lb: Lower bound (background) value shown as a green horizontal line.
        ub: Upper bound (light artifact) value shown as a red horizontal line.
    """
    fig, ax = plt.subplots(1, 1, figsize=(25, 3))

    ax.boxplot(roi_avgs, positions=roi_ids)
    ax.axhline(lb, c='green', label='lb')
    ax.axhline(ub, c='r', label='ub')
    ax.set(ylabel='pixel means', xlabel='ROI')
    ax.xaxis.set_tick_params(rotation=90)
    ax.legend()

    ax = ax.twinx()
    ax.plot(roi_ids, sr_idxs, 'bx')
    ax.set(ylabel='SR idx')

    plt.show()
