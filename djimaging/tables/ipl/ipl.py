"""
Defines tables for IPL depth analysis, e.g. to determine where in the IPL ROIs are located.

Example usage:

from djimaging.tables import ipl

@schema
class IplBorders(ipl.IplBordersTemplate):
    field_or_pres_table = Field
    userinfo_table = UserInfo


@schema
class RoiIplDepth(ipl.RoiIplDepthTemplate):
    roimask_table = RoiMask
    ipl_border_table = IplBorders
    roi_table = Roi
"""

from abc import abstractmethod

import datajoint as dj
import numpy as np

from djimaging.utils.dj_utils import get_primary_key
from matplotlib import pyplot as plt

from djimaging.utils.plot_utils import plot_roi_mask_boundaries, prep_long_title
from djimaging.utils.scanm_utils import get_roi_centers
from djimaging.utils.math_utils import normalize_soft_zero_one


class IplBordersTemplate(dj.Manual):
    database = ""

    @property
    def definition(self):
        definition = """
        # Manually determined index and thickness information for the IPL.
        # Use XZ widget notebook to determine values.
        -> self.field_or_pres_table
        ---
        left   :tinyint    # pixel index where gcl/ipl border intersects on left of image (with GCL up)
        right  :tinyint    # pixel index where gcl/ipl border intersects on right side of image
        thick  :tinyint    # pixel width of the ipl
        """
        return definition

    @property
    @abstractmethod
    def field_or_pres_table(self):
        pass

    @property
    @abstractmethod
    def userinfo_table(self):
        pass

    @property
    def key_source(self):
        try:
            return self.field_or_pres_table.proj()
        except (AttributeError, TypeError):
            pass

    def list_missing_keys(self):
        missing_keys = (self.key_source - self.proj()).fetch(as_dict=True)
        return missing_keys

    def fetch1_and_norm_ch_avgs(self, key, q_clip0=0.0, q_clip1=0.0):
        data_stack_name, alt_stack_name = (self.userinfo_table & key).fetch1('data_stack_name', 'alt_stack_name')

        ch0_avg = (self.field_or_pres_table.StackAverages & key & dict(ch_name=data_stack_name)).fetch1('ch_average')
        ch0_avg = normalize_soft_zero_one(ch0_avg, dq=q_clip0)

        ch1_avg = (self.field_or_pres_table.StackAverages & key & dict(ch_name=alt_stack_name)).fetch1('ch_average')
        ch1_avg = normalize_soft_zero_one(ch1_avg, dq=q_clip1)

        return ch0_avg, ch1_avg

    def gui(self, key, figsize=(8, 5), left0=12, right0=12, thick0=30, q_clip0=0.0, q_clip1=0.0):
        import ipywidgets as widgets

        ch0_avg, ch1_avg = self.fetch1_and_norm_ch_avgs(key, q_clip0=q_clip0, q_clip1=q_clip1)
        extent = (0, ch0_avg.shape[0], 0, ch0_avg.shape[1])
        title = str(key)

        add_or_update = self.add_or_update

        if len((self & key).proj()) == 1:
            left0, right0, thick0 = (self & key).fetch1('left', 'right', 'thick')

        w_bot = widgets.IntSlider(left0, min=1, max=ch0_avg.shape[0] - 1, step=1)
        w_top = widgets.IntSlider(right0, min=1, max=ch0_avg.shape[0] - 1, step=1)
        w_thick = widgets.IntSlider(thick0, min=1, max=ch0_avg.shape[0] - 1, step=1)
        w_save = widgets.Checkbox(False)

        @widgets.interact(bot=w_bot, top=w_top, thick=w_thick, save=w_save)
        def plot_fit(left=left0, right=right0, thick=thick0, save=False):
            nonlocal title, key, add_or_update

            plot_field_and_fit(left, right, thick, ch0_avg, ch1_avg, figsize=figsize, extent=extent, title=title)

            if save:
                add_or_update(left=left, right=right, thick=thick, key=key)
                title = f'SAVED: left={left}, right={right}, thick={thick} ### {str(key)}'
                w_save.value = False

    def add_or_update(self, key, left, right, thick):
        if len((self & key).proj()) == 0:
            self.insert1(dict(**key, left=left, right=right, thick=thick))
        else:
            self.update1(dict(**key, left=left, right=right, thick=thick))


class RoiIplDepthTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = """
        # Gives the IPL depth of the field's ROIs relative to the GCL (=0) and INL (=1)
        -> self.ipl_border_table
        -> self.roi_table
        ---
        ipl_depth : float  # Depth in the IPL relative to the GCL (=0) and INL (=1)
        """
        return definition

    @property
    @abstractmethod
    def ipl_border_table(self):
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
    def key_source(self):
        try:
            return self.roimask_table.proj() * self.ipl_border_table.proj()
        except (AttributeError, TypeError):
            pass

    def make(self, key):
        left, right, thick = (self.ipl_border_table & key).fetch1('left', 'right', 'thick')

        roi_mask = (self.roimask_table & key).fetch1('roi_mask')
        roi_ids = (self.roi_table & key).fetch("roi_id")

        roi_centers = get_roi_centers(roi_mask, roi_ids)

        # calculate the depth relative to the IPL borders
        m1, b1 = get_line([(0, left), (roi_mask.shape[0] - 1, right)])

        for roi_id, roi_center_xy in zip(roi_ids, roi_centers):
            shifts = m1 * roi_center_xy[0] + b1
            ipl_depth = (roi_center_xy[1] - shifts) / thick

            self.insert1(dict(**key, roi_id=roi_id, ipl_depth=ipl_depth))

    def plot_field(self, key=None, figsize=(8, 5), q_clip0=0.0, q_clip1=0.0):
        if key is None:
            key = get_primary_key(self.key_source & self)

        left, right, thick = (self.ipl_border_table & key).fetch1('left', 'right', 'thick')
        roi_mask = (self.roimask_table & key).fetch1('roi_mask')

        ch0_avg, ch1_avg = self.ipl_border_table().fetch1_and_norm_ch_avgs(key, q_clip0=q_clip0, q_clip1=q_clip1)

        extent = (0, ch0_avg.shape[0], 0, ch0_avg.shape[1])

        fig, axs = plot_field_and_fit(
            left=left, right=right, thick=thick, ch0_avg=ch0_avg, ch1_avg=ch1_avg, figsize=figsize, title=str(key))

        roi_idxs, ipl_depths = (self & key).fetch('roi_id', 'ipl_depth')

        # get colormap for ipl depths
        cmap = plt.cm.get_cmap('viridis')

        vmin = np.minimum(0., np.min(ipl_depths))
        vmax = np.maximum(1., np.max(ipl_depths))

        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

        roi_centers = get_roi_centers(roi_mask, roi_idxs)

        for ax in axs:
            plot_roi_mask_boundaries(ax=ax, roi_mask=roi_mask, extent=extent)

            for roi_id, ipl_depth, roi_center in zip(roi_idxs, ipl_depths, roi_centers):
                c = sm.to_rgba(ipl_depth)
                ax.plot(roi_center[0], roi_center[1], 'o', color=c, zorder=200000, ms=8, mec='k')

            cbar = fig.colorbar(sm, ax=ax)
            cbar.set_label('IPL depth')


def get_line(points):
    x_coords, y_coords = zip(*points)
    a = np.vstack([x_coords, np.ones(len(x_coords))]).T
    m, c = np.linalg.lstsq(a, y_coords, rcond=None)[0]
    return m, c


def plot_field_and_fit(left, right, thick, ch0_avg, ch1_avg, figsize=(8, 5), extent=None, title=''):
    if extent is None:
        extent = (0, ch0_avg.shape[0], 0, ch0_avg.shape[1])

    fig, axs = plt.subplots(1, 2, figsize=figsize)

    fig.suptitle(prep_long_title(title=title))

    m1, b1 = get_line([(0, left), (63, right)])
    m2, b2 = get_line([(0, left + thick), (63, right + thick)])

    gcl_line = np.zeros((ch0_avg.shape[0], 2))
    inl_line = np.zeros((ch0_avg.shape[0], 2))

    gcl_line[:, 0] = np.arange(ch0_avg.shape[0])
    gcl_line[:, 1] = m1 * gcl_line[:, 0] + b1

    inl_line[:, 0] = np.arange(ch0_avg.shape[0])
    inl_line[:, 1] = m2 * inl_line[:, 0] + b2

    axs[0].imshow(ch0_avg.T, cmap='viridis', vmin=0, vmax=1, origin='lower', extent=extent)
    axs[0].plot(gcl_line[:, 0], gcl_line[:, 1], alpha=0.8)
    axs[0].plot(inl_line[:, 0], inl_line[:, 1], alpha=0.8)
    axs[0].set_title('Data Channel')

    axs[1].imshow(ch1_avg.T, cmap='viridis', vmin=0, vmax=1, origin='lower', extent=extent)
    axs[1].set_title('2nd Channel (e.g. SR101)')
    axs[1].plot(gcl_line[:, 0], gcl_line[:, 1], alpha=0.8)
    axs[1].plot(inl_line[:, 0], inl_line[:, 1], alpha=0.8)

    plt.tight_layout()

    return fig, axs
