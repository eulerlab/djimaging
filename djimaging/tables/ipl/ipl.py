from abc import abstractmethod

import datajoint as dj
import numpy as np

from djimaging.utils.dj_utils import get_primary_key
from matplotlib import pyplot as plt

from djimaging.utils.plot_utils import plot_roi_mask_boundaries

try:
    import ipywidgets as widgets
except ImportError:
    raise ImportError("Please install ipywidgets to use this module")

from djimaging.utils.scanm_utils import get_roi_centers
from djimaging.utils.math_utils import normalize_soft_zero_one


class IplBordersTemplate(dj.Manual):
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
    def key_source(self):
        try:
            return self.field_or_pres_table.proj()
        except (AttributeError, TypeError):
            pass

    def list_missing_keys(self):
        missing_keys = (self.key_source - self.proj()).fetch(as_dict=True)
        return missing_keys

    def gui(self, key, figsize=(6, 6), left0=12, right0=12, thick0=30, q_clip=2.5):

        ch_avg = (self.field_or_pres_table.StackAverages & key).fetch('ch_average')[0]
        ch_avg = normalize_soft_zero_one(ch_avg, dq=q_clip)

        extent = (0, ch_avg.shape[0], 0, ch_avg.shape[1])
        title = ' '

        ch1_avg = (IplBorders().field_or_pres_table.StackAverages & key).fetch('ch_average')[1]

        add_or_update = self.add_or_update

        if len((self & key).proj()) == 1:
            left0, right0, thick0 = (self & key).fetch1('left', 'right', 'thick')

        w_bot = widgets.IntSlider(left0, min=1, max=ch_avg.shape[0] - 1, step=1)
        w_top = widgets.IntSlider(right0, min=1, max=ch_avg.shape[0] - 1, step=1)
        w_thick = widgets.IntSlider(thick0, min=1, max=ch_avg.shape[0] - 1, step=1)
        w_save = widgets.Checkbox(False)

        @widgets.interact(bot=w_bot, top=w_top, thick=w_thick, save=w_save)
        def plot_fit(left=left0, right=right0, thick=thick0, save=False):
            nonlocal title, key, add_or_update

            plot_field_and_fit(left, right, thick, ch_avg, ch1_avg, figsize=figsize, extent=extent, title=title)

            if save:
                add_or_update(key=key, left=left, right=right, thick=thick)
                title = f'saved: left={left}, right={right}, thick={thick}'
                w_save.value = False

    def add_or_update(self, key, left, right, thick):
        if len((self & key).proj()) == 0:
            self.insert1(dict(**key, left=left, right=right, thick=thick))
        else:
            self.update1(dict(**key, left=left, right=right, thick=thick))


class RoiIplDepthTemplate(dj.Computed):
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
    def _field_or_pres_table(self):
        try:
            field_or_pres_table = self.roi_table.field_or_pres_table
        except (AttributeError, TypeError):
            field_or_pres_table = self.roi_table.field_table
        return field_or_pres_table

    @property
    def key_source(self):
        try:
            return self._field_or_pres_table.proj() * self.ipl_border_table.proj()
        except (AttributeError, TypeError):
            pass

    def make(self, key):
        left, right, thick = (self.ipl_border_table & key).fetch1('left', 'right', 'thick')

        roi_ids = (self.roi_table & key).fetch("roi_id")
        roi_mask = (self.roi_table.field_or_pres_table.RoiMask & key).fetch1('roi_mask')

        roi_centers = get_roi_centers(roi_mask, roi_ids)

        # calculate the depth relative to the IPL borders
        m1, b1 = get_line([(0, left), (roi_mask.shape[0] - 1, right)])

        for roi_id, roi_center_xy in zip(roi_ids, roi_centers):
            shifts = m1 * roi_center_xy[0] + b1
            ipl_depth = (roi_center_xy[1] - shifts) / thick

            self.insert1(dict(**key, roi_id=roi_id, ipl_depth=ipl_depth))

    def plot_field(self, key=None, figsize=(6, 6), q_clip=2.5):
        if key is None:
            key = get_primary_key(self.key_source)

        left, right, thick = (self.ipl_border_table & key).fetch1('left', 'right', 'thick')
        ch_avg = (self._field_or_pres_table.StackAverages & key).fetch('ch_average')[0]
        ch_avg = normalize_soft_zero_one(ch_avg, dq=q_clip)
        roi_mask = (self.roi_table.field_or_pres_table.RoiMask & key).fetch1('roi_mask')

        extent = (0, ch_avg.shape[0], 0, ch_avg.shape[1])

        fig, ax = plot_field_and_fit(left=left, right=right, thick=thick, ch_avg=ch_avg, figsize=figsize)
        plot_roi_mask_boundaries(ax=ax, roi_mask=roi_mask, extent=extent)

        roi_idxs, ipl_depths = (self & key).fetch('roi_id', 'ipl_depth')

        # get colormap for ipl depths
        cmap = plt.cm.get_cmap('viridis')
        norm = plt.Normalize(vmin=np.min(ipl_depths), vmax=np.max(ipl_depths))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

        roi_centers = get_roi_centers(roi_mask, roi_idxs)

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


def plot_field_and_fit(left, right, thick, ch_avg, ch1_avg, figsize=(6, 6), extent=None, title=''):
    if extent is None:
        extent = (0, ch_avg.shape[0], 0, ch_avg.shape[1])

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    m1, b1 = get_line([(0, left), (63, right)])
    m2, b2 = get_line([(0, left + thick), (63, right + thick)])

    gcl_line = np.zeros((ch_avg.shape[0], 2))
    inl_line = np.zeros((ch_avg.shape[0], 2))

    gcl_line[:, 0] = np.arange(ch_avg.shape[0])
    gcl_line[:, 1] = m1 * gcl_line[:, 0] + b1

    inl_line[:, 0] = np.arange(ch_avg.shape[0])
    inl_line[:, 1] = m2 * inl_line[:, 0] + b2

    ax[0].imshow(ch_avg.T, cmap='gray', origin='lower', extent=extent)
    ax[0].plot(gcl_line[:, 0], gcl_line[:, 1], alpha=0.8)
    ax[0].plot(inl_line[:, 0], inl_line[:, 1], alpha=0.8)
    ax[0].set_title('Channel 0')

    ax[1].imshow(ch1_avg.T, cmap='gray', origin='lower')  # aspect=aspect)
    ax[1].set_title('Sulforhodamine 101')
    ax[1].plot(gcl_line[:, 0], gcl_line[:, 1], alpha=0.8)
    ax[1].plot(inl_line[:, 0], inl_line[:, 1], alpha=0.8)

    fig.suptitle(f'{key}', fontsize=25)
    plt.tight_layout()

    return fig, ax
