import random

import datajoint as dj
import numpy as np

from djimaging.utils.dj_utils import PlaceholderTable
from djimaging.utils.plot_utils import plot_field


class RoiTemplate(dj.Computed):
    database = ""  # hack to suppress DJ error

    @property
    def definition(self):
        definition = """
        # ROI information
        -> self.field_table
        roi_id                  :int                # integer id of each ROI
        ---
        roi_size                :int                # number of pixels in ROI
        roi_size_um2            :float              # size of ROI in micrometers squared
        roi_dia_um              :float              # diameter of ROI in micrometers, if it was a circle
        """
        return definition

    field_table = PlaceholderTable

    def make(self, key):
        # load roi_mask for insert roi for a specific experiment and field
        roi_mask = (self.field_table.RoiMask() & key).fetch1("roi_mask")
        pixel_size_um = (self.field_table.FieldInfo() & key).fetch1("pixel_size_um")

        if not np.any(roi_mask):
            return

        field_key = dict(
            experimenter=key["experimenter"],
            date=key["date"],
            exp_num=key["exp_num"],
            field=key["field"],
        )

        roi_idxs = np.unique(roi_mask)
        roi_idxs = roi_idxs[roi_idxs < 0]  # remove background indeces (0 or 1)
        roi_idxs = roi_idxs[np.argsort(np.abs(roi_idxs))]  # Sort by value

        # add every roi to list and the bulk add to roi table
        for roi_idx in roi_idxs:
            roi_size = np.sum(roi_mask == roi_idx)
            roi_size_um2 = roi_size * pixel_size_um ** 2
            roi_dia_um = 2 * np.sqrt(roi_size_um2 / np.pi)

            self.insert1({
                **field_key,
                'roi_id': int(abs(roi_idx)),
                'roi_size': roi_size,
                'roi_size_um2': roi_size_um2,
                'roi_dia_um': roi_dia_um})

    def plot1(self, key=None):
        if key is not None:
            key = {k: v for k, v in key.items() if k in self.primary_key}
        else:
            key = random.choice(self.fetch(*self.primary_key, as_dict=True))

        ch0_average = (self.field_table.FieldInfo() & key).fetch1("ch0_average").T
        ch1_average = (self.field_table.FieldInfo() & key).fetch1("ch1_average").T
        roi_mask = (self.field_table.RoiMask() & key).fetch1("roi_mask").T

        plot_field(ch0_average, ch1_average, roi_mask=roi_mask, title=key, figsize=(16, 4), highlight_roi=key['roi_id'])
