import datajoint as dj
import numpy as np

from djimaging.utils.dj_utils import PlaceholderTable, get_plot_key
from djimaging.utils.plot_utils import plot_field


class RoiTemplate(dj.Computed):
    database = ""  # hack to suppress DJ error

    @property
    def definition(self):
        definition = """
        # ROI information
        -> self.field_table
        roi_id           :int                # integer id of each ROI
        ---
        roi_size         :int                # number of pixels in ROI
        roi_size_um2     :float              # size of ROI in micrometers squared
        roi_dia_um       :float              # diameter of ROI in micrometers, if it was a circle
        artifact_flag    :tinyint unsigned   # flag if roi contains light artifact (1) or not (0)
        """
        return definition

    field_table = PlaceholderTable

    @property
    def key_source(self):
        return self.field_table.RoiMask()

    def make(self, key):
        # load roi_mask for insert roi for a specific experiment and field
        roi_mask = (self.field_table.RoiMask() & key).fetch1("roi_mask")
        pixel_size_um = (self.field_table() & key).fetch1("pixel_size_um")
        npixartifact = (self.field_table() & key).fetch1("npixartifact")

        if not np.any(roi_mask):
            return

        field_key = dict(
            experimenter=key["experimenter"],
            date=key["date"],
            exp_num=key["exp_num"],
            field=key["field"],
        )

        roi_idxs = np.unique(roi_mask)
        roi_idxs = roi_idxs[roi_idxs < 0]  # remove background indexes (0 or 1)
        roi_idxs = roi_idxs[np.argsort(np.abs(roi_idxs))]  # Sort by value

        # add every roi to list and the bulk add to roi table
        for roi_idx in roi_idxs:
            roi_mask_i = roi_mask == roi_idx
            artifact_flag = np.any(roi_mask_i[:npixartifact, :])
            roi_size = np.sum(roi_mask_i)
            roi_size_um2 = roi_size * pixel_size_um ** 2
            roi_dia_um = 2 * np.sqrt(roi_size_um2 / np.pi)

            self.insert1({
                **field_key,
                'roi_id': int(abs(roi_idx)),
                'artifact_flag': int(artifact_flag),
                'roi_size': roi_size,
                'roi_size_um2': roi_size_um2,
                'roi_dia_um': roi_dia_um})

    def plot1(self, key=None):
        key = get_plot_key(table=self, key=key)

        data_stack_name = (self.field_table.userinfo_table() & key).fetch1('data_stack_name')

        ch0_average = (self.field_table() & key).fetch1("ch0_average")
        ch1_average = (self.field_table() & key).fetch1("ch1_average")
        roi_mask = (self.field_table.RoiMask() & key).fetch1("roi_mask")

        plot_field(ch0_average, ch1_average, roi_mask=roi_mask,
                   roi_ch_average=ch1_average if '1' in data_stack_name else ch0_average,
                   title=key, figsize=(16, 4), highlight_roi=key['roi_id'])
