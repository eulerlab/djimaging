from abc import abstractmethod

import datajoint as dj
import numpy as np

from djimaging.utils.dj_utils import get_primary_key
from djimaging.utils.plot_utils import plot_field
from djimaging.utils.scanm_utils import extract_roi_idxs


class RoiTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = """
        # ROI information
        -> self.roi_mask_table
        roi_id           :int                # integer id of each ROI
        ---
        roi_size         :int                # number of pixels in ROI
        roi_size_um2     :float              # size of ROI in micrometers squared
        roi_dia_um       :float              # diameter of ROI in micrometers, if it was a circle
        artifact_flag    :tinyint unsigned   # flag if roi contains light artifact (1) or not (0)
        """
        return definition

    @property
    @abstractmethod
    def field_table(self):
        pass

    @property
    @abstractmethod
    def roi_mask_table(self):
        pass

    @property
    @abstractmethod
    def userinfo_table(self):
        pass

    @property
    def key_source(self):
        try:
            return self.roi_mask_table.proj()
        except (AttributeError, TypeError):
            pass

    def make(self, key):
        # load roi_mask for insert roi for a specific experiment and field
        roi_mask = (self.roi_mask_table & key).fetch1("roi_mask")

        scan_type = (self.field_table & key).fetch1("scan_type")
        pixel_size_um = (self.field_table & key).fetch1("pixel_size_um")
        z_step_um = (self.field_table & key).fetch1("z_step_um")
        npixartifact = (self.field_table & key).fetch1("npixartifact")

        if not np.any(roi_mask):
            return

        roi_idxs = extract_roi_idxs(roi_mask)

        # add every roi to list and the bulk add to roi table
        for roi_idx in roi_idxs:
            roi_mask_i = roi_mask == roi_idx
            artifact_flag = np.any(roi_mask_i[:npixartifact, :])
            roi_size = np.sum(roi_mask_i)
            if scan_type == 'xy':
                roi_size_um2 = roi_size * pixel_size_um ** 2
            elif scan_type == 'xz':
                roi_size_um2 = roi_size * pixel_size_um * z_step_um
            else:
                raise NotImplementedError(f"Can't compute ROI size for scan_type={scan_type}.")
            roi_dia_um = 2 * np.sqrt(roi_size_um2 / np.pi)

            self.insert1({
                **key,
                'roi_id': int(abs(roi_idx)),
                'artifact_flag': int(artifact_flag),
                'roi_size': roi_size,
                'roi_size_um2': roi_size_um2,
                'roi_dia_um': roi_dia_um})

    def plot1(self, key=None):
        key = get_primary_key(table=self, key=key)

        npixartifact = (self.field_table & key).fetch1('npixartifact')
        roi_mask = (self.roi_mask_table & key).fetch1("roi_mask")

        data_name, alt_name = (self.userinfo_table() & key).fetch1('data_stack_name', 'alt_stack_name')
        main_ch_average = (self.field_table.StackAverages & key & f'ch_name="{data_name}"').fetch1('ch_average')
        alt_ch_average = (self.field_table.StackAverages & key & f'ch_name="{alt_name}"').fetch1('ch_average')

        plot_field(main_ch_average, alt_ch_average, roi_mask=roi_mask, roi_ch_average=main_ch_average,
                   title=key, highlight_roi=key['roi_id'], npixartifact=npixartifact)
