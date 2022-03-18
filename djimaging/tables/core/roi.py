import datajoint as dj
import numpy as np
from matplotlib import pyplot as plt

from djimaging.utils.dj_utils import PlaceholderTable


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

        experimenter = key["experimenter"]
        date = key["date"]
        exp_num = key["exp_num"]
        field = key["field"]

        roi_idxs = np.unique(roi_mask)
        roi_idxs = roi_idxs[(roi_idxs != 0) & (roi_idxs != 1)]  # remove background index (0 or 1)
        roi_idxs = roi_idxs[np.argsort(np.abs(roi_idxs))]  # Sort by value

        # add every roi to list and the bulk add to roi table
        roi_keys = []
        for roi_idx in roi_idxs:
            roi_size = np.sum(roi_mask == roi_idx)
            roi_size_um2 = roi_size * pixel_size_um ** 2
            roi_dia_um = 2 * np.sqrt(roi_size_um2 / np.pi)

            roi_keys.append({
                'experimenter': experimenter,
                'date': date,
                'exp_num': exp_num,
                'field': field,
                'roi_id': int(abs(roi_idx)),
                'roi_size': roi_size,
                'roi_size_um2': roi_size_um2,
                'roi_dia_um': roi_dia_um,
            })
        self.insert(roi_keys)

    def plot1(self, key):
        fig, axs = plt.subplots(1, 3, figsize=(15, 3.5))
        stack_average = (self.field_table.FieldInfo() & key).fetch1("stack_average").T
        roi_mask = (self.field_table.RoiMask() & key).fetch1("roi_mask").T
        axs[0].imshow(stack_average)
        axs[0].set(title='stack_average')
        roi_mask_im = axs[1].imshow(roi_mask, cmap='jet')
        plt.colorbar(roi_mask_im, ax=axs[1])
        axs[1].set(title='roi_mask')
        axs[2].imshow(roi_mask == -key['roi_id'])
        axs[2].set(title='ROI')
        plt.show()
