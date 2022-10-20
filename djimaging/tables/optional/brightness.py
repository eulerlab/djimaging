import datajoint as dj
import numpy as np

from djimaging.utils import math_utils
from djimaging.utils.dj_utils import PlaceholderTable


class RoiBrightnessTemplate(dj.Computed):
    database = ""  # hack to suppress DJ error

    @property
    def definition(self):
        definition = """
        # ROI brightness, can be used to estimate how strongly a cell has be labeled
        -> self.presentation_table
        -> self.roi_table
        ---
        brightness  :float  # Mean brightness in [0, 1] of ROI normalized to min and max of stack average
        """
        return definition

    field_table = PlaceholderTable
    presentation_table = PlaceholderTable
    roi_table = PlaceholderTable

    def make(self, key):
        roi_mask = (self.field_table.RoiMask() & key).fetch1('roi_mask')
        stack_average = math_utils.normalize_zero_one((self.presentation_table() & key).fetch1('stack_average'))
        brightness = np.mean(stack_average[roi_mask == -key['roi_id']])

        self.insert1(dict(key, brightness=brightness))
