"""
Example usage:

class RoiBrightness(misc.RoiBrightnessTemplate):
    presentation_table = Presentation
    roimask_table = RoiMask
    roi_table = Roi
"""

from abc import abstractmethod

import datajoint as dj
import numpy as np

from djimaging.utils import math_utils


class RoiBrightnessTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = """
        # ROI brightness, can be used to estimate how strongly a cell has be labeled
        -> self.presentation_table
        -> self.roi_table
        ---
        brightness_rel2all  : float  # Mean brightness in [0, 1] of ROI normalized to min and max of all pixels in field
        brightness_rel2cells : float  # Mean brightness in [0, 1] of ROI normalized to min and max of all cells in field
        """
        return definition

    @property
    @abstractmethod
    def presentation_table(self):
        pass

    @property
    @abstractmethod
    def roimask_table(self):
        pass

    @property
    @abstractmethod
    def roi_table(self):
        pass

    @property
    def key_source(self):
        try:
            return self.presentation_table.proj()
        except (AttributeError, TypeError):
            pass

    def make(self, key):
        stack_average = math_utils.normalize_zero_one((self.presentation_table & key).fetch1('stack_average'))
        roi_mask = (self.roimask_table & key).fetch1('roi_mask')

        roi_ids = (self.roi_table & key).fetch('roi_id')

        brightness_rel2all = np.full(len(roi_ids), np.nan)
        for i, roi_id in enumerate(roi_ids):
            brightness_rel2all[i] = np.mean(stack_average[roi_mask == roi_id])

        brightness_rel2cells = math_utils.normalize_zero_one(brightness_rel2all)

        entries = [dict(**key, roi_id=roi_id, brightness=brightness)
                   for roi_id, brightness in zip(roi_ids, brightness_rel2cells)]

        self.insert(entries)
