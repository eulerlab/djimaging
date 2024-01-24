"""
Table for ROI brightness, can be used to estimate how strongly a cell has be labeled.

Example usage:

from djimaging.tables import misc

@schema
class RoiBrightness(misc.RoiBrightnessTemplate):
    userinfo_table = UserInfo
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
        # Computed as mean brightness in [0, 1] of ROI normalized to min and max of stack average
        -> self.presentation_table
        -> self.roi_table
        ---
        brightness_ch_data  :float  # For data channel defined by user
        brightness_ch_alt  :float   # For alternative channel defined by user
        """
        return definition

    @property
    def key_source(self):
        try:
            return self.presentation_table.proj() & self.roi_table.proj()
        except (AttributeError, TypeError):
            pass

    @property
    @abstractmethod
    def userinfo_table(self):
        pass

    @property
    @abstractmethod
    def roimask_table(self):
        pass

    @property
    @abstractmethod
    def presentation_table(self):
        pass

    @property
    @abstractmethod
    def roi_table(self):
        pass

    def make(self, key):
        roi_mask = (self.roimask_table & key).fetch1('roi_mask')

        data_name, alt_name = (self.userinfo_table & key).fetch1('data_stack_name', 'alt_stack_name')
        main_ch_average = (self.presentation_table.StackAverages & key & f'ch_name="{data_name}"').fetch1('ch_average')
        alt_ch_average = (self.presentation_table.StackAverages & key & f'ch_name="{alt_name}"').fetch1('ch_average')

        main_ch_average = math_utils.normalize_zero_one(main_ch_average)
        alt_ch_average = math_utils.normalize_zero_one(alt_ch_average)

        brightness_ch_data = np.mean(main_ch_average[roi_mask == -key['roi_id']])
        brightness_ch_alt = np.mean(alt_ch_average[roi_mask == -key['roi_id']])

        self.insert1(dict(key, brightness_ch_data=brightness_ch_data, brightness_ch_alt=brightness_ch_alt))
