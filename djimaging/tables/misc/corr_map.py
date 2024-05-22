"""
Table for Correlation Mask.

Example usage:

from djimaging.tables import misc

@schema
class CorrMap(misc.CorrMapTemplate):
    userinfo_table = UserInfo
    presentation_table = Presentation
    raw_params_table = RawDataParams
"""
from abc import abstractmethod

import datajoint as dj

from djimaging.autorois.corr_roi_mask_utils import stack_corr_image
from djimaging.utils.dj_utils import get_primary_key
from djimaging.utils.scanm import read_utils


class CorrMapTemplate(dj.Computed):
    database = ""
    _cut_x = (1, 1)
    _cut_z = (1, 1)

    @property
    def definition(self):
        definition = """
        -> self.presentation_table
        ---
        corr_map : longblob  # Correlation mask for stack
        """
        return definition

    @property
    def key_source(self):
        try:
            return self.presentation_table.proj()
        except (AttributeError, TypeError):
            pass

    @property
    @abstractmethod
    def presentation_table(self):
        pass

    @property
    @abstractmethod
    def userinfo_table(self):
        pass

    @property
    @abstractmethod
    def raw_params_table(self):
        pass

    def make(self, key):
        filepath = (self.presentation_table & key).fetch1('pres_data_file')
        from_raw_data = (self.raw_params_table & key).fetch1('from_raw_data')
        data_name = (self.userinfo_table & key).fetch1('data_stack_name')

        stack = read_utils.load_stacks(filepath, from_raw_data, ch_names=(data_name,))[0][data_name]
        corr_map = stack_corr_image(stack, cut_x=self._cut_x, cut_z=self._cut_z)
        self.insert1(dict(key, corr_map=corr_map))

    def plot1(self, key=None):
        key = get_primary_key(self, key=key)

        data_name = (self.userinfo_table & key).fetch1('data_stack_name')
        main_ch_average = (self.presentation_table.StackAverages & key & f'ch_name="{data_name}"').fetch1('ch_average')

        corr_map = (self & key).fetch1('corr_map')

        from matplotlib import pyplot as plt

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        im = axs[0].imshow(main_ch_average, cmap='gray')
        plt.colorbar(im, ax=axs[0], label='Brightness')
        im = axs[1].imshow(corr_map, vmin=-1, vmax=+1, cmap='bwr')
        plt.colorbar(im, ax=axs[1], label='Correlation')
        plt.tight_layout()
        plt.show()
