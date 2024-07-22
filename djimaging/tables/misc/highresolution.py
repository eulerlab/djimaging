"""
High resolution stack images.

Example usage:

from djimaging.tables import misc

@schema
class HighRes(misc.HighResTemplate):
    field_table = Field
    experiment_table = Experiment
    userinfo_table = UserInfo

    class StackAverages(misc.HighResTemplate.StackAverages):
        pass
"""
import warnings
from abc import abstractmethod
from copy import deepcopy

import datajoint as dj
import numpy as np

from djimaging.utils.dj_utils import get_primary_key
from djimaging.utils.plot_utils import plot_field
from djimaging.utils.scanm.recording import ScanMRecording


class HighResTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = """
        # High resolution stack information.
        -> self.field_table
        ---
        highres_file :varchar(191)          # path to file (e.g. h5 file)
        absx: float  # absolute position of the center (of the cropped field) in the x axis as recorded by ScanM
        absy: float  # absolute position of the center (of the cropped field) in the y axis as recorded by ScanM
        absz: float  # absolute position of the center (of the cropped field) in the z axis as recorded by ScanM
        scan_type: enum("xy", "xz", "xyz")  # Type of scan
        npixartifact : int unsigned         # number of pixel with light artifact
        nxpix: int unsigned                 # number of pixels in x
        nypix: int unsigned                 # number of pixels in y
        nzpix: int unsigned                 # number of pixels in z
        nxpix_offset: int unsigned          # number of offset pixels in x
        nxpix_retrace: int unsigned         # number of retrace pixels in x
        pixel_size_um :float                # width of a pixel in um (also height if y is second dimension)
        z_step_um = NULL :float             # z-step in um
        nframes: int unsigned               # number of pixels in time
        """
        return definition

    @property
    def key_source(self):
        try:
            return self.field_table.proj()
        except (AttributeError, TypeError):
            pass

    @property
    @abstractmethod
    def field_table(self):
        pass

    @property
    @abstractmethod
    def experiment_table(self):
        pass

    @property
    @abstractmethod
    def userinfo_table(self):
        pass

    class StackAverages(dj.Part):
        @property
        def definition(self):
            definition = """
            # Stack median (over time of the available channels)
            -> master
            ch_name : varchar(191)  # name of the channel
            ---
            ch_average :longblob  # Stack median over time
            """
            return definition

    def load_field_stim_file_info_df(self, field_stim_key):
        file_info_df = self.field_table().load_exp_file_info_df(field_stim_key, filter_kind='hr')

        for new_key in self.field_table().new_primary_keys:
            file_info_df = file_info_df[file_info_df[new_key] == field_stim_key[new_key]]
        return file_info_df

    def make(self, key):
        setupid = (self.experiment_table().ExpInfo & key).fetch1("setupid")

        file_info_df = self.load_field_stim_file_info_df(key)
        if len(file_info_df) == 0:
            warnings.warn(f'No highres files found for field {key}. Skipping.')
            return
        elif len(file_info_df) > 1:
            warnings.warn(f'Multiple highres files found for field {key}. Using first one.')

        filepath = file_info_df.iloc[0].filepath

        rec = ScanMRecording(filepath=filepath, setup_id=setupid, date=key['date'])
        hr_entry, avg_entries = self.complete_keys(key, rec)

        self.insert1(hr_entry)
        for avg_key in avg_entries:
            (self.StackAverages & key).insert1(avg_key, allow_direct_insert=True)

    def plot1(self, key=None, figsize=(8, 4), gamma=0.7):
        key = get_primary_key(table=self, key=key)

        data_name, alt_name = (self.userinfo_table & key).fetch1('data_stack_name', 'alt_stack_name')
        main_ch_average = (self.StackAverages & key & f'ch_name="{data_name}"').fetch1('ch_average')
        try:
            alt_ch_average = (self.StackAverages & key & f'ch_name="{alt_name}"').fetch1('ch_average')
        except dj.DataJointError:
            alt_ch_average = np.full_like(main_ch_average, np.nan)
        plot_field(main_ch_average, alt_ch_average, roi_mask=None, title=key, figsize=figsize, gamma=gamma)

    @staticmethod
    def complete_keys(base_key, rec) -> (dict, list):

        hr_entry = deepcopy(base_key)
        hr_entry["highres_file"] = rec.filepath

        hr_entry["absx"] = rec.pos_x_um
        hr_entry["absy"] = rec.pos_y_um
        hr_entry["absz"] = rec.pos_z_um
        hr_entry["scan_type"] = rec.scan_type
        hr_entry["npixartifact"] = rec.pix_n_artifact
        hr_entry["nxpix"] = rec.pix_nx
        hr_entry["nypix"] = rec.pix_ny
        hr_entry["nzpix"] = rec.pix_nz
        hr_entry["nxpix_offset"] = rec.pix_n_line_offset
        hr_entry["nxpix_retrace"] = rec.pix_n_retrace
        hr_entry["pixel_size_um"] = rec.pix_dx_um
        hr_entry["z_step_um"] = rec.pix_dz_um
        hr_entry["nframes"] = rec.ch_n_frames

        # get stack avgs
        avg_entries = []
        for name, stack in rec.ch_stacks.items():
            avg_entry = deepcopy(base_key)
            avg_entry["ch_name"] = name
            avg_entry["ch_average"] = np.median(stack, 2)
            avg_entries.append(avg_entry)

        return hr_entry, avg_entries
