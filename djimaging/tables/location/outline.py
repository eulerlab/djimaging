"""
Tables for location of the recorded outline fields.

Example usage:

from djimaging.tables import location

@schema
class OutlineAbs(location.OutlineAbsTemplate):
    _from_raw_data = True
    _num_loc = None  # Can be used if number of field is not in field name

    experiment_table = Experiment
    userinfo_table = UserInfo

    class OutlineAbsField(location.OutlineAbsTemplate.OutlineAbsField):
        pass

@schema
class OutlineRel(location.OutlineRelTemplate):
    opticdisk_table = OpticDisk
    outline_abs_table = OutlineAbs
    expinfo_table = Experiment.ExpInfo

    class OutlineRelField(location.OutlineRelTemplate.OutlineRelField):
        pass
"""

import os
import warnings
from abc import abstractmethod
from copy import deepcopy

import datajoint as dj
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from djimaging.utils.dj_utils import get_primary_key
from djimaging.utils.filesystem_utils import get_file_info_df
from djimaging.utils.scanm.recording import ScanMRecording
from djimaging.utils.scanm.setup_utils import get_retinal_position


class OutlineAbsTemplate(dj.Computed):
    database = ""
    _from_raw_data = True  # Use SMP or h5 files?
    _num_loc = 3  # In which location of filename is the number found?

    @property
    def definition(self):
        definition = """
        -> self.experiment_table
        ---
        outline_abs_xy : blob  # outline of the retinal field in absolute coordinates
        """
        return definition

    class OutlineAbsField(dj.Part):
        definition = """
        -> master
        field   :varchar(32)          # string identifying files corresponding to field
        ---
        field_data_file: varchar(191)  # info extracted from which file?
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

    @property
    @abstractmethod
    def experiment_table(self):
        pass

    @property
    @abstractmethod
    def opticdisk_table(self):
        pass

    @property
    @abstractmethod
    def userinfo_table(self):
        pass

    @property
    def key_source(self):
        try:
            return self.experiment_table.proj()
        except (AttributeError, TypeError):
            pass

    def load_exp_file_info_df(self, exp_key):
        from_raw_data = self._from_raw_data
        header_path = (self.experiment_table & exp_key).fetch1('header_path')
        data_folder = (self.userinfo_table & exp_key).fetch1("raw_data_dir" if from_raw_data else "pre_data_dir")
        user_dict = (self.userinfo_table & exp_key).fetch1()

        if os.path.isdir(os.path.join(header_path, data_folder)):
            file_info_df = get_file_info_df(os.path.join(header_path, data_folder), user_dict, from_raw_data)
        else:
            warnings.warn(f"Data folder {data_folder} not found for key {exp_key}")
            return pd.DataFrame()

        file_info_df = file_info_df[file_info_df['kind'] == 'outline']

        if len(file_info_df) == 0:
            warnings.warn(
                f'No outline files found for field:\n\t{exp_key}.\n'
                f'Found files {os.listdir(os.path.join(header_path, data_folder))}'
            )

        return file_info_df

    def make(self, key):
        setupid = (self.experiment_table().ExpInfo & key).fetch1("setupid")

        file_info_df = self.load_exp_file_info_df(key)
        if len(file_info_df) == 0:
            return

        outline_abs_xy = []
        field_entries = []
        nums = []

        for i, row in file_info_df.iterrows():
            rec = ScanMRecording(filepath=row.filepath, setup_id=setupid, date=key['date'])
            field_key = key.copy()
            field_key['field'] = row['field']

            if self._num_loc is not None:
                try:
                    num = str(os.path.splitext(os.path.basename(row['filepath']))[0].split('_')[self._num_loc])
                    num = ''.join(filter(str.isdigit, num))
                    num = int(num)
                except ValueError:
                    raise ValueError(f'Could not extract number from {row["filepath"]}')
                nums.append(num)

            field_entry = self.complete_key(field_key, rec)
            field_entries.append(field_entry)

            outline_abs_xy.append((rec.pos_x_um, rec.pos_y_um))

        # Sort by numbers
        if self._num_loc is not None:
            idxs = np.argsort(nums)
            outline_abs_xy = [outline_abs_xy[i] for i in idxs]
            field_entries = [field_entries[i] for i in idxs]
            nums = [nums[i] for i in idxs]

        # Add num to field name if not already present
        if self._num_loc is not None:
            for i, field_entry in enumerate(field_entries):
                if str(nums[i]) not in field_entry['field']:
                    field_entry['field'] = field_entry['field'] + str(nums[i]).rjust(2, '0')

        main_entry = key.copy()
        main_entry['outline_abs_xy'] = outline_abs_xy

        self.insert1(main_entry)
        self.OutlineAbsField().insert(field_entries)

    @staticmethod
    def complete_key(base_key, rec) -> dict:

        field_entry = deepcopy(base_key)
        field_entry["field_data_file"] = rec.filepath

        field_entry["absx"] = rec.pos_x_um
        field_entry["absy"] = rec.pos_y_um
        field_entry["absz"] = rec.pos_z_um
        field_entry["scan_type"] = rec.scan_type
        field_entry["npixartifact"] = rec.pix_n_artifact
        field_entry["nxpix"] = rec.pix_nx
        field_entry["nypix"] = rec.pix_ny
        field_entry["nzpix"] = rec.pix_nz
        field_entry["nxpix_offset"] = rec.pix_n_line_offset
        field_entry["nxpix_retrace"] = rec.pix_n_retrace
        field_entry["pixel_size_um"] = rec.pix_dx_um
        field_entry["z_step_um"] = rec.pix_dz_um
        field_entry["nframes"] = rec.ch_n_frames

        return field_entry

    def plot1(self, key=None):
        key = get_primary_key(table=self, key=key)
        outline_abs_xy = (self & key).fetch1('outline_abs_xy')

        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        ax.plot(*np.array(outline_abs_xy).T, '.-')
        ax.set(title='Absolute', xlabel='absx_um', ylabel='absy_um')
        ax.set_aspect(aspect="equal", adjustable="datalim")

        plt.show()


class OutlineRelTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = """
        -> self.outline_abs_table
        -> self.opticdisk_table
        ---
        outline_rel_xy : blob  # outline of the retinal field in coordinates relative to the optic disk
        outline_retina_xy : blob  # outline of the retinal field in retinal coordinates
        """
        return definition

    class OutlineRelField(dj.Part):
        definition = """
        -> master
        field   :varchar(32)          # string identifying files corresponding to field
        ---
        relx: float  # relative position of the center (of the cropped field) in the x axis as recorded by ScanM
        rely: float  # relative position of the center (of the cropped field) in the y axis as recorded by ScanM
        relz: float  # relative position of the center (of the cropped field) in the z axis as recorded by ScanM
        ventral_dorsal_pos_um: float
        temporal_nasal_pos_um: float
        """

    @property
    @abstractmethod
    def opticdisk_table(self):
        pass

    @property
    @abstractmethod
    def outline_abs_table(self):
        pass

    @property
    @abstractmethod
    def expinfo_table(self):
        pass

    @property
    def key_source(self):
        try:
            return self.outline_abs_table.proj() * self.opticdisk_table.proj()
        except (AttributeError, TypeError):
            pass

    def make(self, key):
        print(key)

        odx, ody, odz = (self.opticdisk_table() & key).fetch1("odx", "ody", "odz")
        outline_abs_xy = (self.outline_abs_table() & key).fetch1('outline_abs_xy')
        absxs, absys, abszs, field_keys = (self.outline_abs_table().OutlineAbsField & key).fetch(
            'absx', 'absy', 'absz', 'KEY')

        eye, prepwmorient = (self.expinfo_table() & key).fetch1('eye', 'prepwmorient')

        if prepwmorient == -1:
            warnings.warn(f'prepwmorient is -1 for {key}. skipping')
            return

        outline_rel_xy = [
            (absx - odx, absy - ody)
            for absx, absy in outline_abs_xy
        ]

        outline_retina_xy = [
            get_retinal_position(rel_xcoord_um=relx, rel_ycoord_um=rely, rotation=prepwmorient, eye=eye)
            for relx, rely in outline_rel_xy
        ]

        field_entries = []
        for absx, absy, absz, field_key in zip(absxs, absys, abszs, field_keys):
            field_entry = key.copy()
            field_entry.update(field_key)
            field_entry["relx"] = absx - odx
            field_entry["rely"] = absy - ody
            field_entry["relz"] = absz - odz
            field_entry["ventral_dorsal_pos_um"], field_entry["temporal_nasal_pos_um"] = get_retinal_position(
                rel_xcoord_um=field_entry["relx"], rel_ycoord_um=field_entry["rely"], rotation=prepwmorient, eye=eye)

            field_entries.append(field_entry)

        main_entry = key.copy()
        main_entry['outline_rel_xy'] = outline_rel_xy
        main_entry['outline_retina_xy'] = outline_retina_xy

        self.insert1(main_entry)
        self.OutlineRelField().insert(field_entries)

    def plot1(self, key=None):
        key = get_primary_key(table=self, key=key)

        outline_abs_xy = (self.outline_abs_table & key).fetch1('outline_abs_xy')
        outline_rel_xy = (self & key).fetch1('outline_rel_xy')
        outline_retina_xy = (self & key).fetch1('outline_retina_xy')

        fields, relxs, relys = (self.OutlineRelField & key).fetch('field', 'relx', 'rely')

        fig, axs = plt.subplots(1, 3, figsize=(12, 3))

        ax = axs[0]
        ax.plot(*np.array(outline_abs_xy).T, '.-')
        ax.set(title='Absolute', xlabel='absx_um', ylabel='absy_um')
        ax.set_aspect(aspect="equal", adjustable="datalim")

        ax = axs[1]
        ax.plot(*np.array(outline_rel_xy).T, '.-')
        ax.set(title='Relative', xlabel='relx_um', ylabel='rely_um')
        ax.set_aspect(aspect="equal", adjustable="datalim")

        for field, relx, rely in zip(fields, relxs, relys):
            ax.text(relx, rely, field)

        ax = axs[2]
        ax.plot(*np.array(outline_retina_xy).T, '.-')
        ax.set(title='Retina', xlabel='temporal_nasal_pos_um', ylabel='ventral_dorsal_pos_um')
        ax.set_aspect(aspect="equal", adjustable="datalim")

        plt.tight_layout()
        plt.show()
