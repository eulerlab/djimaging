"""
This module defines the optic disk table.
Mostly used for location of the recorded field center relative to the optic disk.
"""

import os
import warnings
from abc import abstractmethod

import datajoint as dj

from djimaging.utils.scanm.read_h5_utils import load_h5_table
from djimaging.utils.filesystem_utils import get_file_info_df
from djimaging.utils.scanm import read_smp_utils


class OpticDiskTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = """
        # location of the recorded field center relative to the optic disk
        # XCoord_um is the relative position from back towards curtain, i.e. larger XCoord_um means closer to curtain
        # YCoord_um is the relative position from left to right, i.e. larger YCoord_um means more right
        -> self.experiment_table
        -> self.raw_params_table
        ---
        od_fromfile :varchar(191)  # File from which optic disc data was extracted
        odx      :float            # XCoord_um relative to the optic disk
        ody      :float            # YCoord_um relative to the optic disk
        odz      :float            # ZCoord_um relative to the optic disk
        """
        return definition

    @property
    def key_source(self):
        try:
            return self.experiment_table.proj() * self.raw_params_table.proj()
        except (AttributeError, TypeError):
            pass

    @property
    @abstractmethod
    def experiment_table(self):
        pass

    @property
    @abstractmethod
    def raw_params_table(self):
        pass

    @property
    @abstractmethod
    def userinfo_table(self):
        pass

    def load_exp_file_info_df(self, exp_key):
        from_raw_data = (self.raw_params_table & exp_key).fetch1('from_raw_data')
        header_path = (self.experiment_table & exp_key).fetch1('header_path')
        data_folder = (self.userinfo_table & exp_key).fetch1("raw_data_dir" if from_raw_data else "pre_data_dir")
        user_dict = (self.userinfo_table & exp_key).fetch1()

        file_info_df = get_file_info_df(os.path.join(header_path, data_folder), user_dict, from_raw_data)

        if len(file_info_df) > 0:
            file_info_df = file_info_df[file_info_df['kind'] == 'od']

        return file_info_df, from_raw_data

    def make(self, key):
        fromfile, odx, ody, odz = None, None, None, None

        file_info_df, from_raw_data = self.load_exp_file_info_df(key)
        if len(file_info_df) > 0:
            fromfile = file_info_df['filepath'].iloc[0]

        if fromfile is not None:
            pre_data_dir, raw_data_dir = (self.userinfo_table & key).fetch1("pre_data_dir", "raw_data_dir")
            odx, ody, odz = load_od_pos_from_file(
                fromfile, from_raw_data, fallback_raw=True, pre_data_dir=pre_data_dir, raw_data_dir=raw_data_dir)
        else:
            if (self.experiment_table().ExpInfo & key).fetch1("od_ini_flag") == 1:
                odx, ody, odz = (self.experiment_table().ExpInfo & key).fetch1("odx", "ody", "odz")
                fromfile = os.path.join(*(self.experiment_table & key).fetch1("header_path", "header_name"))

        if fromfile is None:
            warnings.warn(f'No optic disk information found for {key}')
            return

        loc_key = key.copy()
        loc_key["odx"] = odx
        loc_key["ody"] = ody
        loc_key["odz"] = odz
        loc_key["od_fromfile"] = fromfile

        self.insert1(loc_key)

def load_od_pos_from_file(filepath, from_raw_data, fallback_raw=True, raw_data_dir='Raw', pre_data_dir='Pre'):
    if from_raw_data:
        odx, ody, odz = load_od_pos_from_smp_file(filepath)
    else:
        try:
            odx, ody, odz = load_od_pos_from_h5_file(filepath)
        except OSError as e:
            if fallback_raw:
                try:
                    filepath_raw = os.path.splitext(filepath.replace(f'/{pre_data_dir}/', f'/{raw_data_dir}/'))[0] + '.smp'
                    odx, ody, odz = load_od_pos_from_smp_file(filepath_raw)
                except:
                    raise e
            else:
                raise e
    return odx, ody, odz

def load_od_pos_from_h5_file(filepath):
    wparams = load_h5_table('wParamsNum', filename=filepath, lower_keys=True)
    odx, ody, odz = wparams['xcoord_um'], wparams['ycoord_um'], wparams['zcoord_um']
    return odx, ody, odz


def load_od_pos_from_smp_file(filepath):
    wparams = read_smp_utils.load_wparams(filepath, return_file=False)
    odx, ody, odz = wparams['xcoord_um'], wparams['ycoord_um'], wparams['zcoord_um']
    return odx, ody, odz
