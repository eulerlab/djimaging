import os

import datajoint as dj
from matplotlib import pyplot as plt

from djimaging.tables.core.field import scan_fields_and_files, load_scan_info
from djimaging.utils.scanm_utils import get_retinal_position, get_pixel_size_um
from djimaging.utils.data_utils import load_h5_table
from djimaging.utils.dj_utils import PlaceholderTable


class OpticDiskTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = """
        # location of the recorded field center relative to the optic disk
        # XCoord_um is the relative position from back towards curtain, i.e. larger XCoord_um means closer to curtain
        # YCoord_um is the relative position from left to right, i.e. larger YCoord_um means more right
        -> self.experiment_table
        ---
        fromfile :varchar(255)  # File from which data was extraced
        odx      :float         # XCoord_um relative to the optic disk
        ody      :float         # YCoord_um relative to the optic disk
        odz      :float         # ZCoord_um relative to the optic disk
        """
        return definition

    experiment_table = PlaceholderTable
    userinfo_table = PlaceholderTable

    def make(self, key):
        user_dict = (self.userinfo_table() & key).fetch1()
        pre_data_path = (self.experiment_table() & key).fetch1('pre_data_path')

        field2info = scan_fields_and_files(pre_data_path, user_dict=user_dict)

        # Get correct
        for field, info in field2info.items():
            if field.lower() in user_dict['opticdisk_alias'].split('_'):
                files = info['files']
                assert len(files) == 1, files
                file = files[0]
                break
        else:
            file = None

        # Try to get OD information, either from file or from header
        if file is not None:

            filepath = os.path.join(pre_data_path, file)
            wparamsnum = load_h5_table('wParamsNum', filename=filepath)

            # Refers to center of fields
            odx = wparamsnum['XCoord_um']
            ody = wparamsnum['YCoord_um']
            odz = wparamsnum['ZCoord_um']

            fromfile = filepath

        elif (self.experiment_table.ExpInfo() & key).fetch1("od_ini_flag") == 1:
            print(f'Optic disk header information found for {key}')
            odx, ody, odz = (self.experiment_table.ExpInfo() & key).fetch1("odx", "ody", "odz")
            fromfile = os.path.join(*(self.experiment_table() & key).fetch1("header_path", "header_name"))

        else:
            print(f'No optic disk information found for {key}')
            return

        loc_key = key.copy()
        loc_key["odx"] = odx
        loc_key["ody"] = ody
        loc_key["odz"] = odz
        loc_key["fromfile"] = fromfile

        self.insert1(loc_key)


class RelativeFieldLocationTemplate(dj.Computed):
    database = ""  # hack to suppress DJ error

    @property
    def definition(self):
        definition = """
        # location of the recorded field center relative to the optic disk
        # XCoord_um is the relative position from back towards curtain, i.e. larger XCoord_um means closer to curtain
        # YCoord_um is the relative position from left to right, i.e. larger YCoord_um means more right
        
        -> self.opticdisk_table
        -> self.field_table
        ---
        relx   :float      # XCoord_um relative to the optic disk
        rely   :float      # YCoord_um relative to the optic disk
        relz   :float      # ZCoord_um relative to the optic disk
        """
        return definition

    opticdisk_table = PlaceholderTable
    field_table = PlaceholderTable

    def make(self, key):
        od_key = key.copy()
        od_key.pop('field', None)
        print(od_key)

        odx, ody, odz = (self.opticdisk_table() & od_key).fetch1("odx", "ody", "odz")
        absx, absy, absz = (self.field_table.FieldInfo() & key).fetch1('absx', 'absy', 'absz')

        loc_key = key.copy()
        loc_key["relx"] = absx - odx
        loc_key["rely"] = absy - ody
        loc_key["relz"] = absz - odz

        self.insert1(loc_key)

    def plot(self, key=None):
        relx, rely = self.fetch("relx", "rely")
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.scatter(rely, relx, label='all')
        if key is not None:
            krelx, krely = (self & key).fetch1("relx", "rely")
            ax.scatter(krely, krelx, label='key')
            ax.legend()
        ax.set(xlabel="rely", ylabel="relx")
        ax.set_aspect(aspect="equal", adjustable="datalim")
        plt.show()


class RetinalFieldLocationTemplate(dj.Computed):
    database = ""  # hack to suppress DJ error

    @property
    def definition(self):
        definition = """
        # location of the recorded fields relative to the optic disk
        # XCoord_um is the relative position from back towards curtain, i.e. larger XCoord_um means closer curtain
        # YCoord_um is the relative position from left to right, i.e. larger YCoord_um means more right
        
        -> self.relativefieldlocalation_table
        ---
        ventral_dorsal_pos_um       :float      # position on the ventral-dorsal axis, greater 0 means dorsal
        temporal_nasal_pos_um       :float      # position on the temporal-nasal axis, greater 0 means nasal
        """
        return definition

    relativefieldlocalation_table = PlaceholderTable
    expinfo_table = PlaceholderTable

    def make(self, key):
        relx, rely = (self.relativefieldlocalation_table() & key).fetch1('relx', 'rely')
        eye, prepwmorient = (self.expinfo_table() & key).fetch1('eye', 'prepwmorient')

        ventral_dorsal_pos_um, temporal_nasal_pos_um = get_retinal_position(
            rel_xcoord_um=relx, rel_ycoord_um=rely, rotation=prepwmorient, eye=eye)

        rfl_key = key.copy()
        rfl_key['ventral_dorsal_pos_um'] = ventral_dorsal_pos_um
        rfl_key['temporal_nasal_pos_um'] = temporal_nasal_pos_um
        self.insert1(rfl_key)

    def plot(self, key=None):
        temporal_nasal_pos_um, ventral_dorsal_pos_um = self.fetch("temporal_nasal_pos_um", "ventral_dorsal_pos_um")
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.scatter(temporal_nasal_pos_um, ventral_dorsal_pos_um, label='all')
        if key is not None:
            ktemporal_nasal_pos_um, kventral_dorsal_pos_um = (self & key).fetch1(
                "temporal_nasal_pos_um", "ventral_dorsal_pos_um")
            ax.scatter(ktemporal_nasal_pos_um, kventral_dorsal_pos_um, label='key')
            ax.legend()
        ax.set(xlabel="temporal_nasal_pos_um", ylabel="ventral_dorsal_pos_um")
        ax.set_aspect(aspect="equal", adjustable="datalim")
        plt.show()
