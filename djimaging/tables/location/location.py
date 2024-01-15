import os
import warnings
from abc import abstractmethod

import datajoint as dj
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from djimaging.utils.datafile_utils import get_file_info_df
from djimaging.utils.data_utils import load_h5_table
from djimaging.utils.scanm_utils import get_retinal_position, load_wparams_from_smp


def load_od_pos_from_h5_file(filepath):
    wparams = load_h5_table('wParamsNum', filename=filepath, lower_keys=True)
    odx, ody, odz = wparams['xcoord_um'], wparams['ycoord_um'], wparams['zcoord_um']
    return odx, ody, odz


def load_od_pos_from_smp_file(filepath):
    wparams = load_wparams_from_smp(filepath, return_file=False)
    odx, ody, odz = wparams['xcoord_um'], wparams['ycoord_um'], wparams['zcoord_um']
    return odx, ody, odz


def plot_rel_xy_pos(relx, rely, view='igor_local'):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.scatter(rely, relx, label='all', s=1, alpha=0.5)
    ax.set(xlabel="rely", ylabel="relx")

    if view == 'igor_local':
        pass
    elif view == 'igor_setup':
        ax.invert_xaxis()
    else:
        raise NotImplementedError(view)

    ax.set_aspect(aspect="equal", adjustable="datalim")

    plt.show()


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
        od_fromfile :varchar(255)  # File from which optic disc data was extracted
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
        file_info_df = file_info_df[file_info_df['kind'] == 'od']

        return file_info_df, from_raw_data

    def make(self, key):
        fromfile, odx, ody, odz = None, None, None, None

        file_info_df, from_raw_data = self.load_exp_file_info_df(key)
        if len(file_info_df) > 0:
            fromfile = file_info_df['file'].iloc[0]

        if fromfile is not None:
            if from_raw_data:
                odx, ody, odz = load_od_pos_from_h5_file(fromfile)
            else:
                odx, ody, odz = load_od_pos_from_smp_file(fromfile)
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


class RelativeFieldLocationTemplate(dj.Computed):
    database = ""

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

    @property
    def key_source(self):
        try:
            return self.opticdisk_table.proj() * self.field_table.proj()
        except (AttributeError, TypeError):
            pass

    @property
    @abstractmethod
    def opticdisk_table(self):
        pass

    @property
    @abstractmethod
    def field_table(self):
        pass

    def make(self, key):
        od_key = key.copy()
        od_key.pop('field', None)

        odx, ody, odz = (self.opticdisk_table() & od_key).fetch1("odx", "ody", "odz")
        absx, absy, absz = (self.field_table() & key).fetch1('absx', 'absy', 'absz')

        loc_key = key.copy()
        loc_key["relx"] = absx - odx
        loc_key["rely"] = absy - ody
        loc_key["relz"] = absz - odz

        self.insert1(loc_key)

    def plot(self, view='igor_local'):
        relx, rely = self.fetch("relx", "rely")
        plot_rel_xy_pos(relx, rely, view=view)


class RetinalFieldLocationTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = """
        # location of the recorded fields relative to the optic disk
        # XCoord_um is the relative position from back towards curtain, i.e. larger XCoord_um means closer curtain
        # YCoord_um is the relative position from left to right, i.e. larger YCoord_um means more right

        -> self.relativefieldlocation_table
        ---
        ventral_dorsal_pos_um       :float      # position on the ventral-dorsal axis, greater 0 means dorsal
        temporal_nasal_pos_um       :float      # position on the temporal-nasal axis, greater 0 means nasal
        """
        return definition

    @property
    def key_source(self):
        try:
            return self.relativefieldlocation_table.proj()
        except (AttributeError, TypeError):
            pass

    @property
    @abstractmethod
    def relativefieldlocation_table(self):
        pass

    @property
    @abstractmethod
    def expinfo_table(self):
        pass

    def make(self, key):
        relx, rely = (self.relativefieldlocation_table() & key).fetch1('relx', 'rely')
        eye, prepwmorient = (self.expinfo_table() & key).fetch1('eye', 'prepwmorient')

        if prepwmorient == -1:
            warnings.warn(f'prepwmorient is -1 for {key}. skipping')
            return

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


class RetinalFieldLocationCatTemplate(dj.Computed):
    database = ""

    _ventral_dorsal_key = 'ventral_dorsal_pos_um'
    _temporal_nasal_key = 'temporal_nasal_pos_um'
    _center_dist = 0.

    @property
    def definition(self):
        definition = """
        -> self.retinalfieldlocation_table
        ---
        nt_side  : enum('n', 'c', 't')
        vd_side  : enum('v', 'c', 'd')
        ntvd_side : enum('nv', 'nc', 'nd', 'cv', 'cc', 'cd', 'tv', 'tc', 'td')
        n_tvd_side  : enum('n', 'cv', 'cc', 'cd', 'tv', 'tc', 'td')
        nvd_t_side  : enum('nv', 'nc', 'nd', 'cv', 'cc', 'cd', 't')
        """
        return definition

    @property
    def key_source(self):
        try:
            return self.retinalfieldlocation_table.proj()
        except (AttributeError, TypeError):
            pass

    @property
    @abstractmethod
    def retinalfieldlocation_table(self):
        pass

    def make(self, key):
        ventral_dorsal, temporal_nasal = (self.retinalfieldlocation_table() & key).fetch1(
            self._ventral_dorsal_key, self._temporal_nasal_key)

        if temporal_nasal < -self._center_dist:
            nt_side = 't'
        elif temporal_nasal > self._center_dist:
            nt_side = 'n'
        else:
            nt_side = 'c'

        if ventral_dorsal < -self._center_dist:
            vd_side = 'v'
        elif ventral_dorsal > self._center_dist:
            vd_side = 'd'
        else:
            vd_side = 'c'

        rfl_key = key.copy()
        rfl_key['nt_side'] = nt_side
        rfl_key['vd_side'] = vd_side
        rfl_key['ntvd_side'] = nt_side + vd_side
        rfl_key['n_tvd_side'] = nt_side + vd_side if nt_side == 't' else nt_side
        rfl_key['nvd_t_side'] = nt_side + vd_side if nt_side == 'n' else nt_side
        self.insert1(rfl_key)

    def plot(self, restriction=None):

        if restriction is None:
            restriction = dict()

        df_plot = pd.DataFrame((self * self.retinalfieldlocation_table()) & restriction)

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

        sns.scatterplot(
            ax=ax, data=df_plot, x=self._temporal_nasal_key, y=self._ventral_dorsal_key, hue='ntvd_side')

        ax.set(xlabel="temporal_nasal", ylabel="ventral_dorsal")
        ax.set_aspect(aspect="equal", adjustable="datalim")
        plt.show()
