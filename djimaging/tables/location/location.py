import warnings
from abc import abstractmethod

import datajoint as dj
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from djimaging.utils.scanm.setup_utils import get_retinal_position


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
