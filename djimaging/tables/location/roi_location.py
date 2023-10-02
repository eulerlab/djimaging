import warnings
from abc import abstractmethod
import datajoint as dj
import numpy as np
from matplotlib import pyplot as plt
from djimaging.utils.scanm_utils import get_retinal_position, get_rel_roi_pos


class RelativeRoiLocationTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = """
        # location of the recorded field center relative to the optic disk
        # XCoord_um is the relative position from back towards curtain, i.e. larger XCoord_um means closer to curtain
        # YCoord_um is the relative position from left to right, i.e. larger YCoord_um means more right

        -> self.relative_field_location_table
        -> self.roi_table
        ---
        relx   :float      # XCoord_um relative to the optic disk
        rely   :float      # YCoord_um relative to the optic disk
        relz   :float      # ZCoord_um relative to the optic disk
        """
        return definition

    @property
    def key_source(self):
        try:
            return self.relative_field_location_table.proj() * self.roi_table.proj()
        except (AttributeError, TypeError):
            pass

    @property
    @abstractmethod
    def relative_field_location_table(self):
        pass

    @property
    @abstractmethod
    def roi_table(self):
        pass

    @property
    @abstractmethod
    def roi_mask_table(self):
        pass

    @property
    @abstractmethod
    def field_table(self):
        pass

    @property
    @abstractmethod
    def presentation_table(self):
        pass

    def make(self, key):
        roi_id = (self.roi_table & key).fetch1('roi_id')
        roi_mask = (self.roi_mask_table & key).fetch1('roi_mask')
        pixel_size_um, scan_type = (self.field_table & key).fetch('pixel_size_um', 'scan_type')

        ang_deg_list = (self.presentation_table.ScanInfo() & (self.field_table & key)).fetch('angle_deg')
        assert np.unique(ang_deg_list).size == 1, f'Found different angles for different presentations: {ang_deg_list}'
        ang_deg = ang_deg_list[0]

        # Add Roi offset to field offset
        if scan_type == 'xy':
            field_relx, field_rely, field_relz = (self.relative_field_location_table & key).fetch1(
                'relx', 'rely', 'relz')
            dx_um, dy_um = get_rel_roi_pos(roi_id, roi_mask, pixel_size_um, ang_deg=ang_deg)
            roi_relx = field_relx - dy_um
            roi_rely = field_rely - dx_um
            roi_relz = field_relz
        else:
            raise NotImplementedError(scan_type)

        roi_key = key.copy()
        roi_key['relx'] = roi_relx
        roi_key['rely'] = roi_rely
        roi_key['relz'] = roi_relz

        self.insert1(roi_key)

    def plot(self, key=None, view='igor_local'):
        relx, rely = self.fetch("relx", "rely")
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.scatter(rely, relx, label='all', s=1, alpha=0.5)
        if key is not None:
            krelx, krely = (self & key).fetch1("relx", "rely")
            ax.scatter(krely, krelx, label='key')
            ax.legend()
        ax.set(xlabel="rely", ylabel="relx")

        if view == 'igor_local':
            ax.invert_xaxis()
            ax.invert_yaxis()
        elif view == 'igor_setup':
            ax.invert_yaxis()
        else:
            raise NotImplementedError(view)

        ax.set_aspect(aspect="equal", adjustable="datalim")

        plt.show()


class RetinalRoiLocationTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = """
        -> self.relative_roi_location_table
        ---
        ventral_dorsal_pos_um       :float      # position on the ventral-dorsal axis, greater 0 means dorsal
        temporal_nasal_pos_um       :float      # position on the temporal-nasal axis, greater 0 means nasal
        """
        return definition

    @property
    def key_source(self):
        try:
            return self.relative_roi_location_table.proj()
        except (AttributeError, TypeError):
            pass

    @property
    @abstractmethod
    def relative_roi_location_table(self):
        pass

    @property
    @abstractmethod
    def expinfo_table(self):
        pass

    def make(self, key):
        relx, rely = (self.relative_roi_location_table() & key).fetch1('relx', 'rely')
        eye, prepwmorient = (self.expinfo_table() & key).fetch1('eye', 'prepwmorient')

        if prepwmorient == -1:
            warnings.warn(f'prepwmorient is -1 for {key}. Skipping.')
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
        ax.scatter(temporal_nasal_pos_um, ventral_dorsal_pos_um, label='all', s=1, alpha=0.5)
        if key is not None:
            ktemporal_nasal_pos_um, kventral_dorsal_pos_um = (self & key).fetch1(
                "temporal_nasal_pos_um", "ventral_dorsal_pos_um")
            ax.scatter(ktemporal_nasal_pos_um, kventral_dorsal_pos_um, label='key')
            ax.legend()
        ax.set(xlabel="temporal_nasal_pos_um", ylabel="ventral_dorsal_pos_um")
        ax.set_aspect(aspect="equal", adjustable="datalim")
        plt.show()
