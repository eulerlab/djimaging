import datajoint as dj
from matplotlib import pyplot as plt

from djimaging.utils.scanm_utils import get_retinal_position
from djimaging.utils.dj_utils import PlaceholderTable


class RetinaEdgesTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = """
        # location of the recorded field center relative to the optic disk
        # XCoord_um is the relative position from back towards curtain, i.e. larger XCoord_um means closer to curtain
        # YCoord_um is the relative position from left to right, i.e. larger YCoord_um means more right

        -> self.expinfo_table
        ---
        -> self.field_table.proj(source='field')
        odx  :float    # XCoord_um relative to the optic disk
        ody  :float    # YCoord_um relative to the optic disk
        odz  :float    # ZCoord_um relative to the optic disk
        """
        return definition


class OpticDiskTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = """
        # location of the recorded field center relative to the optic disk
        # XCoord_um is the relative position from back towards curtain, i.e. larger XCoord_um means closer to curtain
        # YCoord_um is the relative position from left to right, i.e. larger YCoord_um means more right

        -> self.expinfo_table
        ---
        -> self.field_table.proj(source='field')
        odx  :float    # XCoord_um relative to the optic disk
        ody  :float    # YCoord_um relative to the optic disk
        odz  :float    # ZCoord_um relative to the optic disk
        """
        return definition

    expinfo_table = PlaceholderTable
    field_table = PlaceholderTable

    def make(self, key):
        od_table = (self.field_table() * self.field_table.FieldInfo() & key & "od_flag=1")

        if len(od_table) > 0:
            print(f'Optic disk h5 file found for {key}')
            assert len(od_table) == 1, 'Multiple OD recordings found'
            absx, absy, absz, nxpix, nxpix_offset, nypix, pixel_size_um = od_table.fetch1(
                'absx', 'absy', 'absz', 'nxpix', 'nxpix_offset', 'nypix', 'pixel_size_um')

            # Use center of od field
            odx = absx + (nxpix_offset + nxpix / 2.) * pixel_size_um
            ody = absy + (nxpix_offset + nypix / 2.) * pixel_size_um
            odz = absz
            source = od_table.fetch1('field')

        elif (self.expinfo_table() & key).fetch1("od_ini_flag") == 1:
            print(f'Optic disk header information found for {key}')
            odx, ody, odz = (self.expinfo_table() & key).fetch1("odx", "ody", "odz")
            source = "header"

        else:
            print(f'No optic disk information found for {key}')
            return

        loc_key = key.copy()
        loc_key["odx"] = odx
        loc_key["ody"] = ody
        loc_key["odz"] = odz
        loc_key["source"] = source

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

    @property
    def key_source(self):
        return self.opticdisk_table() * (self.field_table() & 'od_flag=0')

    def make(self, key):
        od_key = key.copy()
        od_key.pop('field', None)
        print(od_key)

        odx, ody, odz = (self.opticdisk_table() & od_key).fetch1("odx", "ody", "odz")

        absx, absy, absz, nxpix, nxpix_offset, nypix, pixel_size_um = (self.field_table.FieldInfo() & key).fetch1(
            'absx', 'absy', 'absz', 'nxpix', 'nxpix_offset', 'nypix', 'pixel_size_um')

        # Get center of scan field
        cabsx = absx + (nxpix_offset + nxpix / 2.) * pixel_size_um
        cabsy = absy + (nxpix_offset + nypix / 2.) * pixel_size_um

        loc_key = key.copy()
        loc_key["relx"] = cabsx - odx
        loc_key["rely"] = cabsy - ody
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
