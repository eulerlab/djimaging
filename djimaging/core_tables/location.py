import datajoint as dj

from djimaging.utils.scanm_utils import get_retinal_position
from djimaging.utils.dj_utils import PlaceholderTable


class RelativeFieldLocationTemplate(dj.Computed):
    database = ""  # hack to suppress DJ error

    @property
    def definition(self):
        definition = """
        # location of the recorded field center relative to the optic disk
        # XCoord_um is the relative position from back towards curtain, i.e. larger XCoord_um means closer to curtain
        # YCoord_um is the relative position from left to right, i.e. larger YCoord_um means more right
        
        -> self.field_table
        -> self.expinfo_table
        ---
        relx        :float      # XCoord_um relative to the optic disk
        rely        :float      # YCoord_um relative to the optic disk
        relz        :float      # ZCoord_um relative to the optic disk
        """
        return definition

    field_table = PlaceholderTable
    expinfo_table = PlaceholderTable

    @property
    def key_source(self):
        return self.field_table & 'od_flag=0'

    def make(self, key):
        exp_key = {'experimenter': key["experimenter"], 'date': key["date"], 'exp_num': key["exp_num"]}
        od_table = (self.field_table() * self.field_table.FieldInfo() & exp_key & "od_flag=1")

        if len(od_table) > 0:
            print(f'Optic disk h5 file found for {exp_key}')
            assert len(od_table) == 1, 'Multiple OD recordings found'
            odabsx, odabsy, odabsz, odnxpix, odnxpix_offset, odnypix, odpixel_size_um = od_table.fetch1(
                'absx', 'absy', 'absz', 'nxpix', 'nxpix_offset', 'nypix', 'pixel_size_um')

            # Use center of od field
            odx = odabsx + (odnxpix_offset + odnxpix / 2.) * odpixel_size_um
            ody = odabsy + (odnxpix_offset + odnypix / 2.) * odpixel_size_um
            odz = odabsz

        elif (self.expinfo_table() & key).fetch1("od_ini_flag") == 1:
            print(f'Optic disk header information found for {exp_key}')
            odx, ody, odz = (self.expinfo_table() & key).fetch1("odx", "ody", "odz")
        else:
            print(f'No optic disk information found for {exp_key}')
            return

        absx, absy, absz, nxpix, nxpix_offset, nypix, pixel_size_um = (self.field_table.FieldInfo & key).fetch1(
            'absx', 'absy', 'absz', 'nxpix', 'nxpix_offset', 'nypix', 'pixel_size_um')

        # Get center of scan field
        cabsx = absx + (nxpix_offset + nxpix / 2.) * pixel_size_um
        cabsy = absy + (nxpix_offset + nypix / 2.) * pixel_size_um

        loc_key = key.copy()
        loc_key["relx"] = cabsx - odx
        loc_key["rely"] = cabsy - ody
        loc_key["relz"] = absz - odz

        self.insert1(loc_key)


class RetinalFieldLocationTemplate(dj.Computed):
    database = ""  # hack to suppress DJ error

    @property
    def definition(self):
        definition = """
        # location of the recorded fields relative to the optic disk
        # XCoord_um is the relative position from back towards curtain, i.e. larger XCoord_um means closer curtain
        # YCoord_um is the relative position from left to right, i.e. larger YCoord_um means more right
        
        -> self.relativefieldlocalation_table
        -> self.expinfo_table
        ---
        ventral_dorsal_pos_um       :float      # position on the ventral-dorsal axis, greater 0 means dorsal
        temporal_nasal_pos_um       :float      # position on the temporal-nasal axis, greater 0 means nasal
        """
        return definition

    relativefieldlocalation_table = PlaceholderTable
    expinfo_table = PlaceholderTable

    def make(self, key):
        relx, rely = (self.relativefieldlocalation_table & key).fetch1('relx', 'rely')
        eye, prepwmorient = (self.expinfo_table & key).fetch1('eye', 'prepwmorient')

        ventral_dorsal_pos_um, temporal_nasal_pos_um = get_retinal_position(
            rel_xcoord_um=relx, rel_ycoord_um=rely, rotation=prepwmorient, eye=eye)

        rfl_key = key.copy()
        rfl_key['ventral_dorsal_pos_um'] = ventral_dorsal_pos_um
        rfl_key['temporal_nasal_pos_um'] = temporal_nasal_pos_um
        self.insert1(rfl_key)
