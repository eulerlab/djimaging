import os
from copy import deepcopy
import datajoint as dj
import numpy as np
import h5py

from ..utils.data_utils import load_h5_table
from ..utils.filename_utils import get_file_info
from ..utils.scanm_utils import get_pixel_size_um


class Field(dj.Computed):
    database = ""  # hack to suppress DJ error

    @property
    def definition(self):
        definition = """
        # Recording fields
        -> self.experiment_table
        field                :varchar(255)          # string identifying files corresponding to field
        ---
        od_flag              :tinyint unsigned      # is this the optic disk recording?
        loc_flag             :tinyint unsigned      # is this a location recording? e.g. the edge
        """
        return definition

    experiment_table = None
    userinfo_table = None

    class Zstack(dj.Part):
        @property
        def definition(self):
            definition = """
            #was there a zstack in the field
            -> Field
            ---
            zstack      :tinyint    #flag marking whether field was a zstack
            zstep       :float      #size of step size in um
            """
            return definition

    class RoiMask(dj.Part):
        @property
        def definition(self):
            definition = """
            # ROI Mask
            -> Field
            ---
            fromfile                   :varchar(255)   # from which file the roi mask was extracted from
            roi_mask                   :longblob       # roi mask for the recording field
            """
            return definition

    class FieldInfo(dj.Part):
        @property
        def definition(self):
            definition = """
            # ROI Mask
            -> Field
            ---
            absx: float  # absolute position in the x axis as recorded by ScanM
            absy: float  # absolute position in the y axis as recorded by ScanM
            absz: float  # absolute position in the z axis as recorded by ScanM
            nxpix: int  # number of pixels in x
            nypix: int  # number of pixels in y
            nxpix_offset: int  # number of offset pixels in x
            nxpix_retrace: int  # number of retrace pixels in x
            pixel_size_um :float  # width / height of a pixel in um
            recording_depth :float  # XY-scan: single element list with IPL depth, XZ: list of ROI depths
            fromfile: varchar(255)  # info extracted from which file?
            """
            return definition

    def make(self, key):
        self.__add_experiment_fields(key)

    def rescan_filesystem(self, restrictions=None, verbose=0):

        if restrictions is None:
            restrictions = dict()

        for row in (self.experiment_table & restrictions):
            key = dict(experimenter=row['experimenter'], date=row['date'], exp_num=row['exp_num'])
            if verbose:
                print('Adding fields for:', key)
            self.__add_experiment_fields(key, only_new=True, verbose=verbose)

    def __add_experiment_fields(self, key, only_new=False, verbose=0):
        pre_data_path = (self.experiment_table & key).fetch1('pre_data_path')

        assert os.path.exists(pre_data_path), f"Error: Data folder does not exist: {pre_data_path}"

        if verbose:
            print("--> Processing fields in:", pre_data_path)

        datatype_loc, animal_loc, region_loc, field_loc, stimulus_loc, pharm_loc = (self.userinfo_table & key).fetch1(
            "datatype_loc", "animal_loc", "region_loc", "field_loc", "stimulus_loc", "pharm_loc")

        # walk through the filenames belonging to this experiment and fetch all fields from filenames
        fields_info = dict()
        for file in sorted(os.listdir(pre_data_path)):
            if file.startswith('.') or not file.endswith('.h5'):
                continue

            datatype, animal, region, field, stimulus, pharm = get_file_info(
                file, datatype_loc, animal_loc, region_loc, field_loc, stimulus_loc, pharm_loc)

            if field is None:
                if verbose:
                    print(f"\tSkipping file with unrecognized field: {file}")
                continue

            if field not in fields_info:
                fields_info[field] = dict(files=[file], region=region)
            else:
                assert fields_info[field]['region'] == region, f"{fields_info[field]['region']} vs. {region}"

            fields_info[field]['files'].append(file)

        for field, info in fields_info.items():
            if only_new and len((self & key & dict(field=field)).fetch()) > 0:
                if verbose > 1:
                    print(f"\tSkipping field {field} with files: {info['files']}")
                continue

            if verbose:
                print(f"\tAdding field: {field} with files: {info['files']}")
            self.__add_field(key=key, field=field, files=info['files'], region=info['region'])

    def __add_field(self, key, field, files, region):
        assert field is not None

        pre_data_path = (self.experiment_table & key).fetch1("pre_data_path")

        if field is not None and (field.startswith('loc') or field.startswith('outline')) or \
                region is not None and (region.startswith('loc') or region.startswith('outline')):
            loc_flag = 1
        else:
            loc_flag = 0

        if field is not None and (field.startswith('od') or field.startswith('opticdis')) or \
                region is not None and (region.startswith('od') or region.startswith('opticdis')):
            od_flag = 1
        else:
            od_flag = 0

        primary_key = deepcopy(key)
        primary_key["field"] = field

        field_key = deepcopy(primary_key)
        field_key["od_flag"] = od_flag
        field_key["loc_flag"] = loc_flag

        # Get more info
        roi_mask = np.zeros(0)
        for file in sorted(files):
            with h5py.File(pre_data_path + file, 'r', driver="stdio") as h5_file:
                if 'rois' in [k.lower() for k in h5_file.keys()]:
                    for h5_keys in h5_file.keys():
                        if h5_keys.lower() == 'rois':
                            roi_mask = np.copy(h5_file[h5_keys])
                    break
        else:
            assert loc_flag or od_flag, f'ROIs not found in {files}'
            file = files[0]

        w_params_num = load_h5_table('wParamsNum', filename=pre_data_path + file)
        absx = w_params_num['XCoord_um']
        absy = w_params_num['YCoord_um']
        absz = w_params_num['ZCoord_um']
        zstep = w_params_num['ZStep_um']
        zstack = w_params_num['User_ScanType']

        setupid = (self.experiment_table.ExpInfo() & key).fetch1('setupid')
        nxpix_offset = w_params_num["User_nXPixLineOffs"]
        nxpix_retrace = w_params_num["User_nPixRetrace"]
        nxpix = w_params_num["User_dxPix"] - nxpix_retrace - nxpix_offset
        nypix = w_params_num["User_dyPix"]
        pixel_size_um = get_pixel_size_um(zoom=w_params_num["Zoom"], setupid=setupid, nypix=nypix)

        # subkey for fieldinfo
        fieldinfo_key = deepcopy(primary_key)
        fieldinfo_key["absx"] = absx
        fieldinfo_key["absy"] = absy
        fieldinfo_key["absz"] = absz
        fieldinfo_key["nxpix"] = nxpix
        fieldinfo_key["nxpix_offset"] = nxpix_offset
        fieldinfo_key["nxpix_retrace"] = nxpix_retrace
        fieldinfo_key["nypix"] = nypix
        fieldinfo_key["pixel_size_um"] = pixel_size_um
        fieldinfo_key["fromfile"] = file
        fieldinfo_key['recording_depth'] = -9999

        # subkey for adding Fields to ZStack
        zstack_key = deepcopy(primary_key)
        zstack_key["zstack"] = 1 if zstack == 11 else 0
        zstack_key["zstep"] = zstep

        # subkey for adding Fields to RoiMask
        roimask_key = deepcopy(primary_key)
        roimask_key["fromfile"] = file if roi_mask.size > 0 else ''
        roimask_key["roi_mask"] = roi_mask

        self.insert1(field_key, allow_direct_insert=True)
        (self.RoiMask & field_key).insert1(roimask_key, allow_direct_insert=True)
        (self.Zstack & field_key).insert1(zstack_key, allow_direct_insert=True)
        (self.FieldInfo & field_key).insert1(fieldinfo_key, allow_direct_insert=True)


class Roi(dj.Computed):
    database = ""  # hack to suppress DJ error

    @property
    def definition(self):
        definition = """
        # ROI information
        -> self.field_table
        roi_id                  :int                # integer id of each ROI
        ---
        roi_size                :int                # number of pixels in ROI
        roi_size_um2            :float              # size of ROI in micrometers squared
        roi_dia_um              :float              # diameter of ROI in micrometers, if it was a circle
        """
        return definition

    field_table = None

    @property
    def key_source(self):
        return self.field_table & 'od_flag=0 and loc_flag=0'

    def make(self, key):
        # load roi_mask for insert roi for a specific experiment and field
        roi_mask = (Field.RoiMask & key).fetch1("roi_mask")
        pixel_size_um = (Field.FieldInfo & key).fetch1("pixel_size_um")

        if not np.any(roi_mask):
            return

        experimenter = key["experimenter"]
        date = key["date"]
        exp_num = key["exp_num"]
        field = key["field"]

        roi_idxs = np.unique(roi_mask)
        roi_idxs = roi_idxs[(roi_idxs != 0) & (roi_idxs != 1)]  # remove background index (0 or 1)
        roi_idxs = roi_idxs[np.argsort(np.abs(roi_idxs))]  # Sort by value

        # add every roi to list and the bulk add to roi table
        roi_keys = []
        for roi_idx in roi_idxs:
            roi_size = np.sum(roi_mask == roi_idx)
            roi_size_um2 = roi_size * pixel_size_um ** 2
            roi_dia_um = 2 * np.sqrt(roi_size_um2 / np.pi)

            roi_keys.append({
                'experimenter': experimenter,
                'date': date,
                'exp_num': exp_num,
                'field': field,
                'roi_id': int(abs(roi_idx)),
                'roi_size': roi_size,
                'roi_size_um2': roi_size_um2,
                'roi_dia_um': roi_dia_um,
            })
        self.insert(roi_keys)
