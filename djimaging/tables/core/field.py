import os
from copy import deepcopy
import datajoint as dj
import numpy as np
from matplotlib import pyplot as plt
import h5py

from djimaging.utils.data_utils import load_h5_table
from djimaging.utils.datafile_utils import get_filename_info
from djimaging.utils.scanm_utils import get_pixel_size_um
from djimaging.utils.dj_utils import PlaceholderTable


class FieldTemplate(dj.Computed):
    database = ""  # hack to suppress DJ error

    @property
    def definition(self):
        definition = """
        # Recording fields
        -> self.experiment_table
        field   :varchar(255)          # string identifying files corresponding to field
        """
        return definition

    experiment_table = PlaceholderTable
    userinfo_table = PlaceholderTable

    class Zstack(dj.Part):
        @property
        def definition(self):
            definition = """
            #was there a zstack in the field
            -> master
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
            -> master
            ---
            fromfile    :varchar(255)   # from which file the roi mask was extracted from
            roi_mask    :longblob       # roi mask for the recording field
            """
            return definition

    class FieldInfo(dj.Part):
        @property
        def definition(self):
            definition = """
            # Field information
            -> master
            ---
            fromfile: varchar(255)  # info extracted from which file?
            absx: float  # absolute position in the x axis as recorded by ScanM
            absy: float  # absolute position in the y axis as recorded by ScanM
            absz: float  # absolute position in the z axis as recorded by ScanM
            nxpix: int  # number of pixels in x
            nypix: int  # number of pixels in y
            nxpix_offset: int  # number of offset pixels in x
            nxpix_retrace: int  # number of retrace pixels in x
            pixel_size_um :float  # width / height of a pixel in um
            stack_average: longblob  # Average of the data stack for visualization
            """
            return definition

    def make(self, key):
        self.__add_experiment_fields(key)

    def rescan_filesystem(self, restrictions=None, verbose=0):

        if restrictions is None:
            restrictions = dict()

        for row in (self.experiment_table() & restrictions):
            key = dict(experimenter=row['experimenter'], date=row['date'], exp_num=row['exp_num'])
            if verbose:
                print('Adding fields for:', key)
            self.__add_experiment_fields(key, only_new=True, verbose=verbose)

    def __add_experiment_fields(self, key, only_new=False, verbose=0):

        pre_data_path = os.path.join(
            (self.experiment_table() & key).fetch1('header_path'),
            (self.userinfo_table() & key).fetch1("pre_data_dir"))
        assert os.path.exists(pre_data_path), f"Error: Data folder does not exist: {pre_data_path}"

        if verbose:
            print("--> Processing fields in:", pre_data_path)

        user_dict = (self.userinfo_table() & key).fetch1()

        # Collect all files belonging to this experiment
        field2info = scan_fields_and_files(pre_data_path, user_dict=user_dict, verbose=verbose)

        # Remove opticdisk and outline recordings
        remove_fields = []
        for field in field2info.keys():
            if field.lower() in user_dict['opticdisk_alias'].split('_') + user_dict['outline_alias'].split('_'):
                remove_fields.append(field)
        for remove_field in remove_fields:
            field2info.pop(remove_field)

        # Go through remaining fields and add them
        for field, info in field2info.items():
            exists = len((self & key & dict(field=field)).fetch()) > 0
            if only_new and exists:
                if verbose > 1:
                    print(f"\tSkipping field {field} with files: {info['files']}")
                continue

            if verbose:
                print(f"\tAdding field: {field} with files: {info['files']}")
            self.__add_field(key=key, field=field, files=info['files'])

    def __add_field(self, key, field, files):
        assert field is not None

        pre_data_path = os.path.join(
            (self.experiment_table() & key).fetch1('header_path'),
            (self.userinfo_table() & key).fetch1("pre_data_dir"))
        assert os.path.exists(pre_data_path), f"Error: Data folder does not exist: {pre_data_path}"

        data_stack_name = (self.userinfo_table() & key).fetch1("data_stack_name")
        setupid = (self.experiment_table.ExpInfo() & key).fetch1("setupid")

        roi_mask, file = get_field_roi_mask(pre_data_path, files)

        field_key, fieldinfo_key, zstack_key = load_scan_info(
            key=key, field=field, pre_data_path=pre_data_path, file=file,
            data_stack_name=data_stack_name, setupid=setupid)

        # subkey for adding Fields to RoiMask
        roimask_key = deepcopy(field_key)
        roimask_key["fromfile"] = file if roi_mask.size > 0 else ''
        roimask_key["roi_mask"] = roi_mask

        self.insert1(field_key, allow_direct_insert=True)
        (self.RoiMask & field_key).insert1(roimask_key, allow_direct_insert=True)
        (self.Zstack & field_key).insert1(zstack_key, allow_direct_insert=True)
        (self.FieldInfo & field_key).insert1(fieldinfo_key, allow_direct_insert=True)

    def plot1(self, key):
        fig, axs = plt.subplots(1, 2, figsize=(10, 3.5))
        stack_average = (self.FieldInfo() & key).fetch1("stack_average").T
        roi_mask = (self.RoiMask() & key).fetch1("roi_mask").T
        axs[0].imshow(stack_average)
        axs[0].set(title='stack_average')

        if roi_mask.size > 0:
            roi_mask_im = axs[1].imshow(roi_mask, cmap='jet')
            plt.colorbar(roi_mask_im, ax=axs[1])
            axs[1].set(title='roi_mask')
        plt.show()


def scan_fields_and_files(pre_data_path, user_dict, verbose=0) -> dict:
    """Return a dictonary that maps fields to their respective files"""

    loc_mapper = {k: v for k, v in user_dict.items() if k.endswith('loc')}

    field2info = dict()

    for file in sorted(os.listdir(pre_data_path)):
        if file.startswith('.') or not file.endswith('.h5'):
            continue

        datatype, animal, region, field, stimulus, condition = get_filename_info(file, **loc_mapper)

        if field is None:
            if verbose:
                print(f"\tSkipping file with unrecognized field: {file}")
            continue

        # Create new or check for inconsistencies
        if field not in field2info:
            field2info[field] = dict(files=[], region=region)
        else:
            assert field2info[field]['region'] == region, f"{field2info[field]['region']} vs. {region}"

        # Add file
        field2info[field]['files'].append(file)
    return field2info


def get_field_roi_mask(pre_data_path, files):
    # TODO: makes this prefer chirp masks without being so harcoded and ugly

    sorted_files = sorted(files)

    sort_index = np.zeros(len(sorted_files))
    for i, file in enumerate(sorted_files):
        if 'chirp' in file:
            sort_index[i] = -10
        if 'mb' in file or 'os' in file or 'movingbar' in file or 'mb' in file:
            sort_index[i] = -9

    sorted_files = np.array(sorted_files)[np.argsort(sort_index)]

    for file in sorted_files:
        with h5py.File(os.path.join(pre_data_path, file), 'r', driver="stdio") as h5_file:
            if 'rois' in [k.lower() for k in h5_file.keys()]:
                for h5_keys in h5_file.keys():
                    if h5_keys.lower() == 'rois':
                        roi_mask = np.copy(h5_file[h5_keys])
                        break
                else:
                    raise Exception('This should not happen')
                break
    else:
        raise ValueError(f'No ROI mask found in any of the {files}')

    return roi_mask, file


def load_scan_info(key, field, pre_data_path, file, data_stack_name, setupid):
    # TODO: Clean this

    # Get parameters
    wparamsnum = load_h5_table('wParamsNum', filename=os.path.join(pre_data_path, file))

    nxpix_offset = int(wparamsnum["User_nXPixLineOffs"])
    nxpix_retrace = int(wparamsnum["User_nPixRetrace"])
    nxpix = int(wparamsnum["User_dxPix"] - nxpix_retrace - nxpix_offset)
    nypix = int(wparamsnum["User_dyPix"])
    pixel_size_um = get_pixel_size_um(zoom=wparamsnum["Zoom"], setupid=setupid, nypix=nypix)

    # Get stack
    with h5py.File(os.path.join(pre_data_path, file), 'r', driver="stdio") as h5_file:
        stack = np.copy(h5_file[data_stack_name])

    assert stack.ndim == 3, 'Stack does not match expected shape'
    assert stack.shape[:2] == (nxpix, nypix), f'Stack shape error: {stack.shape} vs {(nxpix, nypix)}'

    stack_average = np.mean(stack, 2)

    # keys
    field_key = deepcopy(key)
    field_key["field"] = field

    # subkey for fieldinfo
    fieldinfo_key = deepcopy(field_key)
    fieldinfo_key["fromfile"] = file
    fieldinfo_key["absx"] = wparamsnum['XCoord_um']
    fieldinfo_key["absy"] = wparamsnum['YCoord_um']
    fieldinfo_key["absz"] = wparamsnum['ZCoord_um']
    fieldinfo_key["nxpix"] = nxpix
    fieldinfo_key["nxpix_offset"] = nxpix_offset
    fieldinfo_key["nxpix_retrace"] = nxpix_retrace
    fieldinfo_key["nypix"] = nypix
    fieldinfo_key["pixel_size_um"] = pixel_size_um
    fieldinfo_key['stack_average'] = stack_average

    # subkey for adding Fields to ZStack
    zstack_key = deepcopy(field_key)
    zstack_key["zstack"] = 1 if wparamsnum['User_ScanType'] == 11 else 0
    zstack_key["zstep"] = wparamsnum['ZStep_um']

    return field_key, fieldinfo_key, zstack_key
