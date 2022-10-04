import os
import random
from copy import deepcopy

import datajoint as dj
import h5py
import numpy as np

from djimaging.utils.alias_utils import check_shared_alias_str
from djimaging.utils.datafile_utils import get_filename_info
from djimaging.utils.dj_utils import PlaceholderTable
from djimaging.utils.plot_utils import plot_field
from djimaging.utils.scanm_utils import get_pixel_size_xy_um, load_ch0_ch1_stacks_from_h5


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
            absx: float  # absolute position of the center (of the cropped field) in the x axis as recorded by ScanM
            absy: float  # absolute position of the center (of the cropped field) in the y axis as recorded by ScanM
            absz: float  # absolute position of the center (of the cropped field) in the z axis as recorded by ScanM
            nxpix: int  # number of pixels in x
            nypix: int  # number of pixels in y
            nzpix: int  # number of pixels in z
            nxpix_offset: int  # number of offset pixels in x
            nxpix_retrace: int  # number of retrace pixels in x
            pixel_size_um :float  # width / height of a pixel in um
            ch0_average :longblob # Stack median of channel 0
            ch1_average :longblob # Stack median of channel 1
            """
            return definition

    def make(self, key):
        self.__add_experiment_fields(key, only_new=False, verboselvl=0, suppress_errors=False)

    def rescan_filesystem(self, restrictions: dict = None, verboselvl: int = 0, suppress_errors: bool = False):
        """
        Scan filesystem for new fields and add them to the database.
        :param restrictions: Restrictions for new fields.
        :param verboselvl: Defines level of output.
        :param suppress_errors:
        """
        if restrictions is None:
            restrictions = dict()

        for row in (self.experiment_table() & restrictions):
            key = dict(experimenter=row['experimenter'], date=row['date'], exp_num=row['exp_num'])
            self.__add_experiment_fields(key, only_new=True, verboselvl=verboselvl, suppress_errors=suppress_errors)

    def __add_experiment_fields(self, key, only_new, verboselvl, suppress_errors):

        header_path = (self.experiment_table() & key).fetch1('header_path')
        pre_data_dir = (self.userinfo_table() & key).fetch1("pre_data_dir")
        pre_data_path = os.path.join(header_path, pre_data_dir)
        assert os.path.exists(pre_data_path), f"Error: Data folder does not exist: {pre_data_path}"

        if verboselvl > 0:
            print("Processing fields in:", pre_data_path)

        user_dict = (self.userinfo_table() & key).fetch1()

        # Collect all files belonging to this experiment
        field2info = scan_fields_and_files(pre_data_path, user_dict=user_dict, verbose=verboselvl > 0)

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
                if verboselvl > 1:
                    print(f"\tSkipping field {field} with files: {info['files']}")
                continue

            if verboselvl > 0:
                print(f"\tAdding field: {field} with files: {info['files']}")

            try:
                self.__add_field(key=key, field=field, files=info['files'], verboselvl=verboselvl)
            except Exception as e:
                if suppress_errors:
                    print("Suppressed Error:", e, '\n\tfor key:', key)
                else:
                    raise e

    def __add_field(self, key, field, files, verboselvl):
        assert field is not None

        header_path = (self.experiment_table() & key).fetch1('header_path')
        pre_data_dir = (self.userinfo_table() & key).fetch1("pre_data_dir")
        pre_data_path = os.path.join(header_path, pre_data_dir)

        mask_alias = (self.userinfo_table() & key).fetch1("mask_alias")
        highres_alias = (self.userinfo_table() & key).fetch1("highres_alias")

        assert os.path.exists(pre_data_path), f"Error: Data folder does not exist: {pre_data_path}"

        setupid = (self.experiment_table.ExpInfo() & key).fetch1("setupid")

        roi_mask, roi_file = load_field_roi_mask(
            pre_data_path, files, mask_alias=mask_alias, highres_alias=highres_alias)

        if verboselvl > 1:
            print(f"\t\tUsing roi_mask from {roi_file}")

        field_key, fieldinfo_key, zstack_key = load_scan_info(
            key=key, field=field, pre_data_path=pre_data_path, file=roi_file, setupid=setupid)

        # subkey for adding Fields to RoiMask
        roimask_key = deepcopy(field_key)
        roimask_key["fromfile"] = os.path.join(pre_data_path, roi_file) if roi_mask.size > 0 else ''
        roimask_key["roi_mask"] = roi_mask

        self.insert1(field_key, allow_direct_insert=True)
        (self.RoiMask & field_key).insert1(roimask_key, allow_direct_insert=True)
        (self.Zstack & field_key).insert1(zstack_key, allow_direct_insert=True)
        (self.FieldInfo & field_key).insert1(fieldinfo_key, allow_direct_insert=True)

    def plot1(self, key=None, figsize=(16, 4)):
        if key is not None:
            key = {k: v for k, v in key.items() if k in self.primary_key}
        else:
            key = random.choice(self.fetch(*self.primary_key, as_dict=True))

        ch0_average = (self.FieldInfo() & key).fetch1("ch0_average").T
        ch1_average = (self.FieldInfo() & key).fetch1("ch1_average").T
        roi_mask = (self.RoiMask() & key).fetch1("roi_mask").T

        plot_field(ch0_average, ch1_average, roi_mask, title=key, figsize=figsize)


def scan_fields_and_files(pre_data_path: str, user_dict: dict, verbose: bool = False) -> dict:
    """Return a dictionary that maps fields to their respective files"""

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


def load_field_roi_mask(pre_data_path, files, mask_alias='', highres_alias=''):
    """Load ROI mask for field"""
    files = np.array(files)
    penalties = np.full(files.size, len(mask_alias.split('_')))

    for i, file in enumerate(files):
        if check_shared_alias_str(highres_alias, file.lower().replace('.h5', '')):
            penalties[i] = len(mask_alias.split('_')) + 1

        else:
            for penalty, alias in enumerate(mask_alias.split('_')):
                if alias.lower() in file.lower().replace('.h5', '').split('_'):
                    penalties[i] = penalty

    sorted_files = files[np.argsort(penalties)]

    for file in sorted_files:
        filepath = os.path.join(pre_data_path, file)

        try:
            with h5py.File(filepath, 'r', driver="stdio") as h5_file:
                roi_keys = [k for k in h5_file.keys() if 'rois' in k.lower()]
                if len(roi_keys) == 0:
                    continue
                elif len(roi_keys) == 1:
                    roi_mask = np.copy(h5_file[roi_keys[0]])
                    return roi_mask, file
                else:
                    raise KeyError(f'Multiple ROI masks found in {filepath}')
        except:
            pass
    else:
        raise ValueError(f'No ROI mask found in any of the {pre_data_path}: {files}')


def load_scan_info(key, field, pre_data_path, file, setupid):

    filepath = os.path.join(pre_data_path, file)

    ch0_stack, ch1_stack, wparams = \
        load_ch0_ch1_stacks_from_h5(filepath, ch0_name='wDataCh0', ch1_name='wDataCh1')

    nxpix = wparams["user_dxpix"] - wparams["user_npixretrace"] - wparams["user_nxpixlineoffs"]
    nypix = wparams["user_dypix"]
    nzpix = wparams["user_dzpix"]

    pixel_size_um = get_pixel_size_xy_um(zoom=wparams["zoom"], setupid=setupid, npix=nxpix)

    # keys
    field_key = deepcopy(key)
    field_key["field"] = field

    # subkey for fieldinfo
    fieldinfo_key = deepcopy(field_key)
    fieldinfo_key["fromfile"] = filepath
    fieldinfo_key["absx"] = wparams['xcoord_um']
    fieldinfo_key["absy"] = wparams['ycoord_um']
    fieldinfo_key["absz"] = wparams['zcoord_um']
    fieldinfo_key["nypix"] = nypix
    fieldinfo_key["nxpix"] = nxpix
    fieldinfo_key["nzpix"] = nzpix
    fieldinfo_key["nxpix_offset"] = wparams["user_nxpixlineoffs"]
    fieldinfo_key["nxpix_retrace"] = wparams["user_npixretrace"]
    fieldinfo_key["pixel_size_um"] = pixel_size_um
    fieldinfo_key['ch0_average'] = np.mean(ch0_stack, 2)
    fieldinfo_key['ch1_average'] = np.mean(ch1_stack, 2)

    # subkey for adding Fields to ZStack
    zstack_key = deepcopy(field_key)
    zstack_key["zstack"] = 1 if wparams['user_scantype'] == 11 else 0
    zstack_key["zstep"] = wparams['zstep_um']

    return field_key, fieldinfo_key, zstack_key
