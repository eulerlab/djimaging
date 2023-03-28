import os
from abc import abstractmethod, ABC
from copy import deepcopy

import datajoint as dj
import numpy as np
from djimaging.utils import scanm_utils

from djimaging.utils.alias_utils import check_shared_alias_str
from djimaging.utils.datafile_utils import get_filename_info
from djimaging.utils.dj_utils import get_primary_key
from djimaging.utils.plot_utils import plot_field


class FieldTemplate(dj.Computed):
    database = ""
    _load_field_roi_masks = True  # Set to False if all roi masks and roi ids should be based on presentation

    @property
    def definition(self):
        definition = """
        # Recording fields
        -> self.experiment_table
        field   :varchar(255)          # string identifying files corresponding to field
        ---
        fromfile: varchar(255)  # info extracted from which file?
        absx: float  # absolute position of the center (of the cropped field) in the x axis as recorded by ScanM
        absy: float  # absolute position of the center (of the cropped field) in the y axis as recorded by ScanM
        absz: float  # absolute position of the center (of the cropped field) in the z axis as recorded by ScanM
        scan_type: enum("xy", "xz", "xyz")  # Type of scan
        npixartifact : int unsigned # Number of pixel with light artifact
        nxpix: int unsigned  # number of pixels in x
        nypix: int unsigned  # number of pixels in y
        nzpix: int unsigned  # number of pixels in z
        nxpix_offset: int unsigned  # number of offset pixels in x
        nxpix_retrace: int unsigned  # number of retrace pixels in x
        pixel_size_um :float  # width / height of a pixel in um
        z_step_um :float  # z-step in um
        z_stack_flag : tinyint unsigned  # Is z-stack?
        """
        return definition

    @property
    @abstractmethod
    def experiment_table(self):
        pass

    @property
    @abstractmethod
    def userinfo_table(self):
        pass

    class StackAverages(dj.Part):
        @property
        def definition(self):
            definition = """
            # Stack median (over time of the available channels)
            -> master
            ch_name : varchar(255)  # name of the channel
            ---
            ch_average :longblob  # Stack median over time
            """
            return definition

    class RoiMask(dj.Part):
        @property
        def definition(self):
            definition = """
            # ROI Mask
            -> master
            ---
            roi_mask    :longblob       # roi mask for the recording field
            """
            return definition

    def make(self, key):
        self.add_experiment_fields(key, only_new=False, verboselvl=0, suppress_errors=False)

    def rescan_filesystem(self, restrictions: dict = None, verboselvl: int = 0, suppress_errors: bool = False):
        """Scan filesystem for new fields and add them to the database."""
        if restrictions is None:
            restrictions = dict()

        for row in (self.experiment_table() & restrictions):
            key = {k: v for k, v in row.items() if k in self.experiment_table.primary_key}
            self.add_experiment_fields(key, only_new=True, verboselvl=verboselvl, suppress_errors=suppress_errors)

    def compute_field2info(self, key, verboselvl=0):
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

        return field2info

    def add_experiment_fields(self, key, only_new: bool, verboselvl: int, suppress_errors: bool):
        field2info = self.compute_field2info(key=key, verboselvl=verboselvl)
        # Go through remaining fields and add them
        for field, info in field2info.items():
            exists = len((self & key & dict(field=field)).fetch()) > 0
            if only_new and exists:
                if verboselvl > 1:
                    print(f"\tSkipping field `{field}` with files: {info['files']}")
                continue

            if verboselvl > 0:
                print(f"\tAdding field: `{field}` with files: {info['files']}")

            try:
                self.add_field(key=key, field=field, files=info['files'], verboselvl=verboselvl)
            except Exception as e:
                if suppress_errors:
                    print("Suppressed Error:", e, '\n\tfor key:', key)
                else:
                    raise e

    def add_field(self, key, field, files, verboselvl):
        assert field is not None

        pre_data_path = os.path.join((self.experiment_table() & key).fetch1('header_path'),
                                     (self.userinfo_table() & key).fetch1("pre_data_dir"))
        assert os.path.exists(pre_data_path), f"Error: Data folder does not exist: {pre_data_path}"

        mask_alias = (self.userinfo_table() & key).fetch1("mask_alias")
        highres_alias = (self.userinfo_table() & key).fetch1("highres_alias")

        setupid = (self.experiment_table().ExpInfo() & key).fetch1("setupid")

        field_key, roimask_key, avg_keys = self.load_new_keys(
            key, field, pre_data_path, files, mask_alias, highres_alias, setupid, verboselvl)

        self.insert1(field_key, allow_direct_insert=True)

        if self._load_field_roi_masks:
            if roimask_key is not None:
                (self.RoiMask & field_key).insert1(roimask_key, allow_direct_insert=True)
        for avg_key in avg_keys:
            (self.StackAverages & field_key).insert1(avg_key, allow_direct_insert=True)

    def load_new_keys(self, key, field, pre_data_path, files, mask_alias, highres_alias, setupid, verboselvl):
        field_key, roimask_key, avg_keys = load_scan_info(
            key=key, field=field, pre_data_path=pre_data_path, files=files,
            mask_alias=mask_alias, highres_alias=highres_alias, setupid=setupid, verboselvl=verboselvl)
        return field_key, roimask_key, avg_keys

    def plot1(self, key=None):
        key = get_primary_key(table=self, key=key)

        roi_masks = (self.RoiMask() & key).fetch("roi_mask")
        data_name, alt_name = (self.userinfo_table & key).fetch1('data_stack_name', 'alt_stack_name')
        main_ch_average = (self.StackAverages & key & f'ch_name="{data_name}"').fetch1('ch_average')
        alt_ch_average = (self.StackAverages & key & f'ch_name="{alt_name}"').fetch1('ch_average')
        npixartifact = (self & key).fetch1('npixartifact')

        plot_field(main_ch_average, alt_ch_average,
                   roi_masks[0] if len(roi_masks) == 1 else None,
                   roi_ch_average=main_ch_average, title=key, npixartifact=npixartifact)


class FieldWithCondition(FieldTemplate, ABC):
    @property
    def definition(self):
        new_line = 'condition    :varchar(255)    # condition (pharmacological or other)\n        '
        d = super().definition
        i_primary = d.find('---')
        assert i_primary > 0
        print(d[:i_primary] + new_line + d[i_primary:])
        return d

    def get_condition(self, data_file, userinfo_key):
        condition_loc = (self.userinfo_table() & userinfo_key).fetch('condition_loc')
        split_string = data_file[:data_file.find(".h5")].split("_")
        condition = split_string[condition_loc] if condition_loc < len(split_string) else 'control'

        return condition

    def load_new_keys(self, key, field, pre_data_path, files, mask_alias, highres_alias, setupid, verboselvl):
        field_key, roimask_key, avg_keys = load_scan_info(
            key=key, field=field, pre_data_path=pre_data_path, files=files,
            mask_alias=mask_alias, highres_alias=highres_alias, setupid=setupid, verboselvl=verboselvl)

        condition = self.get_condition(data_file=field_key["fromfile"], userinfo_key=key)
        for file in files:
            if self.get_condition(data_file=file, userinfo_key=key) != condition:
                ValueError(f"{self.get_condition(data_file=file, userinfo_key=key)} != {condition}, {files}")

        field_key["condition"] = condition
        roimask_key["condition"] = condition
        for avg_key in avg_keys:
            avg_key["condition"] = condition

        return field_key, roimask_key, avg_keys

    def add_experiment_fields(self, key, only_new: bool, verboselvl: int, suppress_errors: bool):
        condition_loc = (self.userinfo_table() & key).fetch('condition_loc')

        field2info = self.compute_field2info(key=key, verboselvl=verboselvl)

        for field, info in field2info.items():
            conditions = []
            for data_file in info['files']:
                split_string = data_file[:data_file.find(".h5")].split("_")
                condition = split_string[condition_loc] if condition_loc < len(split_string) else 'control'
                conditions.append(condition)
            field2info[field]['conditions'] = conditions

        # Go through remaining fields and add them
        for field, info in field2info.items():
            files = np.asarray(info['files'])
            conditions = np.asarray(info['conditions'])

            for condition in np.unique(conditions):
                condition_files = files[conditions == condition]
                exists = len((self & key & dict(field=field, condition=condition)).fetch()) > 0
                if only_new and exists:
                    if verboselvl > 1:
                        print(f"\tSkipping field `{field}` with files: {condition_files}")
                    continue

                if verboselvl > 0:
                    print(f"\tAdding field: `{field}` with files: {condition_files}")

                try:
                    self.add_field(key=key, field=field, files=condition_files, verboselvl=verboselvl)
                except Exception as e:
                    if suppress_errors:
                        print("Suppressed Error:", e, '\n\tfor key:', key)
                    else:
                        raise e


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
        roi_mask = scanm_utils.load_roi_mask_from_h5(filepath=os.path.join(pre_data_path, file), ignore_not_found=True)
        if roi_mask is not None:
            return roi_mask, file
    else:
        raise ValueError(f'No ROI mask found in any file in {pre_data_path}: {files}')


def load_scan_info(key, field, pre_data_path, files, mask_alias, highres_alias, setupid, verboselvl=0) \
        -> (dict, dict, list):
    try:
        roi_mask, file = load_field_roi_mask(
            pre_data_path, files, mask_alias=mask_alias, highres_alias=highres_alias)
        if verboselvl > 1:
            print(f"\t\tUsing roi_mask from {file}")
    except ValueError:
        roi_mask = None
        file = files[0]

    filepath = os.path.join(pre_data_path, file)
    ch_stacks, wparams = scanm_utils.load_stacks_from_h5(filepath, ch_names=('wDataCh0', 'wDataCh1'))

    nxpix = wparams["user_dxpix"] - wparams["user_npixretrace"] - wparams["user_nxpixlineoffs"]
    nypix = wparams["user_dypix"]
    nzpix = wparams["user_dzpix"]

    pixel_size_um = scanm_utils.get_pixel_size_xy_um(zoom=wparams["zoom"], setupid=setupid, npix=nxpix)
    z_step_um = wparams.get('zstep_um', 0.)
    z_stack_flag = int(wparams['user_scantype'] == 11)
    npixartifact = scanm_utils.get_npixartifact(setupid=setupid)
    scan_type = scanm_utils.get_scan_type_from_wparams(wparams)

    # keys
    base_key = deepcopy(key)
    base_key["field"] = field

    field_key = deepcopy(base_key)
    field_key["fromfile"] = filepath
    field_key["absx"] = wparams['xcoord_um']
    field_key["absy"] = wparams['ycoord_um']
    field_key["absz"] = wparams['zcoord_um']
    field_key["scan_type"] = scan_type
    field_key["npixartifact"] = npixartifact
    field_key["nxpix"] = nxpix
    field_key["nypix"] = nypix
    field_key["nzpix"] = nzpix
    field_key["nxpix_offset"] = wparams["user_nxpixlineoffs"]
    field_key["nxpix_retrace"] = wparams["user_npixretrace"]
    field_key["pixel_size_um"] = pixel_size_um
    field_key["z_step_um"] = z_step_um
    field_key["z_stack_flag"] = z_stack_flag

    if roi_mask is not None:
        roimask_key = deepcopy(base_key)
        roimask_key["roi_mask"] = roi_mask
    else:
        roimask_key = None

    # get stack avgs
    avg_keys = []
    for name, stack in ch_stacks.items():
        avg_key = deepcopy(base_key)
        avg_key["ch_name"] = name
        avg_key["ch_average"] = np.median(stack, 2)
        avg_keys.append(avg_key)

    return field_key, roimask_key, avg_keys
