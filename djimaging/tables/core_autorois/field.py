import os
from abc import abstractmethod
from copy import deepcopy
from typing import Optional

import datajoint as dj
import numpy as np

from djimaging.utils.mask_utils import sort_roi_mask_files
from djimaging.utils import scanm_utils

from djimaging.utils.datafile_utils import scan_field_file_dicts, clean_field_file_dicts
from djimaging.utils.dj_utils import get_primary_key
from djimaging.utils.plot_utils import plot_field


class FieldTemplate(dj.Computed):
    database = ""
    _include_conditions = False

    @property
    def definition(self):
        definition_head = """
        # Recording fields
        -> self.experiment_table
        -> self.raw_params_table
        field   :varchar(16)          # string identifying files corresponding to field
        """

        if self._include_conditions:
            definition_head += "        condition    :varchar(16)    # condition (pharmacological or other)\n"

        definition_body = """
        ---
        fromfile: varchar(191)  # info extracted from which file?
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
        definition = definition_head + definition_body

        return definition

    @property
    def key_source(self):
        try:
            return self.experiment_table.proj() * self.raw_params_table.proj()
        except (AttributeError, TypeError):
            pass

    @property
    @abstractmethod
    def userinfo_table(self):
        pass

    @property
    @abstractmethod
    def raw_params_table(self):
        pass

    @property
    @abstractmethod
    def experiment_table(self):
        pass

    class StackAverages(dj.Part):
        @property
        def definition(self):
            definition = """
            # Stack median (over time of the available channels)
            -> master
            ch_name : varchar(16)  # name of the channel
            ---
            ch_average :longblob  # Stack median over time
            """
            return definition

    def make(self, key, verboselvl=0):
        self.add_experiment_fields(key, only_new=False, verboselvl=verboselvl, suppress_errors=False,
                                   restr_headers=None)

    def rescan_filesystem(self, restrictions: dict = None, verboselvl: int = 0, suppress_errors: bool = False,
                          restr_headers: Optional[list] = None):
        """Scan filesystem for new fields and add them to the database."""
        if restrictions is None:
            restrictions = dict()

        for key in (self.key_source & restrictions):
            self.add_experiment_fields(
                key, restr_headers=restr_headers, only_new=True, verboselvl=verboselvl, suppress_errors=suppress_errors)

    def add_experiment_fields(self, key, only_new: bool, verboselvl: int, suppress_errors: bool,
                              restr_headers: Optional[list] = None):

        if restr_headers is not None:
            header_path = (self.experiment_table & key).fetch1('header_path')
            if header_path not in restr_headers:
                if verboselvl > 1:
                    print(f"\tSkipping header_path `{header_path}` because of restriction")
                return

        from_raw_data = (self.raw_params_table & key).fetch1('from_raw_data')

        field_dicts = self.compute_field_dicts(key=key, from_raw_data=from_raw_data, verboselvl=verboselvl)

        if verboselvl > 2:
            print('key=\n', key, '\nfield_dicts=\n', field_dicts)

        # Go through remaining fields and add them
        for field_id, info in field_dicts.items():
            field = field_id[0] if isinstance(field_id, tuple) else field_id

            if not self._include_conditions:
                condition = None
                exists = len((self & key & dict(field=field)).fetch()) > 0
            else:
                condition = field_id[1]
                exists = len((self & key & dict(field=field, condition=condition)).fetch()) > 0

            if only_new and exists:
                if verboselvl > 1:
                    print(f"\tSkipping field `{field_id}` with files: {info['files']}")
                continue

            if verboselvl > 0:
                print(f"\tAdding field: `{field_id}` with files: {info['files']}")

            try:
                self.add_field(key=key, field=field, files=info['files'],
                               from_raw_data=from_raw_data, condition=condition, verboselvl=verboselvl)
            except Exception as e:
                if suppress_errors:
                    print("Suppressed Error:", e, '\n\tfor key:\n', key, '\n\t', field, '\n\t', info['files'])
                else:
                    raise e

    def compute_field_dicts(self, key, from_raw_data, verboselvl=0):
        data_path = os.path.join(
            (self.experiment_table() & key).fetch1('header_path'),
            (self.userinfo_table() & key).fetch1("raw_data_dir" if from_raw_data else "pre_data_dir"))
        user_dict = (self.userinfo_table() & key).fetch1()

        assert os.path.exists(data_path), f"Error: Data folder does not exist: {data_path}"

        if verboselvl > 0:
            print("Processing fields in:", data_path)

        field_dicts = scan_field_file_dicts(
            data_path, user_dict=user_dict, from_raw_data=from_raw_data, incl_condition=self._include_conditions,
            verbose=verboselvl > 0)
        field_dicts = clean_field_file_dicts(field_dicts, user_dict=user_dict)
        return field_dicts

    def add_field(self, key, field, files, from_raw_data, verboselvl, condition=None):
        data_folder = os.path.join(
            (self.experiment_table() & key).fetch1('header_path'),
            (self.userinfo_table() & key).fetch1("raw_data_dir" if from_raw_data else "pre_data_dir"))
        assert os.path.exists(data_folder), f"Error: Data folder does not exist: {data_folder}"

        mask_alias, highres_alias, data_stack_name, alt_stack_name = (self.userinfo_table() & key).fetch1(
            "mask_alias", "highres_alias", "data_stack_name", "alt_stack_name")
        setupid = (self.experiment_table().ExpInfo() & key).fetch1("setupid")

        file_paths = [os.path.join(data_folder, file) for file in files]

        field_key, avg_keys = load_scan_info(
            key=key, field=field, files=file_paths, from_raw_data=from_raw_data,
            ch_names=(data_stack_name, alt_stack_name),
            mask_alias=mask_alias, highres_alias=highres_alias, setupid=setupid)

        if self._include_conditions:
            field_key['condition'] = condition
            for avg_key in avg_keys:
                avg_key['condition'] = condition

        if verboselvl > 2:
            print(f"For key=\n{key} add \nfield_key=\n{field_key}")

        self.insert1(field_key, allow_direct_insert=True)

        for avg_key in avg_keys:
            (self.StackAverages & field_key).insert1(avg_key, allow_direct_insert=True)

    def plot1(self, key=None):
        key = get_primary_key(table=self, key=key)
        data_name, alt_name = (self.userinfo_table & key).fetch1('data_stack_name', 'alt_stack_name')
        main_ch_average = (self.StackAverages & key & f'ch_name="{data_name}"').fetch1('ch_average')
        alt_ch_average = (self.StackAverages & key & f'ch_name="{alt_name}"').fetch1('ch_average')
        npixartifact = (self & key).fetch1('npixartifact')

        plot_field(main_ch_average, alt_ch_average, title=key, npixartifact=npixartifact, figsize=(8, 4))


def load_scan_info(key, field, files, from_raw_data, ch_names, mask_alias, highres_alias, setupid) -> (dict, list):
    filepaths = sort_roi_mask_files(
        files=files, mask_alias=mask_alias, highres_alias=highres_alias, suffix='.smp' if from_raw_data else '.h5')

    for i, filepath in enumerate(filepaths):
        try:
            ch_stacks, wparams = scanm_utils.load_stacks(filepath, from_raw_data=from_raw_data, ch_names=ch_names)
            break
        except Exception as e:
            error_msg = f"Failed to load file with error {e}:\n{filepath}"
            if filepath == filepaths[-1]:
                raise OSError(error_msg)
            if input(f"{error_msg}\nTry again for {filepaths[i + 1]}? (y/n)') != 'y'") == 'y':
                continue
            else:
                raise OSError(error_msg)
    else:
        raise OSError(f"Failed to load any of the files:\n{filepaths}")

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

    # get stack avgs
    avg_keys = []
    for name, stack in ch_stacks.items():
        avg_key = deepcopy(base_key)
        avg_key["ch_name"] = name
        avg_key["ch_average"] = np.median(stack, 2)
        avg_keys.append(avg_key)

    return field_key, avg_keys
