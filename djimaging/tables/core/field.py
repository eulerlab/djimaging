import os
from abc import abstractmethod
from copy import deepcopy
from typing import Optional

import datajoint as dj
import numpy as np

from djimaging.utils.scanm import read_utils, setup_utils, wparams_utils

from djimaging.utils.datafile_utils import get_file_info_df
from djimaging.utils.dj_utils import get_primary_key
from djimaging.utils.plot_utils import plot_field


class FieldTemplate(dj.Computed):
    database = ""
    incl_region = True  # Include region as primary key?
    incl_cond1 = False  # Include condition 1 as primary key?
    incl_cond2 = False  # Include condition 2 as primary key?
    incl_cond3 = False  # Include condition 3 as primary key?

    @property
    def definition(self):
        definition = """
        # Recording fields
        -> self.experiment_table
        -> self.raw_params_table
        field   :varchar(255)          # string identifying files corresponding to field
        """

        if self.incl_region:
            definition += "    region   :varchar(255)    # region (e.g. LR or RR)\n"
        if self.incl_cond1:
            definition += "    cond1    :varchar(255)    # condition (pharmacological or other)\n"
        if self.incl_cond2:
            definition += "    cond2    :varchar(255)    # condition (pharmacological or other)\n"
        if self.incl_cond3:
            definition += "    cond3    :varchar(255)    # condition (pharmacological or other)\n"

        definition += """
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
            ch_name : varchar(255)  # name of the channel
            ---
            ch_average :longblob  # Stack median over time
            """
            return definition

    def make(self, key, verboselvl=0):
        self.add_experiment_fields(
            key, only_new=False, verboselvl=verboselvl, suppress_errors=False, restr_headers=None)

    def rescan_filesystem(self, restrictions: dict = None, verboselvl: int = 0, suppress_errors: bool = False,
                          restr_headers: Optional[list] = None):
        """Scan filesystem for new fields and add them to the database."""
        if restrictions is None:
            restrictions = dict()

        for key in (self.key_source & restrictions):
            self.add_experiment_fields(
                key, restr_headers=restr_headers, only_new=True, verboselvl=verboselvl, suppress_errors=suppress_errors)

    @property
    def new_primary_keys(self):
        new_primary_keys = ['field']
        if self.incl_region:
            new_primary_keys.append('region')
        if self.incl_cond1:
            new_primary_keys.append('cond1')
        if self.incl_cond2:
            new_primary_keys.append('cond2')
        if self.incl_cond3:
            new_primary_keys.append('cond3')
        return new_primary_keys

    def load_exp_file_info_df(self, exp_key):
        from_raw_data = (self.raw_params_table & exp_key).fetch1('from_raw_data')
        header_path = (self.experiment_table & exp_key).fetch1('header_path')
        data_folder = (self.userinfo_table & exp_key).fetch1("raw_data_dir" if from_raw_data else "pre_data_dir")
        user_dict = (self.userinfo_table & exp_key).fetch1()

        file_info_df = get_file_info_df(os.path.join(header_path, data_folder), user_dict, from_raw_data)
        file_info_df = file_info_df[file_info_df['field'].notnull() & (file_info_df['kind'] == 'response')]
        file_info_df.sort_values('mask_order', inplace=True, ascending=True)

        # Set defaults
        if self.incl_cond1:
            file_info_df['cond1'].fillna('control', inplace=True)
        if self.incl_cond2:
            file_info_df['cond2'].fillna('control', inplace=True)
        if self.incl_cond3:
            file_info_df['cond3'].fillna('control', inplace=True)

        return file_info_df

    def add_experiment_fields(
            self, exp_key, only_new: bool, verboselvl: int, suppress_errors: bool,
            restr_headers: Optional[list] = None):

        if restr_headers is not None:
            header_path = (self.experiment_table & exp_key).fetch1('header_path')
            if header_path not in restr_headers:
                if verboselvl > 1:
                    print(f"\tSkipping header_path `{header_path}` because of restriction")
                return

        file_info_df = self.load_exp_file_info_df(exp_key)
        field_dfs = file_info_df.groupby(self.new_primary_keys)

        if verboselvl > 0:
            print(f"Found {len(file_info_df)} files in {len(field_dfs)} for key={exp_key}")

        for field_info, field_df in field_dfs:
            field_key = {**dict(zip(self.new_primary_keys, field_info)), **exp_key}

            if only_new and (len((self & field_key).proj()) > 0):
                if verboselvl > 1:
                    print(f"\tSkipping field `{field_key}` because it already exists")
                continue

            if verboselvl > 0:
                print(f"\tAdding field: `{field_key}`")

            try:
                self.add_field(
                    field_key=field_key, filepaths=field_df['filepath'].values, verboselvl=verboselvl)
            except Exception as e:
                if suppress_errors:
                    print("Suppressed Error:", e, '\n\tfor key:\n', field_key, '\n\t', field_df['filepath'])
                else:
                    raise e

    def add_field(self, field_key, filepaths, verboselvl=0):

        if verboselvl > 4:
            print('\t\tAdd field with files:', filepaths)

        from_raw_data = (self.raw_params_table & field_key).fetch1('from_raw_data')
        data_stack_name, alt_stack_name = (self.userinfo_table & field_key).fetch1(
            "data_stack_name", "alt_stack_name")
        setupid = (self.experiment_table().ExpInfo & field_key).fetch1("setupid")

        field_entry, avg_entries = load_field_key_data(
            field_key=field_key, filepaths=filepaths, ch_names=(data_stack_name, alt_stack_name),
            setupid=setupid, from_raw_data=from_raw_data)

        self.insert1(field_entry, allow_direct_insert=True)
        for avg_entry in avg_entries:
            self.StackAverages().insert1(avg_entry, allow_direct_insert=True)

    def plot1(self, key=None):
        key = get_primary_key(table=self, key=key)
        data_name, alt_name = (self.userinfo_table & key).fetch1('data_stack_name', 'alt_stack_name')
        main_ch_average = (self.StackAverages & key & f'ch_name="{data_name}"').fetch1('ch_average')
        alt_ch_average = (self.StackAverages & key & f'ch_name="{alt_name}"').fetch1('ch_average')
        npixartifact = (self & key).fetch1('npixartifact')

        plot_field(main_ch_average, alt_ch_average, title=key, npixartifact=npixartifact, figsize=(8, 4))


def load_field_key_data(field_key, filepaths, from_raw_data, ch_names, setupid) -> (dict, list):
    ch_stacks, wparams, filepath = load_first_file_wo_error(filepaths, from_raw_data, ch_names)

    nxpix = wparams["user_dxpix"] - wparams["user_npixretrace"] - wparams["user_nxpixlineoffs"]
    nypix = wparams["user_dypix"]
    nzpix = wparams["user_dzpix"]

    pixel_size_um = setup_utils.get_pixel_size_xy_um(zoom=wparams["zoom"], setupid=setupid,
                                                     npix=nxpix)
    z_step_um = wparams.get('zstep_um', 0.)
    z_stack_flag = int(wparams['user_scantype'] == 11)
    npixartifact = setup_utils.get_npixartifact(setupid=setupid)
    scan_type = wparams_utils.get_scan_type(wparams)

    # keys
    field_entry = deepcopy(field_key)
    field_entry["fromfile"] = filepath
    field_entry["absx"] = wparams['xcoord_um']
    field_entry["absy"] = wparams['ycoord_um']
    field_entry["absz"] = wparams['zcoord_um']
    field_entry["scan_type"] = scan_type
    field_entry["npixartifact"] = npixartifact
    field_entry["nxpix"] = nxpix
    field_entry["nypix"] = nypix
    field_entry["nzpix"] = nzpix
    field_entry["nxpix_offset"] = wparams["user_nxpixlineoffs"]
    field_entry["nxpix_retrace"] = wparams["user_npixretrace"]
    field_entry["pixel_size_um"] = pixel_size_um
    field_entry["z_step_um"] = z_step_um
    field_entry["z_stack_flag"] = z_stack_flag

    # get stack avgs
    avg_entries = []
    for name, stack in ch_stacks.items():
        avg_entry = deepcopy(field_key)
        avg_entry["ch_name"] = name
        avg_entry["ch_average"] = np.median(stack, 2)
        avg_entries.append(avg_entry)

    return field_entry, avg_entries


def load_first_file_wo_error(filepaths, from_raw_data, ch_names):
    for i, filepath in enumerate(filepaths):
        try:
            ch_stacks, wparams = read_utils.load_stacks(filepath, from_raw_data=from_raw_data, ch_names=ch_names)
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

    return ch_stacks, wparams, filepath
