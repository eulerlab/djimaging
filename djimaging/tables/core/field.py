import os
from abc import abstractmethod
from copy import deepcopy
from typing import Optional

import datajoint as dj
import numpy as np

from djimaging.utils.filesystem_utils import get_file_info_df
from djimaging.utils.dj_utils import get_primary_key
from djimaging.utils.plot_utils import plot_field
from djimaging.utils.scanm.recording import ScanMRecording


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
        field   :varchar(32)          # string identifying files corresponding to field
        """

        if self.incl_region:
            definition += "    region   :varchar(16)    # region (e.g. LR or RR)\n"
        if self.incl_cond1:
            definition += "    cond1    :varchar(16)    # condition (pharmacological or other)\n"
        if self.incl_cond2:
            definition += "    cond2    :varchar(16)    # condition (pharmacological or other)\n"
        if self.incl_cond3:
            definition += "    cond3    :varchar(16)    # condition (pharmacological or other)\n"

        definition += """
        ---
        field_data_file: varchar(191)  # info extracted from which file?
        absx: float  # absolute position of the center (of the cropped field) in the x axis as recorded by ScanM
        absy: float  # absolute position of the center (of the cropped field) in the y axis as recorded by ScanM
        absz: float  # absolute position of the center (of the cropped field) in the z axis as recorded by ScanM
        scan_type: enum("xy", "xz", "xyz")  # Type of scan
        npixartifact : int unsigned         # Number of pixel with light artifact
        nxpix: int unsigned                 # number of pixels in x
        nypix: int unsigned                 # number of pixels in y
        nzpix: int unsigned                 # number of pixels in z
        nxpix_offset: int unsigned          # number of offset pixels in x
        nxpix_retrace: int unsigned         # number of retrace pixels in x
        pixel_size_um :float                # width of a pixel in um (also height if y is second dimension)
        z_step_um = NULL :float             # z-step in um
        """
        return definition

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
            ch_name : varchar(32)  # name of the channel
            ---
            ch_average :longblob  # Stack median over time
            """
            return definition

    def make(self, key, verboselvl=0):
        self.add_experiment_fields(
            key, only_new=False, verboselvl=verboselvl, suppress_errors=False, restr_headers=None,
            allow_user_input=False)

    def rescan_filesystem(self, restrictions: dict = None, verboselvl: int = 0, suppress_errors: bool = False,
                          allow_user_input: bool = True, restr_headers: Optional[list] = None):
        """Scan filesystem for new fields and add them to the database."""
        if restrictions is None:
            restrictions = dict()

        for key in (self.key_source & restrictions):
            self.add_experiment_fields(
                key, restr_headers=restr_headers, only_new=True, verboselvl=verboselvl, suppress_errors=suppress_errors)

    def load_exp_file_info_df(self, exp_key):
        from_raw_data = (self.raw_params_table & exp_key).fetch1('from_raw_data')
        header_path = (self.experiment_table & exp_key).fetch1('header_path')
        data_folder = (self.userinfo_table & exp_key).fetch1("raw_data_dir" if from_raw_data else "pre_data_dir")
        user_dict = (self.userinfo_table & exp_key).fetch1()

        file_info_df = get_file_info_df(os.path.join(header_path, data_folder), user_dict, from_raw_data)
        if len(file_info_df) > 0:
            file_info_df = file_info_df[file_info_df['field'].notnull() & (file_info_df['kind'] == 'response')]
            file_info_df.sort_values('mask_order', inplace=True, ascending=True)

            # Set defaults
            if self.incl_cond1:
                file_info_df['cond1'].fillna('control', inplace=True)
            if self.incl_cond2:
                file_info_df['cond2'].fillna('control', inplace=True)
            if self.incl_cond3:
                file_info_df['cond3'].fillna('control', inplace=True)
            if self.incl_region:
                file_info_df['region'].fillna('N/A', inplace=True)

        return file_info_df

    def add_experiment_fields(
            self, exp_key, only_new: bool, verboselvl: int, suppress_errors: bool, allow_user_input: bool = False,
            restr_headers: Optional[list] = None):

        if verboselvl > 5:
            print("add_experiment_fields")

        if restr_headers is not None:
            header_path = (self.experiment_table & exp_key).fetch1('header_path')
            if header_path not in restr_headers:
                if verboselvl > 1:
                    print(f"\tSkipping header_path `{header_path}` because of restriction")
                return

        file_info_df = self.load_exp_file_info_df(exp_key)
        if len(file_info_df) == 0:
            if verboselvl > 0:
                print(f"\tSkipping because no files found for key={exp_key}")
            return

        field_dfs = file_info_df.groupby(self.new_primary_keys)

        if verboselvl > 0:
            print(f"Found {len(file_info_df)} files in {len(field_dfs)} for key={exp_key}")

        for i, (field_info, field_df) in enumerate(field_dfs):

            if verboselvl > 5:
                print(f"Checking field: {field_info} ({i + 1}/{len(field_dfs)})")

            field_key = {**dict(zip(self.new_primary_keys, field_info)), **exp_key}

            if verboselvl > 5:
                print(f"checking field: {field_key}")

            if only_new and (len((self & field_key).proj()) > 0):
                if verboselvl > 1:
                    print(f"\tSkipping field `{field_key}` because it already exists")
                continue

            if verboselvl > 0:
                print(f"\tAdding field: `{field_key}`")

            try:
                self.add_field(field_key=field_key, filepaths=field_df['filepath'].values,
                               allow_user_input=allow_user_input, verboselvl=verboselvl)
            except Exception as e:
                if suppress_errors:
                    print("Suppressed Error:", e, '\n\tfor key:\n', field_key, '\n\t', field_df['filepath'])
                else:
                    raise e

    def add_field(self, field_key, filepaths, verboselvl=0, allow_user_input=False):
        if verboselvl > 4:
            print('\t\tAdd field with files:', filepaths)

        setupid = (self.experiment_table().ExpInfo & field_key).fetch1("setupid")

        rec = self.load_first_file_wo_error(filepaths=filepaths, setupid=setupid, date=field_key['date'],
                                            allow_user_input=allow_user_input)
        field_entry, avg_entries = self.complete_keys(base_key=field_key, rec=rec)

        self.insert1(field_entry, allow_direct_insert=True)
        for avg_entry in avg_entries:
            self.StackAverages().insert1(avg_entry, allow_direct_insert=True)

    def plot1(self, key=None):
        key = get_primary_key(table=self, key=key)
        data_name, alt_name = (self.userinfo_table & key).fetch1('data_stack_name', 'alt_stack_name')
        main_ch_average = (self.StackAverages & key & f'ch_name="{data_name}"').fetch1('ch_average')
        try:
            alt_ch_average = (self.StackAverages & key & f'ch_name="{alt_name}"').fetch1('ch_average')
        except dj.DataJointError:
            alt_ch_average = np.full_like(main_ch_average, np.nan)

        npixartifact = (self & key).fetch1('npixartifact')

        plot_field(main_ch_average, alt_ch_average, title=key, npixartifact=npixartifact, figsize=(8, 4))

    @staticmethod
    def load_first_file_wo_error(filepaths, setupid, date, allow_user_input=False):
        """Load first file from filepaths and return recording and filepath. Skip all files causes errors."""
        for i, filepath in enumerate(filepaths):
            try:
                rec = ScanMRecording(filepath=filepath, setup_id=setupid, date=date)
                break
            except Exception as e:
                error_msg = f"Failed to load file with error {e}:\n{filepath}"
                if filepath == filepaths[-1]:
                    raise OSError(error_msg)
                elif allow_user_input:
                    if input(f"{error_msg}\nTry again for {filepaths[i + 1]}? (y/n)') != 'y'") == 'y':
                        continue
                else:
                    raise OSError(error_msg)
        else:
            raise OSError(f"Failed to load any of the files:\n{filepaths}")

        return rec

    @staticmethod
    def complete_keys(base_key, rec) -> (dict, list):
        field_entry = deepcopy(base_key)

        field_entry["field_data_file"] = rec.filepath
        field_entry["absx"] = rec.pos_x_um
        field_entry["absy"] = rec.pos_y_um
        field_entry["absz"] = rec.pos_z_um
        field_entry["scan_type"] = rec.scan_type
        field_entry["npixartifact"] = rec.pix_n_artifact
        field_entry["nxpix"] = rec.pix_nx
        field_entry["nypix"] = rec.pix_ny
        field_entry["nzpix"] = rec.pix_nz
        field_entry["nxpix_offset"] = rec.pix_n_line_offset
        field_entry["nxpix_retrace"] = rec.pix_n_retrace
        field_entry["pixel_size_um"] = rec.pix_dx_um
        field_entry["z_step_um"] = rec.pix_dz_um

        # get stack avgs
        avg_entries = []
        for name, stack in rec.ch_stacks.items():
            avg_entry = deepcopy(base_key)
            avg_entry["ch_name"] = name
            avg_entry["ch_average"] = np.median(stack, 2).astype(np.float32)
            avg_entries.append(avg_entry)

        return field_entry, avg_entries
