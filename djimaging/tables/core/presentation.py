import warnings
from abc import abstractmethod
from copy import deepcopy

import datajoint as dj
import numpy as np

from djimaging.utils.dj_utils import get_primary_key
from djimaging.utils.plot_utils import plot_field
from djimaging.utils.scanm.recording import ScanMRecording


class PresentationTemplate(dj.Computed):
    database = ""
    incl_region = True  # Include region as primary key?
    incl_cond1 = True  # Include condition 1 as primary key?
    incl_cond2 = False  # Include condition 2 as primary key?
    incl_cond3 = False  # Include condition 3 as primary key?

    @property
    def definition(self):
        definition = f"""
        # information about each stimulus presentation
        -> self.field_table
        -> self.stimulus_table
        -> self.raw_params_table
        """

        if self.incl_region and not self.field_table.incl_region:
            definition += "    region   :varchar(255)    # region (e.g. LR or RR)\n"
        if self.incl_cond1 and not self.field_table.incl_cond1:
            definition += "    cond1    :varchar(255)    # condition (pharmacological or other)\n"
        if self.incl_cond2 and not self.field_table.incl_cond2:
            definition += "    cond2    :varchar(255)    # condition (pharmacological or other)\n"
        if self.incl_cond3 and not self.field_table.incl_cond3:
            definition += "    cond3    :varchar(255)    # condition (pharmacological or other)\n"

        definition += """
        ---
        pres_data_file :varchar(255)        # path to file (e.g. h5 file)
        triggertimes :longblob              # triggertimes in each presentation
        triggervalues :longblob             # values of the recorded triggers
        trigger_valid :tinyint unsigned     # Are triggers as expected (1) or not (0)?
        absx: float  # absolute position of the center (of the cropped field) in the x axis as recorded by ScanM
        absy: float  # absolute position of the center (of the cropped field) in the y axis as recorded by ScanM
        absz: float  # absolute position of the center (of the cropped field) in the z axis as recorded by ScanM
        scan_type: enum("xy", "xz", "xyz")  # Type of scan
        npixartifact : int unsigned         # number of pixel with light artifact
        nxpix: int unsigned                 # number of pixels in x
        nypix: int unsigned                 # number of pixels in y
        nzpix: int unsigned                 # number of pixels in z
        nxpix_offset: int unsigned          # number of offset pixels in x
        nxpix_retrace: int unsigned         # number of retrace pixels in x
        pixel_size_um :float                # width of a pixel in um (also height if y is second dimension)
        z_step_um = NULL :float             # z-step in um
        nframes: int unsigned               # number of pixels in time
        """
        return definition

    @property
    @abstractmethod
    def raw_params_table(self):
        pass

    @property
    @abstractmethod
    def field_table(self):
        pass

    @property
    @abstractmethod
    def stimulus_table(self):
        pass

    @property
    @abstractmethod
    def experiment_table(self):
        pass

    @property
    @abstractmethod
    def userinfo_table(self):
        pass

    @property
    def key_source(self):
        try:
            return self.field_table.proj() * self.stimulus_table.proj() * self.raw_params_table.proj()
        except (AttributeError, TypeError):
            pass

    class StackAverages(dj.Part):
        @property
        def definition(self):
            definition = """
            # Stack median (over time of the available channels)
            -> master
            ch_name : varchar(255)  # name of the channel
            ---
            ch_average : longblob  # Stack median over time
            """
            return definition

    class ScanInfo(dj.Part):
        @property
        def definition(self):
            definition = """
            # Data read from wParamsNum and wParamsStr tables in h5 file
            -> master
            ---
            scan_frequency=-1 :float             # Scanning frequency in Hz
            scan_period=-1 :float                # Scanning duration per frame in seconds
            line_duration=-1 :float              # Line duration from OS_Parameters
            real_pixel_duration=-1 :float        # Duration of a pixel (un-cropped)
            zoom :float                          # Zoom of objective, changes pixel size
            angle_deg :float                     # Angle of objective, changes rotation of image
            scan_params_dict :longblob           # Other ScanM parameters less frequently used
            """
            return definition

    def make(self, key, verboselvl: int = 0):
        self.add_field_presentations(key, only_new=False, verboselvl=verboselvl, suppress_errors=False)

    def rescan_filesystem(self, restrictions: dict = None, verboselvl: int = 0, suppress_errors: bool = False):
        """Scan filesystem for new fields and add them to the database."""
        if restrictions is None:
            restrictions = dict()

        for key in (self.key_source & restrictions):
            self.add_field_presentations(key, only_new=True, verboselvl=verboselvl, suppress_errors=suppress_errors)

    @property
    def new_primary_keys(self):
        """Primary keys that will be added in this table with respect to the field table."""
        new_primary_keys = ['stim_name']
        if self.incl_region and not self.field_table.incl_region:
            new_primary_keys.append('region')
        if self.incl_cond1 and not self.field_table.incl_cond1:
            new_primary_keys.append('cond1')
        if self.incl_cond2 and not self.field_table.incl_cond2:
            new_primary_keys.append('cond2')
        if self.incl_cond3 and not self.field_table.incl_cond3:
            new_primary_keys.append('cond3')
        return new_primary_keys

    def load_field_stim_file_info_df(self, field_stim_key):
        file_info_df = self.field_table().load_exp_file_info_df(field_stim_key)
        file_info_df = file_info_df[file_info_df['kind'] == 'response']

        for new_key in self.field_table().new_primary_keys:
            file_info_df = file_info_df[file_info_df[new_key] == field_stim_key[new_key]]

        # Filter wrong stimuli
        trg_stim_aliases = (self.stimulus_table & field_stim_key).fetch1('alias').split('_')
        file_info_df = file_info_df[file_info_df['stimulus'].apply(lambda x: x.lower() in trg_stim_aliases)]
        file_info_df['stim_name'] = field_stim_key['stim_name']

        # Set defaults
        if self.incl_cond1 and not self.field_table.incl_cond1:
            file_info_df['cond1'].fillna('control', inplace=True)
        if self.incl_cond2 and not self.field_table.incl_cond2:
            file_info_df['cond2'].fillna('control', inplace=True)
        if self.incl_cond3 and not self.field_table.incl_cond3:
            file_info_df['cond3'].fillna('control', inplace=True)

        return file_info_df

    def add_field_presentations(self, field_stim_key, only_new: bool, verboselvl: int, suppress_errors: bool):
        if verboselvl > 2:
            print('\nProcessing key:', field_stim_key)

        file_info_df = self.load_field_stim_file_info_df(field_stim_key)
        pres_dfs = file_info_df.groupby(self.new_primary_keys)

        if (verboselvl > 0 and len(file_info_df) > 0) or verboselvl > 3:
            print(f"Found {len(file_info_df)} files for key={field_stim_key}")

        if verboselvl > 3:
            print(file_info_df['filepath'].values)

        for pres_info, pres_df in pres_dfs:
            pres_key = {**dict(zip(self.new_primary_keys, pres_info)), **field_stim_key}

            if len(pres_df) > 1:
                raise ValueError(f"Found multiple files for key={pres_key}:\n{pres_df['filepath'].values}\n"
                                 "Probably you have multiple conditions / regions for the same field. "
                                 "Please check the primary keys of the field and presentation table. "
                                 "Consider setting `incl_region=True`, `incl_cond1=True` etc.")

            if only_new and (len((self & pres_key).proj()) > 0):
                if verboselvl > 1:
                    print(f"\tSkipping presentation `{pres_key}` because it already exists")
                continue

            if verboselvl > 0:
                print(f"\tAdding presentation: `{pres_key}`")

            try:
                self.add_presentation(
                    pres_key=pres_key, filepath=pres_df.iloc[0]['filepath'], verboselvl=verboselvl)
            except Exception as e:
                if suppress_errors:
                    print("Suppressed Error:", e, '\n\tfor key:\n', pres_key, '\n\t', pres_df.iloc[0]['filepath'])
                else:
                    raise e

    def add_presentation(self, pres_key, filepath: str, verboselvl=0):

        if verboselvl > 4:
            print('\t\tAdd presentation with file:', filepath)

        setupid = (self.experiment_table().ExpInfo & pres_key).fetch1("setupid")
        from_raw_data, trigger_precision, compute_from_stack = (self.raw_params_table & pres_key).fetch1(
            'from_raw_data', 'trigger_precision', 'compute_from_stack')
        isrepeated, ntrigger_rep = (self.stimulus_table & pres_key).fetch1("isrepeated", "ntrigger_rep")

        rec = ScanMRecording(filepath=filepath, setup_id=setupid, date=pres_key['date'],
                             repeated_stim=isrepeated, ntrigger_rep=ntrigger_rep, time_precision=trigger_precision)
        if compute_from_stack or from_raw_data:
            rec.compute_triggers()
        pres_entry, scaninfo_entry, avg_entries = self.complete_keys(pres_key, rec)

        # Sanity checks
        field_pixel_size_um = (self.field_table & pres_key).fetch1("pixel_size_um")
        if not np.isclose(pres_entry['pixel_size_um'], field_pixel_size_um):
            warnings.warn(
                f"Warning for key={pres_key}.\n"
                f"pixel_size_um {pres_entry['pixel_size_um']:.3g} is different in field {field_pixel_size_um}. "
                f"Did you use a different zoom? This can result in unexpected problems.")

        if pres_entry["trigger_valid"] == 0:
            warnings.warn(
                f"Warning for key={pres_key}.\n"
                f'Found {pres_entry["triggertimes"].size} triggers, expected {ntrigger_rep} '
                f'{"(per rep)" if isrepeated else ""}: {filepath}.')

        self.insert1(pres_entry, allow_direct_insert=True)
        self.ScanInfo().insert1(scaninfo_entry, allow_direct_insert=True)
        for avg_entry in avg_entries:
            self.StackAverages().insert1(avg_entry, allow_direct_insert=True)

    def plot1(self, key=None):
        key = get_primary_key(table=self, key=key)
        npixartifact = (self.field_table & key).fetch1('npixartifact')
        data_name, alt_name = (self.userinfo_table & key).fetch1('data_stack_name', 'alt_stack_name')
        main_ch_average = (self.StackAverages & key & f'ch_name="{data_name}"').fetch1('ch_average')
        alt_ch_average = (self.StackAverages & key & f'ch_name="{alt_name}"').fetch1('ch_average')
        plot_field(main_ch_average, alt_ch_average, title=key, npixartifact=npixartifact, figsize=(8, 4))

    @staticmethod
    def complete_keys(base_key, rec) -> (dict, dict, list):

        pres_entry = deepcopy(base_key)
        pres_entry["pres_data_file"] = rec.filepath

        pres_entry["trigger_valid"] = int(rec.trigger_valid)
        pres_entry["triggertimes"] = rec.trigger_times
        pres_entry["triggervalues"] = rec.trigger_values

        pres_entry["absx"] = rec.pos_x_um
        pres_entry["absy"] = rec.pos_y_um
        pres_entry["absz"] = rec.pos_z_um
        pres_entry["scan_type"] = rec.scan_type
        pres_entry["npixartifact"] = rec.pix_n_artifact
        pres_entry["nxpix"] = rec.pix_nx
        pres_entry["nypix"] = rec.pix_ny
        pres_entry["nzpix"] = rec.pix_nz
        pres_entry["nxpix_offset"] = rec.pix_n_line_offset
        pres_entry["nxpix_retrace"] = rec.pix_n_retrace
        pres_entry["pixel_size_um"] = rec.pix_dx_um
        pres_entry["z_step_um"] = rec.pix_dz_um
        pres_entry["nframes"] = rec.ch_n_frames

        # get stack avgs
        avg_entries = []
        for name, stack in rec.ch_stacks.items():
            avg_entry = deepcopy(base_key)
            avg_entry["ch_name"] = name
            avg_entry["ch_average"] = np.median(stack, 2)
            avg_entries.append(avg_entry)

        # extract params for scaninfo
        scaninfo_entry = deepcopy(base_key)

        scaninfo_entry["scan_frequency"] = rec.scan_frequency
        scaninfo_entry["scan_period"] = rec.scan_period
        scaninfo_entry["line_duration"] = rec.scan_line_duration
        scaninfo_entry["real_pixel_duration"] = rec.real_pixel_duration
        scaninfo_entry["zoom"] = rec.obj_zoom
        scaninfo_entry["angle_deg"] = rec.obj_angle_deg
        scaninfo_entry["scan_params_dict"] = rec.wparams_other

        return pres_entry, scaninfo_entry, avg_entries
