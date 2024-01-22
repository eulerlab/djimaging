import warnings
from abc import abstractmethod
from copy import deepcopy

import datajoint as dj
import numpy as np

from djimaging.utils.scanm import read_h5_utils, read_utils, setup_utils, wparams_utils
from djimaging.utils.dj_utils import get_primary_key
from djimaging.utils.plot_utils import plot_field


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
        pres_data_file        :varchar(255)     # path to file (e.g. h5 file)
        trigger_flag          :tinyint unsigned # Are triggers as expected (1) or not (0)?
        triggertimes          :longblob         # triggertimes in each presentation
        triggervalues         :longblob         # values of the recorded triggers
        scan_type: enum("xy", "xz", "xyz")  # Type of scan
        npixartifact : int unsigned # Number of pixel with light artifact
        pixel_size_um :float  # width / height of a pixel in um
        z_step_um :float  # z-step in um
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
            scan_period=0              :float # Scanning duration per frame in seconds
            scan_frequency=0           :float # Scanning frequency in Hz
            line_duration=0            :float # Line duration from OS_Parameters
            user_dxpix                 :int        
            user_dypix                 :int
            user_dzpix=0               :float        
            user_npixretrace           :int        
            user_nxpixlineoffs         :int
            xcoord_um                  :float        
            ycoord_um                  :float        
            zcoord_um                  :float
            zstep_um=0                 :float        
            zoom                       :float        
            angle_deg                  :float
            realpixdur                 :float 
            scan_params_dict           :longblob     
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
        field_pixel_size_um = (self.field_table & pres_key).fetch1("pixel_size_um")

        pres_entry, scaninfo_entry, avg_entries = load_pres_key_data(
            pres_key=pres_key, filepath=filepath,
            from_raw_data=from_raw_data, compute_from_stack=compute_from_stack, setupid=setupid,
            ntrigger_rep=ntrigger_rep, isrepeated=isrepeated,
            field_pixel_size_um=field_pixel_size_um, trigger_precision=trigger_precision)

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


def load_pres_key_data(pres_key, filepath, from_raw_data, compute_from_stack, setupid, ntrigger_rep, isrepeated,
                       field_pixel_size_um=None, trigger_precision='line'):
    ch_stacks, wparams = read_utils.load_stacks(
        filepath, from_raw_data=from_raw_data,
        ch_names=('wDataCh0', 'wDataCh1', 'wDataCh2') if ntrigger_rep > 0 else ('wDataCh0', 'wDataCh1'))

    if ntrigger_rep > 0:
        if compute_from_stack:
            stimulator_delay = setup_utils.get_stimulator_delay(date=pres_key['date'], setupid=setupid)
            triggertimes, triggervalues = wparams_utils.compute_triggers_from_wparams(
                ch_stacks['wDataCh2'], wparams=wparams, precision=trigger_precision,
                stimulator_delay=stimulator_delay)
        else:
            triggertimes, triggervalues = read_h5_utils.load_triggers(filepath)
    else:
        triggertimes = np.array([])
        triggervalues = np.array([])

    if isrepeated == 0:
        trigger_flag = triggertimes.size == ntrigger_rep
    else:
        trigger_flag = triggertimes.size % ntrigger_rep == 0

    if trigger_flag == 0:
        warnings.warn(f'Found {triggertimes.size} triggers, expected {ntrigger_rep} (per rep): {filepath}.')

    nxpix = wparams["user_dxpix"] - wparams["user_npixretrace"] - wparams["user_nxpixlineoffs"]
    pixel_size_um = setup_utils.get_pixel_size_xy_um(zoom=wparams["zoom"], setupid=setupid, npix=nxpix)
    z_step_um = wparams.get('zstep_um', 0.)
    npixartifact = setup_utils.get_npixartifact(setupid=setupid)
    scan_type = wparams_utils.get_scan_type(wparams)

    if not np.isclose(pixel_size_um, field_pixel_size_um):
        warnings.warn(
            f"pixel_size_um {pixel_size_um:.3g} is different in field {field_pixel_size_um}. "
            f"Did you use a different zoom? This can result in unexpected problems.")

    pres_entry = deepcopy(pres_key)
    pres_entry["pres_data_file"] = filepath
    pres_entry["trigger_flag"] = int(trigger_flag)
    pres_entry["triggertimes"] = triggertimes
    pres_entry["triggervalues"] = triggervalues
    pres_entry["scan_type"] = scan_type
    pres_entry["npixartifact"] = npixartifact
    pres_entry["pixel_size_um"] = pixel_size_um
    pres_entry["z_step_um"] = z_step_um

    # get stack avgs
    avg_keys = []
    for name, stack in ch_stacks.items():
        avg_entry = deepcopy(pres_key)
        avg_entry["ch_name"] = name
        avg_entry["ch_average"] = np.median(stack, 2)
        avg_keys.append(avg_entry)

    # extract params for scaninfo
    scaninfo_key = deepcopy(pres_key)
    wparams = deepcopy(wparams)

    try:
        scaninfo_key["line_duration"] = (wparams['user_dxpix'] * wparams['realpixdur']) * 1e-6
        n_lines = wparams['user_dzpix'] if scan_type == 'xz' else wparams['user_dypix']
        scaninfo_key["scan_period"] = scaninfo_key["line_duration"] * n_lines
        scaninfo_key["scan_frequency"] = 1. / scaninfo_key["scan_period"]

        scaninfo_key["user_dxpix"] = wparams.pop("user_dxpix")
        scaninfo_key["user_dypix"] = wparams.pop("user_dypix")
        scaninfo_key["user_dzpix"] = wparams.pop("user_dzpix")
        scaninfo_key["user_npixretrace"] = wparams.pop("user_npixretrace")
        scaninfo_key["user_nxpixlineoffs"] = wparams.pop("user_nxpixlineoffs")
        scaninfo_key["xcoord_um"] = wparams.pop("xcoord_um")
        scaninfo_key["ycoord_um"] = wparams.pop("ycoord_um")
        scaninfo_key["zcoord_um"] = wparams.pop("zcoord_um")
        scaninfo_key["zstep_um"] = wparams.pop("zstep_um")
        scaninfo_key["zoom"] = wparams.pop("zoom")
        scaninfo_key["angle_deg"] = wparams.pop("angle_deg")
        scaninfo_key["realpixdur"] = wparams.pop("realpixdur")
    except KeyError as e:
        print(f'Could not find ScanInfo key in wparams with keys\n\t{wparams.keys()}')
        raise e
    scaninfo_key["scan_params_dict"] = wparams

    return pres_entry, scaninfo_key, avg_keys
