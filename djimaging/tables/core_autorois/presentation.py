import os
import warnings
from abc import abstractmethod
from copy import deepcopy

import datajoint as dj
import numpy as np

from djimaging.utils.datafile_utils import get_condition, get_stim
from djimaging.utils import scanm_utils
from djimaging.utils.alias_utils import get_field_files
from djimaging.utils.dj_utils import get_primary_key
from djimaging.utils.plot_utils import plot_field
from djimaging.utils.scanm_utils import get_stimulator_delay


class PresentationTemplate(dj.Computed):
    database = ""
    __filepath = 'pres_data_file'

    @property
    def definition(self):
        definition = f"""
        # information about each stimulus presentation
        -> self.field_table
        -> self.stimulus_table
        -> self.raw_params_table
        condition             :varchar(16)     # condition (pharmacological or other)
        ---
        {self.filepath}       :varchar(191)     # path to h5 file
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
    def filepath(self):
        return self.__filepath

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
            ch_name : varchar(16)  # name of the channel
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
            scan_period=0              :float # Scanning frequency in Hz
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

    def add_field_presentations(self, key, only_new: bool, verboselvl: int, suppress_errors: bool):
        if verboselvl > 2:
            print('key:', key)

        field = (self.field_table & key).fetch1("field")
        stim_loc, field_loc, condition_loc = (self.userinfo_table & key).fetch1(
            "stimulus_loc", "field_loc", "condition_loc")
        stim_alias = (self.stimulus_table & key).fetch1("alias").split('_')
        from_raw_data = (self.raw_params_table & key).fetch1("from_raw_data")

        data_folder = os.path.join(
            (self.experiment_table() & key).fetch1('header_path'),
            (self.userinfo_table() & key).fetch1("raw_data_dir" if from_raw_data else "pre_data_dir"))
        assert os.path.exists(data_folder), f"Error: Data folder does not exist: {data_folder}"

        field_files = get_field_files(folder=data_folder, field=field, incl_hidden=False,
                                      field_loc=field_loc - 1 if from_raw_data else field_loc,
                                      ftype='smp' if from_raw_data else 'h5')

        pres_files = []
        for data_file in field_files:
            stim = get_stim(data_file, loc=stim_loc - 1 if from_raw_data else stim_loc)
            condition = get_condition(data_file, loc=condition_loc - 1 if from_raw_data else condition_loc)

            if (stim == 'nostim') or (len(stim) == 0) or (stim.lower() not in stim_alias):
                if verboselvl > 3:
                    print(f"\tSkipping `{data_file}`, which has stim=`{stim}` not in stim_alias={stim_alias}")
                continue
            if ("condition" in key) and (key['condition'] != condition):
                if verboselvl > 3:
                    print(f"\tSkipping `{data_file}`, which has condition=`{condition}` not matching the key")
                continue
            pres_files.append(data_file)

        if len(pres_files) == 0 and verboselvl > 2:
            print(f"\tDid not find a presentation for given key, field_files are\n\t\t{field_files}")

        for data_file in pres_files:
            new_key = deepcopy(key)
            new_key["condition"] = get_condition(data_file, loc=condition_loc - 1 if from_raw_data else condition_loc)

            for k in self.primary_key:
                assert k in new_key, f'Did not find k={k} in new_key.'

            exists = len((self & new_key).fetch()) > 0
            if only_new and exists:
                if verboselvl > 1:
                    print(f"\tSkipping presentation for `{data_file}`, which is already present")
                continue

            try:
                if verboselvl > 0:
                    print(f"\tAdding presentation for `{data_file}`.")
                self.add_presentation(key=new_key, filepath=os.path.join(data_folder, data_file))
            except Exception as e:
                if suppress_errors:
                    print("Suppressed Error:", e, '\n\tfor key:', key)
                else:
                    print(f"Error for key: {key}")
                    raise e

    def add_presentation(self, key, filepath: str, compute_from_stack: bool = True):
        setupid = (self.experiment_table.ExpInfo() & key).fetch1("setupid")
        from_raw_data, trigger_precision = (self.raw_params_table & key).fetch1('from_raw_data', 'trigger_precision')
        isrepeated, ntrigger_rep = (self.stimulus_table & key).fetch1("isrepeated", "ntrigger_rep")

        ch_stacks, wparams = scanm_utils.load_stacks(
            filepath, from_raw_data=from_raw_data,
            ch_names=('wDataCh0', 'wDataCh1', 'wDataCh2') if ntrigger_rep > 0 else ('wDataCh0', 'wDataCh1'))

        if ntrigger_rep > 0:
            if compute_from_stack:
                stimulator_delay = get_stimulator_delay(date=key['date'], setupid=setupid)
                triggertimes, triggervalues = scanm_utils.compute_triggers_from_wparams(
                    ch_stacks['wDataCh2'], wparams=wparams, precision=trigger_precision,
                    stimulator_delay=stimulator_delay)
            else:
                triggertimes, triggervalues = scanm_utils.load_triggers_from_h5(filepath)
        else:
            triggertimes = np.array([])
            triggervalues = np.array([])

        if isrepeated == 0:
            trigger_flag = triggertimes.size == ntrigger_rep
        else:
            trigger_flag = triggertimes.size % ntrigger_rep == 0

        if trigger_flag == 0:
            warnings.warn(f'Found {triggertimes.size} triggers, expected {ntrigger_rep} (per rep): key={key}.')

        nxpix = wparams["user_dxpix"] - wparams["user_npixretrace"] - wparams["user_nxpixlineoffs"]
        pixel_size_um = scanm_utils.get_pixel_size_xy_um(zoom=wparams["zoom"], setupid=setupid, npix=nxpix)
        z_step_um = wparams.get('zstep_um', 0.)
        npixartifact = scanm_utils.get_npixartifact(setupid=setupid)
        scan_type = scanm_utils.get_scan_type_from_wparams(wparams)

        field_pixel_size_um = (self.field_table & key).fetch1("pixel_size_um")
        if not np.isclose(pixel_size_um, field_pixel_size_um):
            warnings.warn(
                f"""pixel_size_um {pixel_size_um:.3g} is different in field {field_pixel_size_um}.
                Did you use a different zoom?
                This can result in unexpected problems.""")

        pres_key = deepcopy(key)
        pres_key[self.filepath] = filepath
        pres_key["trigger_flag"] = int(trigger_flag)
        pres_key["triggertimes"] = triggertimes
        pres_key["triggervalues"] = triggervalues
        pres_key["scan_type"] = scan_type
        pres_key["npixartifact"] = npixartifact
        pres_key["pixel_size_um"] = pixel_size_um
        pres_key["z_step_um"] = z_step_um

        # get stack avgs
        avg_keys = []
        for name, stack in ch_stacks.items():
            avg_key = deepcopy(key)
            avg_key["ch_name"] = name
            avg_key["ch_average"] = np.median(stack, 2)
            avg_keys.append(avg_key)

        # extract params for scaninfo
        scaninfo_key = deepcopy(key)
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

        self.insert1(pres_key, allow_direct_insert=True)
        (self.ScanInfo & key).insert1(scaninfo_key, allow_direct_insert=True)
        for avg_key in avg_keys:
            (self.StackAverages & key).insert1(avg_key, allow_direct_insert=True)

    def plot1(self, key=None):
        key = get_primary_key(table=self, key=key)
        npixartifact = (self.field_table & key).fetch1('npixartifact')
        data_name, alt_name = (self.userinfo_table & key).fetch1('data_stack_name', 'alt_stack_name')
        main_ch_average = (self.StackAverages & key & f'ch_name="{data_name}"').fetch1('ch_average')
        alt_ch_average = (self.StackAverages & key & f'ch_name="{alt_name}"').fetch1('ch_average')
        plot_field(main_ch_average, alt_ch_average, title=key, npixartifact=npixartifact, figsize=(8, 4))
