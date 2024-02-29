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
from djimaging.utils.scanm_utils import get_stimulator_delay, load_roi_mask_from_h5


class PresentationTemplate(dj.Computed):
    database = ""
    __filepath = 'h5_header'

    @property
    def definition(self):
        definition = f"""
        # information about each stimulus presentation
        -> self.field_table
        -> self.stimulus_table
        -> self.params_table
        condition             :varchar(191)     # condition (pharmacological or other)
        ---
        {self.filepath}     :varchar(191)     # path to h5 file
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
    def params_table(self):
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
            return self.params_table.proj() * self.field_table.RoiMask.proj() * self.stimulus_table.proj()
        except (AttributeError, TypeError):
            pass

    class StackAverages(dj.Part):
        @property
        def definition(self):
            definition = """
            # Stack median (over time of the available channels)
            -> master
            ch_name : varchar(191)  # name of the channel
            ---
            ch_average : longblob  # Stack median over time
            """
            return definition

    class RoiMask(dj.Part):
        @property
        def definition(self):
            definition = """
            # ROI Mask of presentation
            -> master
            ---
            pres_and_field_mask  : enum("same", "different", "shifted", "similar", "no_field_mask")  # relationship to field mask
            roi_mask    :longblob       # roi mask for the presentation, can be same as field
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
            hdrleninvaluepairs         :int        
            hdrleninbytes              :float        
            minvolts_ao                :float        
            maxvolts_ao                :float        
            stimchanmask               :float        
            maxstimbufmaplen           :float        
            numberofstimbufs           :int        
            targetedpixdur_us          :float        
            minvolts_ai                :float        
            maxvolts_ai                :float        
            inputchanmask              :int        
            numberofinputchans         :int        
            pixsizeinbytes             :float        
            numberofpixbufsset         :int        
            pixeloffs                  :float        
            pixbufcounter              :float        
            user_scanmode              :float        
            user_dxpix                 :int        
            user_dypix                 :int        
            user_npixretrace           :int        
            user_nxpixlineoffs         :int        
            user_nypixlineoffs=0       :int        
            user_divframebufreq        :float        
            user_scantype              :float        
            user_scanpathfunc          :varchar(191) 
            user_nsubpixoversamp       :int        
            user_nfrperstep            :int        
            user_xoffset_v             :float        
            user_yoffset_v             :float        
            user_offsetz_v=0           :float        
            user_zoomz=0               :float        
            user_noyscan=0             :float        
            realpixdur                 :float        
            oversampfactor             :float        
            xcoord_um                  :float        
            ycoord_um                  :float        
            zcoord_um                  :float        
            zstep_um=0                 :float        
            zoom                       :float        
            angle_deg                  :float        
            datestamp_d_m_y            :varchar(191) 
            timestamp_h_m_s_ms         :varchar(191) 
            inchan_pixbuflenlist       :varchar(191) 
            username                   :varchar(191) 
            guid                       :varchar(191) 
            origpixdatafilename        :varchar(191) 
            stimbuflenlist             :varchar(191) 
            callingprocessver          :varchar(191) 
            callingprocesspath         :varchar(191) 
            targetedstimdurlist        :varchar(191) 
            computername               :varchar(191) 
            scanm_pver_targetos        :varchar(191) 
            user_zlensscaler=0         :float        
            user_stimbufperfr=0        :float        
            user_aspectratiofr=0       :float        
            user_zforfastscan=0        :float        
            user_zlensshifty=0         :float        
            user_nzpixlineoff=0        :float        
            user_dzpix=0               :float        
            user_setupid=0             :float        
            user_nzpixretrace=0        :float        
            user_laserwavelen_nm=0     :float        
            user_scanpathfunc          :varchar(191) 
            user_dzfrdecoded=0         :int        
            user_dxfrdecoded=0         :int        
            user_dyfrdecoded=0         :int        
            user_zeroz_v=0             :float        
            igorguiver                 :varchar(191) 
            user_comment=''            :varchar(191) 
            user_objective=''          :varchar(191) 
            realstimdurlist=""         :varchar(191) 
            user_ichfastscan=0         :float        
            user_trajdefvrange_v=0     :float        
            user_ntrajparams=0         :float        
            user_offset_v=0            :float        
            user_etl_polarity_v=0      :float        
            user_etl_min_v=0           :float        
            user_etl_max_v=0           :float        
            user_etl_neutral_v=0       :float        
            user_nimgperfr=0           :int      
            """
            return definition

    def rescan_filesystem(self, restrictions: dict = None, verboselvl: int = 0, suppress_errors: bool = False):
        """Scan filesystem for new fields and add them to the database."""
        if restrictions is None:
            restrictions = dict()

        for key in (self.key_source & restrictions):
            self.add_field_presentations(key, only_new=True, verboselvl=verboselvl, suppress_errors=suppress_errors)

    def make(self, key, verboselvl=0):
        self.add_field_presentations(key, only_new=False, verboselvl=0, suppress_errors=False)

    def add_field_presentations(self, key, only_new: bool, verboselvl: int, suppress_errors: bool):

        if verboselvl > 2:
            print('key:', key)

        field = (self.field_table() & key).fetch1("field")
        stim_loc, field_loc, condition_loc = (self.userinfo_table() & key).fetch1(
            "stimulus_loc", "field_loc", "condition_loc")
        stim_alias = (self.stimulus_table() & key).fetch1("alias").split('_')

        pre_data_path = os.path.join(
            (self.experiment_table() & key).fetch1('header_path'),
            (self.userinfo_table() & key).fetch1("pre_data_dir"))
        assert os.path.exists(pre_data_path), f"Error: Data folder does not exist: {pre_data_path}"

        field_files = get_field_files(folder=pre_data_path, field=field, field_loc=field_loc, incl_hidden=False)

        pres_files = []
        for data_file in field_files:
            stim = get_stim(data_file, loc=stim_loc, fill_value='nostim')
            condition = get_condition(data_file, loc=condition_loc)
            if (stim == 'nostim') or (len(stim) == 0) or (stim.lower() not in stim_alias):
                continue
            if ("condition" in key) and (key['condition'] != condition):
                continue
            pres_files.append(data_file)

        for data_file in pres_files:
            new_key = deepcopy(key)
            new_key["condition"] = get_condition(data_file, loc=condition_loc)

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
                self.add_presentation(key=new_key, filepath=os.path.join(pre_data_path, data_file))
            except Exception as e:
                if suppress_errors:
                    print("Suppressed Error:", e, '\n\tfor key:', key)
                else:
                    print(f"Error for key: {key}")
                    raise e

    def add_presentation(self, key, filepath: str, compute_from_stack: bool = True):
        ch_stacks, wparams = scanm_utils.load_stacks_from_h5(filepath, ('wDataCh0', 'wDataCh1', 'wDataCh2'))

        if compute_from_stack:
            setupid = (self.experiment_table.ExpInfo() & key).fetch1("setupid")
            stimulator_delay = get_stimulator_delay(date=key['date'], setupid=setupid)
            trigger_precision = (self.params_table() & key).fetch1("trigger_precision")
            triggertimes, triggervalues = scanm_utils.compute_triggers_from_wparams(
                ch_stacks['wDataCh2'], wparams=wparams, precision=trigger_precision, stimulator_delay=stimulator_delay)
        else:
            triggertimes, triggervalues = scanm_utils.load_triggers_from_h5(filepath)

        isrepeated, ntrigger_rep = (self.stimulus_table() & key).fetch1("isrepeated", "ntrigger_rep")

        if isrepeated == 0:
            trigger_flag = triggertimes.size == ntrigger_rep
        else:
            trigger_flag = triggertimes.size % ntrigger_rep == 0

        if trigger_flag == 0:
            warnings.warn(f'Found {triggertimes.size} triggers, expected {ntrigger_rep} (per rep): key={key}.')

        roi_mask = load_roi_mask_from_h5(filepath=filepath, ignore_not_found=True)
        field_roi_masks = (self.field_table.RoiMask & key).fetch('roi_mask')

        if len(field_roi_masks) == 0:
            pres_and_field_mask = 'no_field_mask'
        elif len(field_roi_masks) == 1:
            pres_and_field_mask = scanm_utils.compare_roi_masks(roi_mask, field_roi_masks[0])
        else:
            raise ValueError(f'Found multiple ROI masks for key={key} in Field. Should be 0 or 1.')

        setupid = (self.experiment_table().ExpInfo() & key).fetch1("setupid")
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
        scaninfo_key.update(wparams)
        try:
            scaninfo_key["line_duration"] = (wparams['user_dxpix'] * wparams['realpixdur']) * 1e-6
            scaninfo_key["scan_period"] = (scaninfo_key["line_duration"] * wparams['user_dypix'])
            scaninfo_key["scan_frequency"] = 1. / scaninfo_key["scan_period"]
        except KeyError:
            pass
        remove_list = ["user_warpparamslist", "user_nwarpparams"]
        for k in remove_list:
            scaninfo_key.pop(k, None)

        # Roi mask key
        if roi_mask is not None:
            roimask_key = deepcopy(key)
            roimask_key["roi_mask"] = roi_mask
            roimask_key["pres_and_field_mask"] = pres_and_field_mask
        else:
            roimask_key = None

        self.insert1(pres_key, allow_direct_insert=True)
        (self.ScanInfo & key).insert1(scaninfo_key, allow_direct_insert=True)
        if roimask_key is not None:
            (self.RoiMask & key).insert1(roimask_key, allow_direct_insert=True)
        for avg_key in avg_keys:
            (self.StackAverages & key).insert1(avg_key, allow_direct_insert=True)

    def plot1(self, key=None, plot_field_rois=False):
        key = get_primary_key(table=self, key=key)

        field_roi_mask = (self.field_table.RoiMask & key).fetch1("roi_mask")
        pres_roi_mask, pres_and_field_mask = (self.RoiMask & key).fetch1("roi_mask", "pres_and_field_mask")

        if pres_and_field_mask != 'same':
            warnings.warn(f'field_roi_mask and pres_roi_mask are {pres_and_field_mask}, ' +
                          f'plot_field_rois={plot_field_rois}')

        roi_mask = field_roi_mask if plot_field_rois else pres_roi_mask

        npixartifact = (self.field_table & key).fetch1('npixartifact')
        data_name, alt_name = (self.userinfo_table & key).fetch1('data_stack_name', 'alt_stack_name')
        main_ch_average = (self.StackAverages & key & f'ch_name="{data_name}"').fetch1('ch_average')
        alt_ch_average = (self.StackAverages & key & f'ch_name="{alt_name}"').fetch1('ch_average')
        plot_field(main_ch_average, alt_ch_average, roi_mask, title=key, npixartifact=npixartifact)
