import os
import random
from copy import deepcopy

import datajoint as dj
import h5py
import numpy as np

from djimaging.utils.alias_utils import get_field_files
from djimaging.utils.data_utils import extract_h5_table
from djimaging.utils.dj_utils import PlaceholderTable
from djimaging.utils.plot_utils import plot_field
from djimaging.utils.scanm_utils import extract_ch0_ch1_stacks_from_h5


class PresentationTemplate(dj.Computed):
    database = ""  # hack to suppress DJ error

    @property
    def definition(self):
        definition = """
        # information about each stimulus presentation
        -> self.field_table
        -> self.stimulus_table
        condition             :varchar(255)     # condition (pharmacological or other)
        ---
        h5_header             :varchar(255)     # path to h5 file
        triggertimes          :longblob         # triggertimes in each presentation
        triggervalues         :longblob         # values of the recorded triggers
        ch0_average           :longblob         # Stack median of channel 0
        ch1_average           :longblob         # Stack median of channel 1
        """
        return definition

    field_table = PlaceholderTable
    stimulus_table = PlaceholderTable
    userinfo_table = PlaceholderTable
    experiment_table = PlaceholderTable

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
            user_scanpathfunc          :varchar(255) 
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
            datestamp_d_m_y            :varchar(255) 
            timestamp_h_m_s_ms         :varchar(255) 
            inchan_pixbuflenlist       :varchar(255) 
            username                   :varchar(255) 
            guid                       :varchar(255) 
            origpixdatafilename        :varchar(255) 
            stimbuflenlist             :varchar(255) 
            callingprocessver          :varchar(255) 
            callingprocesspath         :varchar(255) 
            targetedstimdurlist        :varchar(255) 
            computername               :varchar(255) 
            scanm_pver_targetos        :varchar(255) 
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
            user_scanpathfunc          :varchar(255) 
            user_dzfrdecoded=0         :int        
            user_dxfrdecoded=0         :int        
            user_dyfrdecoded=0         :int        
            user_zeroz_v=0             :float        
            igorguiver                 :varchar(255) 
            user_comment=''            :varchar(255) 
            user_objective=''          :varchar(255) 
            realstimdurlist=""         :varchar(255) 
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

    def make(self, key):
        field = (self.field_table() & key).fetch1("field")
        stim_loc, field_loc, condition_loc = (self.userinfo_table() & key).fetch1(
            "stimulus_loc", "field_loc", "condition_loc")

        pre_data_path = os.path.join(
            (self.experiment_table() & key).fetch1('header_path'),
            (self.userinfo_table() & key).fetch1("pre_data_dir"))
        assert os.path.exists(pre_data_path), f"Error: Data folder does not exist: {pre_data_path}"

        h5_files = get_field_files(folder=pre_data_path, field=field, field_loc=field_loc, incl_hidden=False)
        stim_alias = (self.stimulus_table() & key).fetch1("alias").split('_')

        for h5_file in h5_files:
            split_string = h5_file[:h5_file.find(".h5")].split("_")
            stim = split_string[stim_loc] if stim_loc < len(split_string) else 'nostim'
            condition = split_string[condition_loc] if condition_loc < len(split_string) else 'control'

            primary_key = deepcopy(key)
            primary_key["condition"] = condition

            if stim.lower() in stim_alias:
                self.__add_presentation(
                    key=primary_key, filepath=os.path.join(pre_data_path, h5_file))

    def __add_presentation(self, key, filepath):
        pres_key = deepcopy(key)
        pres_key["h5_header"] = filepath

        with h5py.File(filepath, 'r', driver="stdio") as h5_file:
            key_triggertimes = [k for k in h5_file.keys() if k.lower() == 'triggertimes']

            if len(key_triggertimes) == 1:
                pres_key["triggertimes"] = h5_file[key_triggertimes[0]][()]
            elif len(key_triggertimes) == 0:
                pres_key["triggertimes"] = np.zeros(0)
            else:
                raise ValueError('Multiple triggertimes found')

            key_triggervalues = [k for k in h5_file.keys() if k.lower() == 'triggervalues']

            if len(key_triggervalues) == 1:
                pres_key["triggervalues"] = h5_file[key_triggervalues[0]][()]
                assert len(pres_key["triggertimes"]) == len(pres_key["triggervalues"]), 'Trigger mismatch'
            elif len(key_triggervalues) == 0:
                pres_key["triggervalues"] = np.zeros(0)
            else:
                raise ValueError('Multiple triggervalues found')

            os_params = dict()
            if 'OS_Parameters' in h5_file.keys():
                os_params.update(extract_h5_table('OS_Parameters', open_file=h5_file, lower_keys=True))

            ch0_stack, ch1_stack, wparams = \
                extract_ch0_ch1_stacks_from_h5(h5_file, ch0_name='wDataCh0', ch1_name='wDataCh1')

            pres_key["ch0_average"] = np.mean(ch0_stack, 2)
            pres_key["ch1_average"] = np.mean(ch1_stack, 2)

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

        self.insert1(pres_key)
        (self.ScanInfo() & key).insert1(scaninfo_key)

    def plot1(self, key=None, figsize=(16, 4)):
        if key is not None:
            key = {k: v for k, v in key.items() if k in self.primary_key}
        else:
            key = random.choice(self.fetch(*self.primary_key, as_dict=True))

        ch0_average = (self & key).fetch1("ch0_average").T
        ch1_average = (self & key).fetch1("ch1_average").T
        roi_mask = (self.field_table.RoiMask() & key).fetch1("roi_mask").T

        plot_field(ch0_average, ch1_average, roi_mask, title=key, figsize=figsize)
