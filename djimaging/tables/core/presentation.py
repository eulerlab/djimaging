import os
import numpy as np
import datajoint as dj
from copy import deepcopy
import h5py
from matplotlib import pyplot as plt

from djimaging.utils.data_utils import list_data_files, extract_h5_table
from djimaging.utils.dj_utils import PlaceholderTable


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
        stack_average         :longblob         # data stack average
        """
        return definition

    # TODO ADD scan_frequency BACK!!!

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
        data_stack_name = (self.userinfo_table() & key).fetch1("data_stack_name")

        pre_data_path = os.path.join(
            (self.experiment_table() & key).fetch1('header_path'),
            (self.userinfo_table() & key).fetch1("pre_data_dir"))
        assert os.path.exists(pre_data_path), f"Error: Data folder does not exist: {pre_data_path}"

        h5_files = list_data_files(folder=pre_data_path, hidden=False, field=field, field_loc=field_loc)
        stim_alias = (self.stimulus_table() & key).fetch1("alias").split('_')

        for h5_file in h5_files:
            split_string = h5_file[:h5_file.find(".h5")].split("_")
            stim = split_string[stim_loc] if stim_loc < len(split_string) else 'nostim'
            condition = split_string[condition_loc] if condition_loc < len(split_string) else 'control'

            primary_key = deepcopy(key)
            primary_key["condition"] = condition

            if stim.lower() in stim_alias:
                self.__add_presentation(
                    key=primary_key, filepath=os.path.join(pre_data_path, h5_file), data_stack_name=data_stack_name)

    def __add_presentation(self, key, filepath, data_stack_name):
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

            stack = np.copy(h5_file[data_stack_name])

            os_params = dict()
            if 'OS_Parameters' in h5_file.keys():
                os_params.update(extract_h5_table('OS_Parameters', open_file=h5_file, lower_keys=True))

            wparams = dict()
            if 'wParamsStr' in h5_file.keys():
                wparams.update(extract_h5_table('wParamsStr', open_file=h5_file, lower_keys=True))
                wparams.update(extract_h5_table('wParamsNum', open_file=h5_file, lower_keys=True))

            # Check stack average
            try:
                nxpix = wparams["user_dxpix"] - wparams["user_npixretrace"] - wparams["user_nxpixlineoffs"]
                nypix = wparams["user_dypix"]
                nzpix = wparams["user_dzpix"]

                assert stack.ndim == 3, 'Stack does not match expected shape'
                assert stack.shape[:2] in [(nxpix, nypix), (nxpix, nzpix)], \
                    f'Stack shape error: {stack.shape} not in [{(nxpix, nypix)}, {(nxpix, nzpix)}]'
            except KeyError:
                pass
            pres_key["stack_average"] = np.mean(stack, 2)

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

    def plot1(self, key):
        key = {k: v for k, v in key.items() if k in self.primary_key}

        fig, axs = plt.subplots(1, 2, figsize=(10, 3.5))
        stack_average = (self & key).fetch1("stack_average").T
        roi_mask = (self.field_table.RoiMask() & key).fetch1("roi_mask").T
        axs[0].imshow(stack_average)
        axs[0].set(title='stack_average')

        if roi_mask.size > 0:
            roi_mask_im = axs[1].imshow(roi_mask, cmap='jet')
            plt.colorbar(roi_mask_im, ax=axs[1])
            axs[1].set(title='roi_mask')
        plt.show()
