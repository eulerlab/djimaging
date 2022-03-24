import os
import numpy as np
import datajoint as dj
from copy import deepcopy
import h5py
from matplotlib import pyplot as plt

from djimaging.utils.data_utils import list_h5_files, extract_h5_table
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
        ntriggers             :int              # number  of triggers
        stack_average         :longblob         # data stack average
        scan_line_duration=-1 :float            # duration of one line scan
        scan_num_lines=-1     :float            # number of scan lines (in XZ scan)
        scan_frequency=-1     :float            # effective sampling frequency for each pixel in the scan field
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
            #meta data recorded in scamM header file
            -> master
            ---

            hdrleninvaluepairs         :float                      #
            hdrleninbytes              :float                      #
            minvolts_ao                :float                      #
            maxvolts_ao                :float                      #
            stimchanmask               :float                      #
            maxstimbufmaplen           :float                      #
            numberofstimbufs           :float                      #
            targetedpixdur_us          :float                      #
            minvolts_ai                :float                      #
            maxvolts_ai                :float                      #
            inputchanmask              :float                      #
            numberofinputchans         :float                      #
            pixsizeinbytes             :float                      #
            numberofpixbufsset         :float                      #
            pixeloffs                  :float                      #
            pixbufcounter              :float                      #
            user_scanmode              :float                      #
            user_dxpix                 :float                      #
            user_dypix                 :float                      #
            user_npixretrace           :float                      #
            user_nxpixlineoffs         :float                      #
            user_nypixlineoffs=0       :float                      # update 20171113
            user_divframebufreq        :float                      #
            user_scantype              :float                      #
            user_scanpathfunc          :varchar(255)               #
            user_nsubpixoversamp       :float                      #
            user_nfrperstep            :float                      #
            user_xoffset_v             :float                      #
            user_yoffset_v             :float                      #
            user_offsetz_v=0           :float                      #
            user_zoomz=0               :float                      # update 20171113
            user_noyscan=0             :float                      # update 20171113
            realpixdur                 :float                      #
            oversampfactor             :float                      #
            xcoord_um                  :float                      #
            ycoord_um                  :float                      #
            zcoord_um                  :float                      #
            zstep_um=0                 :float                      #
            zoom                       :float                      #
            angle_deg                  :float                      #
            datestamp_d_m_y            :varchar(255)               #
            timestamp_h_m_s_ms         :varchar(255)               #
            inchan_pixbuflenlist       :varchar(255)               #
            username                   :varchar(255)               #
            guid                       :varchar(255)               #
            origpixdatafilename        :varchar(255)               #
            stimbuflenlist             :varchar(255)               #
            callingprocessver          :varchar(255)               #
            callingprocesspath         :varchar(255)               #
            targetedstimdurlist        :varchar(255)               #
            computername               :varchar(255)               #
            scanm_pver_targetos        :varchar(255)               #
            user_zlensscaler=0         :float                      #
            user_stimbufperfr=0        :float                      #
            user_aspectratiofr=0       :float                      #
            user_zforfastscan=0        :float                      #
            user_zlensshifty=0         :float                      #
            user_nzpixlineoff=0        :float                      #
            user_dzpix=0               :float                      #
            user_setupid=0             :float                      #
            user_nzpixretrace=0        :float                      #
            user_laserwavelen_nm=0     :float                      #
            user_scanpathfunc          :varchar(255)               #
            user_dzfrdecoded=0         :float                      #
            user_dxfrdecoded=0         :float                      # update 20171113
            user_dyfrdecoded=0         :float                      # update 20171113
            user_zeroz_v=0             :float                      #
            igorguiver                 :varchar(255)               #
            user_comment=''            :varchar(255)               #
            user_objective=''          :varchar(255)               #
            realstimdurlist=""         :varchar(255)               # update 20180529
            user_ichfastscan=0         :float                      # update 20171113
            user_trajdefvrange_v=0     :float                      # update 20171113
            user_ntrajparams=0         :float                      # update 20171113
            user_offset_v=0            :float                      # update 20180529
            user_etl_polarity_v=0      :float                      # update 20180529
            user_etl_min_v=0           :float                      # update 20180529
            user_etl_max_v=0           :float                      # update 20180529
            user_etl_neutral_v=0       :float                      # update 20180529
            user_nimgperfr=0           :float                      # update 20180529
            """
            return definition

    def make(self, key):
        field = (self.field_table() & key).fetch1("field")
        stim_loc, field_loc, condition_loc = (self.userinfo_table() & key).fetch1(
            "stimulus_loc", "field_loc", "condition_loc")
        data_stack_name = (self.userinfo_table() & key).fetch1("data_stack_name")

        pre_data_path = (self.experiment_table() * self.field_table() & key).fetch1("pre_data_path")
        assert os.path.exists(pre_data_path), f"Could not read path: {pre_data_path}"

        h5_files = list_h5_files(folder=pre_data_path, hidden=False, field=field, field_loc=field_loc)
        stim_alias = (self.stimulus_table() & key).fetch1("alias").split('_')

        for h5_file in h5_files:
            split_string = h5_file[:h5_file.find(".h5")].split("_")
            stim = split_string[stim_loc] if stim_loc < len(split_string) else 'nostim'
            condition = split_string[condition_loc] if condition_loc < len(split_string) else 'none'

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
                pres_key["ntriggers"] = len(pres_key["triggertimes"])
            elif len(key_triggertimes) == 0:
                pres_key["triggertimes"] = np.zeros(0)
                pres_key["ntriggers"] = 0
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

            if 'OS_Parameters' in h5_file.keys():
                os_params = extract_h5_table('OS_Parameters', open_file=h5_file)
            else:
                print(f'Warning: No OS_Parameters found for {filepath}')
                os_params = dict()

            wparams = extract_h5_table('wParamsStr', 'wParamsNum', open_file=h5_file, lower_keys=True)

            # get scanning frequency
            if "LineDuration" in os_params:
                pres_key["scan_line_duration"] = os_params['LineDuration']
                roi_mask = (self.field_table.RoiMask() & key).fetch1('roi_mask')
                if roi_mask.size > 0:
                    pres_key["scan_num_lines"] = roi_mask.shape[-1]  # TODO: set to os info?
                    pres_key["scan_frequency"] = \
                        np.round(1 / (pres_key["scan_line_duration"] * pres_key["scan_num_lines"]), 2)

            nxpix = int(wparams["user_dxpix"] - wparams["user_npixretrace"] - wparams["user_nxpixlineoffs"])
            nypix = int(wparams["user_dypix"])

            stack = np.copy(h5_file[data_stack_name])
            assert stack.ndim == 3, 'Stack does not match expected shape'
            assert stack.shape[:2] == (nxpix, nypix), f'Stack shape error: {stack.shape} vs {(nxpix, nypix)}'
            pres_key["stack_average"] = np.mean(stack, 2)

        # extract params for scaninfo
        scaninfo_key = deepcopy(key)
        scaninfo_key.update(wparams)

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
