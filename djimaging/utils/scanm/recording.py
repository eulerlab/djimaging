"""
# wParamsNum:

'HdrLenInValuePairs',
'HdrLenInBytes',
'MinVolts_AO',
'MaxVolts_AO',
'StimChanMask',
'MaxStimBufMapLen',
'NumberOfStimBufs',
'TargetedPixDur_us',
'MinVolts_AI',
'MaxVolts_AI',
'InputChanMask',
'NumberOfInputChans',
'PixSizeInBytes',
'NumberOfPixBufsSet',
'PixelOffs',
'PixBufCounter',
'User_ScanMode',
'User_dxPix',
'User_dyPix',
'User_nPixRetrace',
'User_nXPixLineOffs',
'User_divFrameBufReq',
'User_ScanType',
'User_nSubPixOversamp',
'RealPixDur',
'OversampFactor',
'XCoord_um',
'YCoord_um',
'ZCoord_um',
'ZStep_um',
'Zoom',
'Angle_deg',
'User_NFrPerStep',
'User_XOffset_V',
'User_YOffset_V',
'User_dzPix',
'User_nZPixRetrace',
'User_nZPixLineOff',
'User_ZForFastScan',
'User_SetupID',
'User_LaserWaveLen_nm',
'User_ZLensScaler',
'User_ZLensShifty'

# wParamsStr:
'GUID',
'ComputerName',
'UserName',
'OrigPixDataFileName',
'DateStamp_d_m_y',
'TimeStamp_h_m_s_ms',
'ScanM_PVer_TargetOS',
'CallingProcessPath',
'CallingProcessVer',
'StimBufLenList',
'TargetedStimDurList',
'InChan_PixBufLenList',
'User_ScanPathFunc',
'IgorGUIVer',
'User_Comment',
'User_Objective'
"""

import os
import warnings

import h5py
import numpy as np

from djimaging.utils.scanm import read_h5_utils, read_smp_utils, wparams_utils, setup_utils, traces_and_triggers_utils


class ScanMRecording:
    def __init__(self, filepath, setup_id=None,
                 date=None, stimulator_delay=None,
                 roi_mask_ignore_not_found=True,
                 trigger_ch_name='wDataCh2', time_precision='line',
                 triggers_ignore_not_found=True, trigger_threshold='auto',
                 repeated_stim=None, ntrigger_rep=None):
        # File
        self.filepath = filepath
        self.filename = os.path.split(self.filepath)[1]
        self.filetype = os.path.splitext(filepath)[-1]

        # Meta info
        self.setup_id = setup_id
        self.date = date
        self.stimulator_delay = stimulator_delay
        self.scan_type = None
        self.scan_type_id = None

        # Channel data
        self.ch_stacks = None
        self.ch_names = None
        self.ch_n_frames = None

        # ROI mask
        self.roi_mask_ignore_not_found = roi_mask_ignore_not_found
        self.roi_mask = None

        # Timing data
        self.time_precision = time_precision
        self.frame_dt = None
        self.frame_times = None
        self.frame_dt_offset = None

        # Trigger data
        self.triggers_ignore_not_found = triggers_ignore_not_found
        self.trigger_ch_name = trigger_ch_name
        self.trigger_threshold = trigger_threshold
        self.repeated_stim = repeated_stim
        self.ntrigger_rep = ntrigger_rep
        self.trigger_times = None
        self.trigger_values = None
        self.trigger_valid = None

        # Pixel info
        self.pix_nx_full = None
        self.pix_n_retrace = None
        self.pix_n_line_offset = None
        self.pix_nx = None
        self.pix_ny = None
        self.pix_nz = None
        self.pix_n_artifact = None

        # Scan info
        self.real_pixel_duration = None
        self.scan_line_duration = None
        self.scan_n_lines = None
        self.scan_period = None
        self.scan_frequency = None

        # Objective info
        self.obj_zoom = None
        self.obj_angle_deg = None

        # Pixel sizes
        self.pix_dx_um = None
        self.pix_dy_um = None
        self.pix_dz_um = None

        # Position data
        self.pos_x_um = None
        self.pos_y_um = None
        self.pos_z_um = None

        # Meta data
        self.guid = None
        self.computer_name = None
        self.user_name = None
        self.orig_pix_data_filename = None
        self.date_stamp_d_m_y = None
        self.time_stamp_h_m_s_ms = None
        self.scanm_pver_targetos = None
        self.calling_process_path = None
        self.calling_process_ver = None
        self.stim_buf_len_list = None
        self.targeted_stim_dur_list = None
        self.in_chan_pix_buf_len_list = None
        self.user_scan_path_func = None
        self.igor_gui_ver = None
        self.user_comment = None
        self.user_objective = None

        # Other params
        self.wparams_other = None

        self.__load_from_file()

    def __repr__(self):
        if self.scan_type == 'xy':
            pix = f"[{self.pix_nx} x {self.pix_ny}]"
        elif self.scan_type == 'xz':
            pix = f"[{self.pix_nx} x {self.pix_nz}]"
        elif self.scan_type == 'xyz':
            pix = f"[{self.pix_nx} x {self.pix_ny} x {self.pix_nz}]"
        else:
            pix = "[???]"
        return f"ScanMRecording(`...\\{self.filename}`, {self.scan_type}: {pix})"

    def __eq__(self, other):
        if isinstance(other, ScanMRecording):
            if set(self.ch_names) != set(other.ch_names):
                return False
            for ch_name in self.ch_names:
                if not np.array_equal(self.ch_stacks[ch_name], other.ch_stacks[ch_name]):
                    return False

            if self.trigger_times is not None and other.trigger_times is not None:
                if not np.array_equal(self.trigger_times, other.trigger_times):
                    return False

            return True
        return False

    def __load_from_file(self):
        if self.filetype == '.smp':
            self.__load_from_smp_file()
        elif self.filetype == '.h5':
            self.__load_from_h5_file()

    def __load_from_h5_file(self):
        with h5py.File(self.filepath, 'r', driver="stdio") as h5_file:
            ch_stacks = read_h5_utils.extract_all_stack_from_h5(h5_file)
            wparams = read_h5_utils.extract_wparams(h5_file, lower_keys=False)
            roi_mask = read_h5_utils.extract_roi_mask(
                h5_file, ignore_not_found=self.roi_mask_ignore_not_found)
            trigger_times, trigger_values = read_h5_utils.extract_triggers(
                h5_file, ignore_not_found=self.triggers_ignore_not_found)

        self.ch_stacks = ch_stacks
        self.extract_ch_stacks_info(ch_stacks)
        self.extract_wparams_info(wparams)

        self.roi_mask = roi_mask
        self.trigger_times = trigger_times
        self.trigger_values = trigger_values

        if self.ntrigger_rep is not None:
            if self.repeated_stim:
                self.trigger_valid = self.trigger_times.size % self.ntrigger_rep == 0
            else:
                self.trigger_valid = self.trigger_times.size == self.ntrigger_rep

    def __load_from_smp_file(self):
        ch_stacks, wparams = read_smp_utils.load_all_stacks_and_wparams(self.filepath, lower_keys=False)

        # Channel info
        self.ch_stacks = ch_stacks
        self.extract_ch_stacks_info(ch_stacks)
        self.extract_wparams_info(wparams)

    def extract_ch_stacks_info(self, ch_stacks):
        self.ch_names = sorted(ch_stacks.keys())
        self.ch_n_frames = ch_stacks[self.ch_names[0]].shape[-1]

    def extract_wparams_info(self, wparams):
        # Scan type
        self.scan_type = wparams_utils.get_scan_type(wparams)
        self.scan_type_id = wparams.pop("User_ScanType")

        # Pixel info
        self.pix_n_retrace = int(wparams.pop("User_nPixRetrace"))
        self.pix_n_line_offset = int(wparams.pop("User_nXPixLineOffs"))
        self.pix_nx_full = int(wparams.pop("User_dxPix"))
        self.pix_nx = max(0, int(self.pix_nx_full - self.pix_n_retrace - self.pix_n_line_offset))
        self.pix_ny = int(wparams.pop("User_dyPix", 0))
        self.pix_nz = int(wparams.pop("User_dzPix", 0))
        self.pix_n_artifact = setup_utils.get_npixartifact(setupid=self.setup_id) if self.setup_id is not None else None

        # Scan info
        self.real_pixel_duration = wparams.pop('RealPixDur') * 1e-6
        self.scan_line_duration = self.pix_nx_full * self.real_pixel_duration
        self.scan_n_lines = self.pix_nz if self.scan_type == 'xz' else self.pix_ny
        self.scan_period = self.scan_line_duration * self.scan_n_lines
        self.scan_frequency = 1. / self.scan_period

        # Objective info
        self.obj_zoom = wparams.pop("Zoom")
        self.obj_angle_deg = wparams.pop("Angle_deg")

        # Pixel sizes
        if self.pix_nx > 0 and self.setup_id is not None:
            self.pix_dx_um = setup_utils.get_pixel_size_xy_um(
                zoom=self.obj_zoom, setupid=self.setup_id, npix=self.pix_nx)
        else:
            self.pix_dx_um = 0.

        if self.scan_type in ['xy', 'xyz']:
            if wparams.get("User_AspectRatioFr", 1.) != 1.:
                raise NotImplementedError(f"Aspect ratio is not 1 for {self.scan_type} scan.")
            self.pix_dy_um = self.pix_dx_um

        self.pix_dz_um = wparams.pop('ZStep_um')

        # Position data
        self.pos_x_um = wparams.pop("XCoord_um")
        self.pos_y_um = wparams.pop("YCoord_um")
        self.pos_z_um = wparams.pop("ZCoord_um")

        # Meta parameters
        self.guid = wparams.pop("GUID", "")
        self.computer_name = wparams.pop("ComputerName", "")
        self.user_name = wparams.pop("UserName", "")
        self.orig_pix_data_filename = wparams.pop("OrigPixDataFileName", "")
        self.date_stamp_d_m_y = wparams.pop("DateStamp_d_m_y", "")
        self.time_stamp_h_m_s_ms = wparams.pop("TimeStamp_h_m_s_ms", "")
        self.scanm_pver_targetos = wparams.pop("ScanM_PVer_TargetOS", "")
        self.calling_process_path = wparams.pop("CallingProcessPath", "")
        self.calling_process_ver = wparams.pop("CallingProcessVer", "")
        self.stim_buf_len_list = wparams.pop("StimBufLenList", "")
        self.targeted_stim_dur_list = wparams.pop("TargetedStimDurList", "")
        self.in_chan_pix_buf_len_list = wparams.pop("InChan_PixBufLenList", "")
        self.user_scan_path_func = wparams.pop("User_ScanPathFunc", "")
        self.igor_gui_ver = wparams.pop("IgorGUIVer", "")
        self.user_comment = wparams.pop("User_Comment", "")
        self.user_objective = wparams.pop("User_Objective", "")

        self.wparams_other = wparams

    def compute_frame_times(self, time_precision=None):
        if time_precision is not None:
            self.time_precision = time_precision

        self.frame_times, self.frame_dt_offset, self.frame_dt = traces_and_triggers_utils.compute_frame_times(
            n_frames=self.ch_n_frames,
            pix_dt=self.real_pixel_duration,
            npix_x=self.pix_nx_full,
            npix_2nd=self.pix_nz if self.scan_type == 'xz' else self.pix_ny,
            npix_x_offset_left=self.pix_n_line_offset,
            npix_x_offset_right=self.pix_n_retrace,
            precision=self.time_precision)

    def set_auto_trigger_threshold(self):
        vmin = self.ch_stacks[self.trigger_ch_name].min()
        vmax = self.ch_stacks[self.trigger_ch_name].max()

        if vmin < 25_000 and vmax > 35_000:
            self.trigger_threshold = 30_000
        else:
            self.trigger_threshold = 0.5 * (vmin + vmax)

    def compute_triggers(self, trigger_threshold=None, time_precision=None, repeated_stim=None, ntrigger_rep=None):
        if trigger_threshold is not None:
            self.trigger_threshold = trigger_threshold

        if repeated_stim is not None:
            self.repeated_stim = repeated_stim

        if ntrigger_rep is not None:
            self.ntrigger_rep = ntrigger_rep

        if self.stimulator_delay is None:
            if self.date is None or self.setup_id is None:
                raise ValueError("Please specify `date` or `stimulator_delay`.")
            self.stimulator_delay = setup_utils.get_stimulator_delay(self.date, setupid=self.setup_id)

        if self.frame_times is None or (time_precision != self.time_precision):
            self.compute_frame_times(time_precision=time_precision)

        if self.trigger_threshold == 'auto':
            self.set_auto_trigger_threshold()

        if self.ntrigger_rep is None or self.ntrigger_rep > 0:
            self.trigger_times, self.trigger_values = traces_and_triggers_utils.compute_triggers(
                stack=self.ch_stacks[self.trigger_ch_name],
                frame_times=self.frame_times,
                frame_dt_offset=self.frame_dt_offset,
                threshold=self.trigger_threshold,
                stimulator_delay=self.stimulator_delay)
        else:
            self.trigger_times, self.trigger_values = np.array([]), np.array([])

        if self.ntrigger_rep is not None:
            if self.repeated_stim:
                self.trigger_valid = self.trigger_times.size % self.ntrigger_rep == 0
            else:
                self.trigger_valid = self.trigger_times.size == self.ntrigger_rep

    def to_h5(self, filepath, include_wparams_num=True, include_wparams_str=True, include_triggers=True,
              overwrite=False):
        if not overwrite and os.path.isfile(filepath):
            raise FileExistsError(f"File already exists: {filepath}. Set `overwrite=True` to overwrite.")

        if include_triggers and (self.trigger_times is None or self.trigger_values is None):
            warnings.warn("Triggers are not computed yet. Doing that now...")
            self.compute_triggers()

        import h5py

        wparams_num = dict()
        if include_wparams_num:
            wparams_num.update(self.get_wparams_num_dict())

        wparams_str = dict()
        if include_wparams_str:
            wparams_str.update(self.get_wparams_str_dict())

        with h5py.File('data.h5', 'w') as f:
            for k, v in self.ch_stacks.items():
                f.create_dataset(k, data=v)

            if include_wparams_num:
                # TODO: Don't use groups but what Igor is using.
                wparams_num_group = f.create_group('wParamsNum')
                for key, value in wparams_num.items():
                    wparams_num_group.attrs[key] = value

            if include_wparams_str:
                # TODO: Don't use groups but what Igor is using.
                wparams_str_group = f.create_group('wParamsStr')
                for key, value in wparams_str.items():
                    wparams_str_group.attrs[key] = value

            if include_triggers:
                f.create_dataset('Triggertimes', data=self.trigger_times, dtype='f4')
                f.create_dataset('Triggervalues', data=self.trigger_values, dtype='f4')

    def get_wparams_num_dict(self, lower_keys=False):
        wparams_num = dict()
        wparams_num['User_dxPix'] = self.pix_nx_full
        wparams_num['User_dyPix'] = self.pix_ny
        wparams_num['User_dzPix'] = self.pix_nz
        wparams_num['User_nPixRetrace'] = self.pix_n_retrace
        wparams_num['User_nXPixLineOffs'] = self.pix_n_line_offset
        wparams_num['RealPixDur'] = self.real_pixel_duration
        wparams_num['Zoom'] = self.obj_zoom
        wparams_num['Angle_deg'] = self.obj_angle_deg
        wparams_num['XCoord_um'] = self.pos_x_um
        wparams_num['YCoord_um'] = self.pos_y_um
        wparams_num['ZCoord_um'] = self.pos_z_um
        wparams_num['ZStep_um'] = self.pix_dz_um
        wparams_num['User_ScanType'] = self.scan_type_id

        if lower_keys:
            wparams_num = {k.lower(): v for k, v in wparams_num.items()}

        return wparams_num

    def get_wparams_str_dict(self, lower_keys=False):
        wparams_str = dict()

        wparams_str['GUID'] = self.guid
        wparams_str['ComputerName'] = self.computer_name
        wparams_str['UserName'] = self.user_name
        wparams_str['OrigPixDataFileName'] = self.orig_pix_data_filename
        wparams_str['DateStamp_d_m_y'] = self.date_stamp_d_m_y
        wparams_str['TimeStamp_h_m_s_ms'] = self.time_stamp_h_m_s_ms
        wparams_str['ScanM_PVer_TargetOS'] = self.scanm_pver_targetos
        wparams_str['CallingProcessPath'] = self.calling_process_path
        wparams_str['CallingProcessVer'] = self.calling_process_ver
        wparams_str['StimBufLenList'] = self.stim_buf_len_list
        wparams_str['TargetedStimDurList'] = self.targeted_stim_dur_list
        wparams_str['InChan_PixBufLenList'] = self.in_chan_pix_buf_len_list
        wparams_str['User_ScanPathFunc'] = self.user_scan_path_func
        wparams_str['IgorGUIVer'] = self.igor_gui_ver
        wparams_str['User_Comment'] = self.user_comment

        if lower_keys:
            wparams_str = {k.lower(): v for k, v in wparams_str.items()}

        return wparams_str
