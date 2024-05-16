import os

import h5py
import matplotlib.pyplot as plt
import numpy as np

from djimaging.utils.scanm import read_h5_utils, read_smp_utils, wparams_utils, setup_utils, traces_and_triggers_utils


class ScanMRecording:
    def __init__(self, filepath, setup_id,
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
            wparams = read_h5_utils.extract_wparams(h5_file)
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
        ch_stacks, wparams = read_smp_utils.load_all_stacks_and_wparams(self.filepath)

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
        self.scan_type_id = wparams.pop("user_scantype")

        # Pixel info
        self.pix_n_retrace = int(wparams.pop("user_npixretrace"))
        self.pix_n_line_offset = int(wparams.pop("user_nxpixlineoffs"))
        self.pix_nx_full = int(wparams.pop("user_dxpix"))
        self.pix_nx = int(self.pix_nx_full - self.pix_n_retrace - self.pix_n_line_offset)
        self.pix_ny = int(wparams.pop("user_dypix"))
        self.pix_nz = int(wparams.pop("user_dzpix"))
        self.pix_n_artifact = setup_utils.get_npixartifact(setupid=self.setup_id)

        # Scan info
        self.real_pixel_duration = wparams.pop('realpixdur') * 1e-6
        self.scan_line_duration = self.pix_nx_full * self.real_pixel_duration
        self.scan_n_lines = self.pix_nz if self.scan_type == 'xz' else self.pix_ny
        self.scan_period = self.scan_line_duration * self.scan_n_lines
        self.scan_frequency = 1. / self.scan_period

        # Objective info
        self.obj_zoom = wparams.pop("zoom")
        self.obj_angle_deg = wparams.pop("angle_deg")

        # Pixel sizes
        self.pix_dx_um = setup_utils.get_pixel_size_xy_um(zoom=self.obj_zoom, setupid=self.setup_id, npix=self.pix_nx)

        if self.scan_type in ['xy', 'xyz']:
            if wparams.get("user_aspectratiofr", 1.) != 1.:
                raise NotImplementedError(f"Aspect ratio is not 1 for {self.scan_type} scan.")
            self.pix_dy_um = self.pix_dx_um

        if self.scan_type in ['xz', 'xyz']:
            self.pix_dz_um = wparams['zstep_um']

        # Position data
        self.pos_x_um = wparams.pop("xcoord_um")
        self.pos_y_um = wparams.pop("ycoord_um")
        self.pos_z_um = wparams.pop("zcoord_um")

        # Other parameters
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

    def compute_triggers(self, trigger_threshold=None, time_precision=None, repeated_stim=None, ntrigger_rep=None):
        if trigger_threshold is not None:
            self.trigger_threshold = trigger_threshold

        if repeated_stim is not None:
            self.repeated_stim = repeated_stim

        if ntrigger_rep is not None:
            self.ntrigger_rep = ntrigger_rep

        if self.stimulator_delay is None:
            if self.date is None:
                raise ValueError("Please specify `date` or `stimulator_delay`.")
            self.stimulator_delay = setup_utils.get_stimulator_delay(self.date, setupid=self.setup_id)

        if self.frame_times is None or (time_precision != self.time_precision):
            self.compute_frame_times(time_precision=time_precision)

        if trigger_threshold == 'auto':
            vmin = self.ch_stacks[self.trigger_ch_name].min()
            vmax = self.ch_stacks[self.trigger_ch_name].max()

            if vmin < 25_000 and vmax > 35_000:
                self.trigger_threshold = 30_000
            else:
                self.trigger_threshold = 0.5 * (vmin + vmax)

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
