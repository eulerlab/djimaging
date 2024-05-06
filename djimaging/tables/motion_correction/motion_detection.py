"""
from djimaging.tables import motion_correction

@schema
class MotionDetectionParams(motion_correction.MotionDetectionParamsTemplate):
    pass


@schema
class MotionDetection(motion_correction.MotionDetectionTemplate):
    mcorr_params_table = MotionDetectionParams
    presentation_table = Presentation
    userinfo_table = UserInfo
"""

import warnings
from abc import abstractmethod

import datajoint as dj
import numpy as np
from matplotlib import pyplot as plt

from djimaging.utils.dj_utils import get_primary_key
from djimaging.utils.filter_utils import lowpass_filter_trace
from djimaging.utils.plot_utils import set_long_title
from djimaging.utils.scanm import read_utils


class MotionDetectionParamsTemplate(dj.Lookup):
    database = ""

    @property
    def definition(self):
        definition = f"""
        mcorr_id : tinyint unsigned
        ---
        mcorr_method : varchar(191)
        mcorr_params : longblob
        """
        return definition

    def add(self, mcorr_method='jnormcorre', mcorr_params=None, mcorr_id=1, skip_duplicates=False):
        if mcorr_params is None:
            mcorr_params = dict()

        key = dict(mcorr_id=mcorr_id, mcorr_method=mcorr_method, mcorr_params=mcorr_params)
        self.insert1(key, skip_duplicates=skip_duplicates)


class MotionDetectionTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = f"""
        -> self.presentation_table
        -> self.mcorr_params_table
        ---
        max_shift_x : float  # Maximum shift after stimulus onset in x direction
        max_shift_y : float  # Maximum shift after stimulus onset in y direction
        shifts_x : mediumblob  # Shift in x direction
        shifts_y : mediumblob  # Shift in y direction
        idx_stim_onset : int  # Index of stimulus onset
        """
        return definition

    @property
    @abstractmethod
    def mcorr_params_table(self):
        pass

    @property
    @abstractmethod
    def presentation_table(self):
        pass

    @property
    @abstractmethod
    def userinfo_table(self):
        pass

    @property
    def key_source(self):
        try:
            return self.presentation_table().proj() * self.mcorr_params_table().proj()
        except (AttributeError, TypeError):
            pass

    def populate(
            self,
            *restrictions,
            suppress_errors=False,
            return_exception_objects=False,
            reserve_jobs=False,
            order="original",
            limit=None,
            max_calls=None,
            display_progress=False,
            processes=1,
            make_kwargs=None,
    ):
        if processes > 1:
            warnings.warn(
                "Parallel processing is not supported for this table. "
                "Setting processes to 1."
            )
            processes = 1

        return super().populate(
            *restrictions,
            suppress_errors=suppress_errors,
            return_exception_objects=return_exception_objects,
            reserve_jobs=reserve_jobs,
            order=order,
            limit=limit,
            max_calls=max_calls,
            display_progress=display_progress,
            processes=processes,
            make_kwargs=make_kwargs,
        )

    def make(self, key, verbose=False):

        if verbose:
            print(f"Computing motion detection for\n{key}")

        scan_frequency = (self.presentation_table.ScanInfo() & key).fetch1('scan_frequency')
        pres_data_file, triggertimes, pixel_size_um, npixartifact = (self.presentation_table & key).fetch1(
            'pres_data_file', 'triggertimes', 'pixel_size_um', 'npixartifact')
        data_stack_name = (self.userinfo_table() & key).fetch1("data_stack_name")
        from_raw_data = (self.presentation_table.raw_params_table & key).fetch1('from_raw_data')

        stacks, wparams = read_utils.load_stacks(
            pres_data_file, from_raw_data=from_raw_data, ch_names=(data_stack_name,))
        stack = stacks[data_stack_name].copy()[npixartifact:, :]

        idx_stim_onset = int(np.floor(triggertimes[0] * scan_frequency))

        mcorr_method, mcorr_params = (self.mcorr_params_table & key).fetch1('mcorr_method', 'mcorr_params')

        if mcorr_method.lower() == 'none':
            shifts_x, shifts_y = np.zeros(stack.shape[2], dtype=int), np.zeros(stack.shape[2], dtype=int)
        elif mcorr_method.lower() == 'jnormcorre':
            shifts_x, shifts_y = compute_shifts_jnormcorre(stack=stack, pixel_size_um=pixel_size_um, **mcorr_params)
        else:
            raise NotImplementedError()

        # Last values are often wrong, so we remove it
        shifts_x[-1] = shifts_x[-2]
        shifts_y[-1] = shifts_y[-2]

        # Define zero relative to stimulus onset
        shifts_x -= np.sort(shifts_x[idx_stim_onset:idx_stim_onset + 11])[5]
        shifts_y -= np.sort(shifts_y[idx_stim_onset:idx_stim_onset + 11])[5]

        # Get maximum shift after stimulus onset
        max_shift_x = np.max(np.abs(shifts_x[idx_stim_onset:]))
        max_shift_y = np.max(np.abs(shifts_y[idx_stim_onset:]))

        self.insert1(dict(key, shifts_x=shifts_x, shifts_y=shifts_y,
                          max_shift_x=max_shift_x, max_shift_y=max_shift_y, idx_stim_onset=idx_stim_onset))

    def plot1_stacks(self, key=None, n_average=10, fupsample=1, f_cutoff=3):
        key = get_primary_key(table=self, key=key)

        pres_data_file, npixartifact = (self.presentation_table & key).fetch1('pres_data_file', 'npixartifact')
        data_stack_name = (self.userinfo_table() & key).fetch1("data_stack_name")
        fs = (self.presentation_table.ScanInfo() & key).fetch1('scan_frequency')
        from_raw_data = (self.presentation_table.raw_params_table & key).fetch1('from_raw_data')

        stacks, wparams = read_utils.load_stacks(
            pres_data_file, from_raw_data=from_raw_data, ch_names=(data_stack_name,))
        stack = stacks[data_stack_name].copy()

        shifts_x, shifts_y = (self & key).fetch1('shifts_x', 'shifts_y')

        stack_corrected = correct_shifts_in_stack(
            stack=stack, shifts_x=shifts_x, shifts_y=shifts_y, fupsample=fupsample, fs=fs, f_cutoff=f_cutoff)

        plot_stack_and_corr_stack(
            stack[npixartifact:, :], stack_corrected[npixartifact:, :],
            shifts_x=shifts_x, shifts_y=shifts_y, n_average=n_average)

    def plot1_shifts(self, key=None):
        key = get_primary_key(table=self, key=key)

        fs = (self.presentation_table.ScanInfo() & key).fetch1('scan_frequency')
        triggertimes = (self.presentation_table & key).fetch1('triggertimes')

        shifts_x, shifts_y, max_shift_x, max_shift_y, idx_stim_onset = (self & key).fetch1(
            'shifts_x', 'shifts_y', 'max_shift_x', 'max_shift_y', 'idx_stim_onset')

        time = np.arange(shifts_x.size) / fs

        fig, axs = plt.subplots(1, 2, figsize=(10, 3))

        set_long_title(fig=fig, title=key)

        axs[0].set_title('shift_x')
        axs[0].plot(time, shifts_x, alpha=0.5, label='shift_x')
        axs[0].axhline(max_shift_x, c='r', ls='--')
        axs[0].axhline(-max_shift_x, c='r', ls='--')
        axs[0].axvline(time[idx_stim_onset], c='k', ls='--')
        axs[0].set(xlabel='Time [s]', ylabel='Shift [px]')
        axs[0].vlines(triggertimes, -max_shift_x, max_shift_x, color='r', ls='-', zorder=-2, alpha=0.5, lw=0.1,
                      label='trigger')

        axs[1].set_title('shift_y')
        axs[1].plot(time, shifts_y, alpha=0.5, label='shift_y')
        axs[1].axhline(max_shift_y, c='r', ls='--')
        axs[1].axhline(-max_shift_y, c='r', ls='--')
        axs[1].axvline(time[idx_stim_onset], c='k', ls='--')
        axs[1].set(xlabel='Time [s]', ylabel='Shift [px]')
        axs[1].vlines(triggertimes, -max_shift_y, max_shift_y, color='r', ls='-', zorder=-2, alpha=0.5, lw=0.1,
                      label='trigger')
        axs[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        return axs


def compute_shifts_jnormcorre(stack, pixel_size_um, **mcorr_params):
    try:
        from jnormcorre import correct_stack
    except ImportError as e:
        print('Please install correct version of jnormcorre')
        raise e
    q = np.percentile(stack, q=2.5)
    stack = np.clip(stack - q, 0, None)

    shifts_x, shifts_y = correct_stack.compute_shifts(stack, pixel_size_um=pixel_size_um, **mcorr_params)
    return shifts_x, shifts_y


def upsample_frame_nearest(arr, f):
    """Upsample frame by repeating pixels"""
    return np.repeat(np.repeat(arr, f, axis=0), f, axis=1)


def downsample_frame_nearest(arr, f):
    """Downsample frame by averaging pixels"""
    return np.mean(arr.reshape(
        arr.shape[0] // f, f, arr.shape[1] // f, f, arr.shape[2]), axis=(1, 3))


def correct_shifts_in_stack(stack, shifts_x, shifts_y, cval=np.min, fupsample=1, fs=8, f_cutoff=3):
    """Apply shift correction to stack"""
    assert stack.shape[2] == shifts_x.size, f"{stack.shape[2]} {shifts_x.size}"
    assert stack.shape[2] == shifts_y.size, f"{stack.shape[2]} {shifts_y.size}"

    fupsample = int(fupsample)

    if fupsample > 1:
        stack = upsample_frame_nearest(stack, fupsample)

    shifts_x_px = shifts_to_pixel_shifts(shifts_x * fupsample, fs=fs, f_cutoff=f_cutoff)
    shifts_y_px = shifts_to_pixel_shifts(shifts_y * fupsample, fs=fs, f_cutoff=f_cutoff)

    stack_corrected = np.full_like(stack, np.nan)

    for i, (shift_x, shift_y) in enumerate(zip(shifts_x_px, shifts_y_px)):
        _cval = cval(stack[:, :, i]) if callable(cval) else cval

        stack_corrected[:, :, i] = np.roll(stack[:, :, i], (-shift_x, -shift_y), axis=(0, 1))
        if shift_x > 0:
            stack_corrected[-shift_x:, :, i] = _cval
        elif shift_x < 0:
            stack_corrected[:-shift_x, :, i] = _cval
        if shift_y > 0:
            stack_corrected[:, -shift_y:, i] = _cval
        elif shift_y < 0:
            stack_corrected[:, :-shift_y, i] = _cval

    if fupsample > 1:
        stack_corrected = downsample_frame_nearest(stack_corrected, fupsample)

    return stack_corrected


def shifts_to_pixel_shifts(shifts, fs, f_cutoff=0.2):
    """Round shifts to integer (pixel) shifts"""
    if f_cutoff is not None:
        shifts = lowpass_filter_trace(shifts, fs, f_cutoff)

    shifts = shifts - np.mean(shifts)
    shifts = np.round(shifts)
    shifts = shifts - int(np.median(shifts))

    return shifts.astype(int)


def plot_stack_and_corr_stack(dat, dat_corr, axs=None, n_average=10, shifts_x=None, shifts_y=None):
    if axs is None:
        fig, axs = plt.subplots(2, 4, sharex='all', sharey='all', figsize=(16, 4))

    axs[0, 0].set_ylabel('raw')
    axs[1, 0].set_ylabel('corrected')

    for i, frame_i in enumerate(np.linspace(0, dat.shape[2] - 1 - n_average, len(axs.T), endpoint=True).astype(int)):
        avg_frame = np.mean(dat[:, :, frame_i:frame_i + n_average], axis=2)
        avg_frame_corr = np.mean(dat_corr[:, :, frame_i:frame_i + n_average], axis=2)

        vmin = np.minimum(np.min(avg_frame), np.min(avg_frame_corr))
        vmax = np.maximum(np.max(avg_frame), np.max(avg_frame_corr))

        title = f"frame={frame_i}"
        if shifts_x is not None and shifts_y is not None:
            title += f"\ndx={shifts_x[frame_i]:.3f} dy={shifts_y[frame_i]:.3f}"

        axs[0, i].set_title(title)
        axs[0, i].imshow(avg_frame.T, vmin=vmin, vmax=vmax, origin='lower')

        axs[1, i].imshow(avg_frame_corr.T, vmin=vmin, vmax=vmax, origin='lower')

        axs[0, i].grid(zorder=10, color='w')
        axs[1, i].grid(zorder=10, color='w')

    plt.tight_layout()

    return axs
