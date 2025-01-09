import numpy as np
from matplotlib import pyplot as plt

from djimaging.utils.filter_utils import lowpass_filter_trace


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


def correct_shifts_in_stack(stack, shifts_x, shifts_y, cval=np.min, fs=8, fupsample=1, f_cutoff=3):
    """Apply shift correction to stack"""
    assert stack.shape[2] == shifts_x.size, f"{stack.shape[2]} {shifts_x.size}"
    assert stack.shape[2] == shifts_y.size, f"{stack.shape[2]} {shifts_y.size}"

    fupsample = int(fupsample)

    if fupsample > 1:
        stack = upsample_frame_nearest(stack, fupsample)

    shifts_x_px = shifts_to_pixel_shifts(shifts_x * fupsample, fs=fs, f_cutoff=f_cutoff)
    shifts_y_px = shifts_to_pixel_shifts(shifts_y * fupsample, fs=fs, f_cutoff=f_cutoff)

    stack_corrected = np.full_like(stack, fill_value=-1)

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
