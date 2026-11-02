import numpy as np
from matplotlib import pyplot as plt

from djimaging.utils.filter_utils import lowpass_filter_trace


def compute_shifts_jnormcorre(stack: np.ndarray, pixel_size_um: float, **mcorr_params) -> tuple:
    """Compute per-frame motion shifts using the jnormcorre algorithm.

    The stack is clipped at its 2.5th percentile before passing it to
    ``jnormcorre.correct_stack.compute_shifts``.

    Args:
        stack: 3-D array of shape ``(x, y, t)`` containing the fluorescence stack.
        pixel_size_um: Physical size of a pixel in micrometers.
        **mcorr_params: Additional keyword arguments forwarded to
            ``jnormcorre.correct_stack.compute_shifts``.

    Returns:
        A 2-tuple ``(shifts_x, shifts_y)`` of 1-D integer arrays with per-frame
        shifts in the x and y directions respectively.

    Raises:
        ImportError: If ``jnormcorre`` is not installed.
    """
    try:
        from jnormcorre import correct_stack
    except ImportError as e:
        print('Please install correct version of jnormcorre')
        raise e
    q = np.percentile(stack, q=2.5)
    stack = np.clip(stack - q, 0, None)

    shifts_x, shifts_y = correct_stack.compute_shifts(stack, pixel_size_um=pixel_size_um, **mcorr_params)
    return shifts_x, shifts_y


def upsample_frame_nearest(arr: np.ndarray, f: int) -> np.ndarray:
    """Upsample a 2-D frame by repeating each pixel ``f`` times in both axes.

    Args:
        arr: 2-D input array to upsample.
        f: Integer upsampling factor.

    Returns:
        Upsampled 2-D array with shape ``(arr.shape[0] * f, arr.shape[1] * f)``.
    """
    return np.repeat(np.repeat(arr, f, axis=0), f, axis=1)


def downsample_frame_nearest(arr: np.ndarray, f: int) -> np.ndarray:
    """Downsample a 3-D stack by averaging blocks of ``f x f`` pixels in the spatial axes.

    Args:
        arr: 3-D array of shape ``(x, y, t)`` to downsample.
        f: Integer downsampling factor. Must evenly divide both spatial dimensions.

    Returns:
        Downsampled 3-D array of shape ``(x // f, y // f, t)``.
    """
    return np.mean(arr.reshape(
        arr.shape[0] // f, f, arr.shape[1] // f, f, arr.shape[2]), axis=(1, 3))


def correct_shifts_in_stack(
        stack: np.ndarray,
        shifts_x: np.ndarray,
        shifts_y: np.ndarray,
        cval=np.min,
        fs: float = 8,
        fupsample: int = 1,
        f_cutoff: float = 3,
) -> np.ndarray:
    """Apply shift correction to a fluorescence stack.

    Optionally upsamples the stack, low-pass-filters and rounds the shifts to
    integer pixel values, rolls each frame by its shift, and pads the vacated
    border pixels with ``cval``.

    Args:
        stack: 3-D array of shape ``(x, y, t)`` to correct.
        shifts_x: 1-D array of per-frame shifts in the x direction (in pixels,
            before upsampling).
        shifts_y: 1-D array of per-frame shifts in the y direction (in pixels,
            before upsampling).
        cval: Fill value for the padded border region. If callable, it is called
            with the current frame to produce the fill value (e.g. ``np.min``).
        fs: Acquisition frame rate in Hz, used for the low-pass filter.
        fupsample: Integer upsampling factor applied before correction and reversed
            afterwards. ``1`` disables upsampling.
        f_cutoff: Low-pass filter cut-off frequency in Hz applied to the shifts
            before rounding. ``None`` disables filtering.

    Returns:
        Motion-corrected stack with the same spatial shape as ``stack`` and dtype
        matching the input.
    """
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


def shifts_to_pixel_shifts(shifts: np.ndarray, fs: float, f_cutoff: float = 0.2) -> np.ndarray:
    """Round continuous shifts to integer pixel shifts after optional low-pass filtering.

    The mean is subtracted, shifts are rounded to the nearest integer, and the
    integer median is subtracted to centre the distribution around zero.

    Args:
        shifts: 1-D array of continuous shift values (in pixels).
        fs: Sampling rate in Hz used by the low-pass filter.
        f_cutoff: Low-pass filter cut-off frequency in Hz. If ``None``, no
            filtering is applied.

    Returns:
        1-D integer array of rounded pixel shifts.
    """
    if f_cutoff is not None:
        shifts = lowpass_filter_trace(shifts, fs, f_cutoff)

    shifts = shifts - np.mean(shifts)
    shifts = np.round(shifts)
    shifts = shifts - int(np.median(shifts))

    return shifts.astype(int)


def plot_stack_and_corr_stack(
        dat: np.ndarray,
        dat_corr: np.ndarray,
        axs: np.ndarray = None,
        n_average: int = 10,
        shifts_x: np.ndarray = None,
        shifts_y: np.ndarray = None,
) -> np.ndarray:
    """Plot side-by-side comparisons of raw and motion-corrected stack frames.

    Displays a 2-row grid of averaged frame snippets: the top row shows the raw
    stack and the bottom row shows the corrected stack. If ``shifts_x`` and
    ``shifts_y`` are provided they are annotated in each column title.

    Args:
        dat: 3-D raw fluorescence stack of shape ``(x, y, t)``.
        dat_corr: 3-D motion-corrected stack of the same shape as ``dat``.
        axs: Pre-existing 2-D axes array of shape ``(2, n_cols)``. If ``None``,
            a new figure with 4 columns is created.
        n_average: Number of consecutive frames to average per column.
        shifts_x: 1-D array of per-frame x shifts used for title annotation.
        shifts_y: 1-D array of per-frame y shifts used for title annotation.

    Returns:
        The axes array used for plotting.
    """
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
