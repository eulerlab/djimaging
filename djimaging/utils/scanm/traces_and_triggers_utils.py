from typing import Optional, Tuple

import numpy as np

from djimaging.tables.motion_correction import motion_utils
from djimaging.utils.mask_format_utils import assert_igor_format
from djimaging.utils.scanm import roi_utils, wparams_utils, read_utils


def compute_triggers(stack: np.ndarray, frame_times: np.ndarray, frame_dt_offset: np.ndarray,
                     threshold: int = 30_000, stimulator_delay: float = 0.) \
        -> Tuple[np.ndarray, np.ndarray]:
    """Extract trigger onset times and values from a trigger channel stack.

    Parameters
    ----------
    stack : np.ndarray
        3-D array of shape ``(x, y, frames)`` representing the trigger channel.
    frame_times : np.ndarray
        1-D array of frame start times in seconds, shape ``(frames,)``.
    frame_dt_offset : np.ndarray
        2-D array of per-pixel time offsets within a frame in seconds,
        shape ``(x, y)``.
    threshold : int, optional
        Signal level above which a trigger onset is detected. Default is
        30 000.
    stimulator_delay : float, optional
        Delay in seconds added to each detected trigger time to account for
        the stimulator latency. Default is 0.

    Returns
    -------
    triggertimes : np.ndarray
        1-D array of trigger onset times in seconds.
    triggervalues : np.ndarray
        1-D array of signal amplitudes at each trigger onset.

    Raises
    ------
    ValueError
        If ``stack`` is not 3-D, if its shape does not match ``frame_times``
        or ``frame_dt_offset``, or if the computed stack times are
        non-monotonic.
    """
    if stack.ndim != 3:
        raise ValueError(f"stack must be 3d but ndim={stack.ndim}")
    if stack.shape[2] != frame_times.size:
        raise ValueError(f"stack shape not not match frame_times {stack.shape} {frame_times.shape}")
    if stack.shape[:2] != frame_dt_offset.shape:
        raise ValueError(f"stack shape not not match frame_dt_offset {stack.shape} {frame_dt_offset.shape}")

    stack_times = np.tile(np.atleast_3d(frame_dt_offset), (frame_times.shape[0])) + frame_times
    stack_times = stack_times.T.flatten()

    if np.any(np.diff(stack_times) < 0):
        raise ValueError(f"Unexpected value in stack_times. This is probably a bug that needs to be fixed.")

    min_diff_n_pixels = stack.shape[0] * 2  # Require at least 2 lines difference
    stack = stack.T.flatten()  # Order in time

    trigger_idxs = np.where((stack[1:] >= threshold) & (stack[:-1] < threshold))[0] + 1
    if np.any(np.diff(trigger_idxs) < min_diff_n_pixels):
        trigger_idxs = trigger_idxs[np.append(True, np.diff(trigger_idxs) >= min_diff_n_pixels)]

    # If trigger is on from the beginning it will be reset the first time a real trigger occurs.
    # In this case we miss the onset of the trigger.
    # In this case we have to derive the trigger-length from the other triggers.
    if stack[0] >= threshold and len(trigger_idxs) > 0:
        first_trigger_end_idx = np.argmax(stack < threshold)
        mean_trigger_dur = int(np.median([np.argmax(stack[trigger_idx + 1:] < threshold)
                                          for trigger_idx in trigger_idxs]))
        first_trigger_start_idx = first_trigger_end_idx - mean_trigger_dur
        trigger_idxs = np.append(first_trigger_start_idx, trigger_idxs)

    triggertimes = stack_times[trigger_idxs] + stimulator_delay
    triggervalues = stack[trigger_idxs]

    return triggertimes, triggervalues


def compute_frame_times(n_frames: int, pix_dt: float, npix_x: int, npix_2nd: int,
                        npix_x_offset_left: int, npix_x_offset_right: int,
                        precision: str = 'line') -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute timepoints of frames and relative delay of individual pixels.

    ``npix_2nd`` is ``npix_y`` for xy-scans and ``npix_z`` for xz-scans.

    Parameters
    ----------
    n_frames : int
        Total number of recorded frames.
    pix_dt : float
        Duration of one pixel acquisition in seconds.
    npix_x : int
        Total number of pixels along the fast (x) scan axis, including
        retrace and line-offset pixels.
    npix_2nd : int
        Number of pixels along the slow scan axis (y for xy-scans,
        z for xz-scans).
    npix_x_offset_left : int
        Number of line-offset pixels to skip at the beginning of each line
        (must be >= 0).
    npix_x_offset_right : int
        Number of retrace pixels to skip at the end of each line
        (must be >= 0).
    precision : str, optional
        Timing precision: ``'line'`` assigns the same timestamp to all pixels
        in a line; ``'pixel'`` assigns individual timestamps. Default is
        ``'line'``.

    Returns
    -------
    frame_times : np.ndarray
        1-D array of frame start times in seconds, shape ``(n_frames,)``.
    frame_dt_offset : np.ndarray
        2-D array of per-pixel time offsets within a frame in seconds,
        shape ``(npix_x_active, npix_2nd)`` where
        ``npix_x_active = npix_x - npix_x_offset_left - npix_x_offset_right``.
    frame_dt : float
        Duration of one complete frame in seconds.

    Raises
    ------
    ValueError
        If ``npix_x_offset_left`` or ``npix_x_offset_right`` is negative.
    """
    if npix_x_offset_left < 0:
        raise ValueError(f"npix_x_offset_left has to be positive or 0, but was {npix_x_offset_left}")
    if npix_x_offset_right < 0:
        raise ValueError(f"npix_x_offset_right has to be positive or 0, but was {npix_x_offset_right}")

    frame_dt = pix_dt * npix_x * npix_2nd

    frame_dt_offset = (np.arange(npix_x * npix_2nd) * pix_dt).reshape(npix_2nd, npix_x).T

    if precision == 'line':
        frame_dt_offset = np.tile(frame_dt_offset[0, :], (npix_x, 1))

    npix_x_offset_right_idx = -npix_x_offset_right if npix_x_offset_right > 0 else None
    frame_dt_offset = frame_dt_offset[npix_x_offset_left:npix_x_offset_right_idx]

    frame_times = np.arange(n_frames) * frame_dt

    return frame_times, frame_dt_offset, frame_dt


def compute_traces(stack: np.ndarray, roi_mask: np.ndarray, wparams: dict, precision: str = 'line') \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Extract per-ROI fluorescence traces and their timestamps from a stack.

    Parameters
    ----------
    stack : np.ndarray
        3-D array of shape ``(x, y, frames)`` containing pixel intensities.
    roi_mask : np.ndarray
        2-D integer ROI mask in Igor format, shape ``(x, y)``.
    wparams : dict
        Scan parameters used to compute frame timing via
        :func:`wparams_utils.compute_frame_times`.
    precision : str, optional
        Timing precision passed to frame-time computation:
        ``'line'`` (default) or ``'pixel'``.

    Returns
    -------
    roi_ids : np.ndarray
        1-D integer array of ROI IDs found in ``roi_mask``.
    traces : np.ndarray
        Array of shape ``(frames, n_rois)`` with mean pixel intensity per ROI
        per frame.
    traces_times : np.ndarray
        Array of shape ``(frames, n_rois)`` with the timestamp of each
        sample in seconds.
    frame_dt : float
        Duration of one frame in seconds.

    Raises
    ------
    ValueError
        If ``stack`` is not 3-D or its spatial dimensions do not match
        ``roi_mask``.
    """
    # TODO: Decide if tracetimes should be numerically equal:
    # Igor uses different method of defining the central pixel, leading to slightly different results.
    # Igor uses GeoC, i.e. it defines a pixel for each ROI.

    if stack.ndim != 3:
        raise ValueError(f"stack must be 3d but ndim={stack.ndim}")

    assert_igor_format(roi_mask)

    if stack.shape[:2] != roi_mask.shape:
        raise ValueError(f"xy-dim of stack roi_mask must match but shapes are {stack.shape} and {roi_mask.shape}")

    frame_times, frame_dt_offset, frame_dt = wparams_utils.compute_frame_times(
        wparams=wparams, n_frames=stack.shape[2], precision=precision)

    n_frames = stack.shape[2]
    roi_ids = roi_utils.extract_roi_ids(roi_mask, npixartifact=0)

    traces_times = np.full((n_frames, roi_ids.size), np.nan)
    traces = np.full((n_frames, roi_ids.size), np.nan)
    t0s = np.array([np.median(frame_dt_offset[roi_mask == roi_idx]) for roi_idx in roi_ids])

    for i, roi_id in enumerate(roi_ids):
        roi_mask_i = roi_mask == roi_id
        traces_times[:, i] = frame_times + t0s[i]
        traces[:, i] = np.mean(stack[roi_mask_i], axis=0)

    return roi_ids, traces, traces_times, frame_dt


def roi2trace_from_stack(
        filepath: str, roi_ids: np.ndarray, roi_mask: np.ndarray,
        data_stack_name: str, precision: str, from_raw_data: bool = False,
        shifts_x: Optional[np.ndarray] = None, shifts_y: Optional[np.ndarray] = None,
        shift_kws: Optional[dict] = None, accept_missing_rois: bool = False) -> Tuple[dict, float]:
    """Load a stack from disk and extract per-ROI traces, optionally after motion correction.

    Parameters
    ----------
    filepath : str
        Path to the recording file (.h5 or .smp).
    roi_ids : np.ndarray
        1-D array of positive integer ROI IDs to extract traces for.
    roi_mask : np.ndarray
        2-D integer ROI mask in Igor format used to average pixels per ROI.
    data_stack_name : str
        Name of the channel dataset to load (e.g. ``'wDataCh0'``).
    precision : str
        Timing precision passed to :func:`compute_traces`:
        ``'line'`` or ``'pixel'``.
    from_raw_data : bool, optional
        If True, load from a raw .smp file; otherwise load from an .h5 file.
        Default is False.
    shifts_x : np.ndarray, optional
        Per-frame shift values along x for motion correction. If None, no
        motion correction is applied.
    shifts_y : np.ndarray, optional
        Per-frame shift values along y for motion correction. If None, no
        motion correction is applied.
    shift_kws : dict, optional
        Additional keyword arguments passed to
        :func:`motion_utils.correct_shifts_in_stack`. Ignored if both
        ``shifts_x`` and ``shifts_y`` are None.
    accept_missing_rois : bool, optional
        If True, silently skip ROI IDs that are present in ``roi_ids`` but
        absent from the mask. Default is False.

    Returns
    -------
    roi2trace : dict
        Mapping from ROI ID (int) to a dict with keys ``'trace'``,
        ``'trace_times'``, and ``'trace_valid'``.
    frame_dt : float
        Duration of one frame in seconds.

    Raises
    ------
    ValueError
        If ``roi_ids`` contains IDs absent from the mask and
        ``accept_missing_rois`` is False.
    """
    ch_stacks, wparams = read_utils.load_stacks(filepath, from_raw_data, ch_names=(data_stack_name,))
    stack = ch_stacks[data_stack_name]

    if (shifts_x is not None) or (shifts_y is not None):
        if shift_kws is None:
            shift_kws = dict()
        stack = motion_utils.correct_shifts_in_stack(
            stack=stack, shifts_x=shifts_x, shifts_y=shifts_y, cval=np.min, **shift_kws)

    roi_ids_from_mask, traces, traces_times, frame_dt = compute_traces(
        stack=stack, roi_mask=roi_mask, wparams=wparams, precision=precision)

    if not set(abs(roi_ids)).issubset(set(abs(roi_ids_from_mask))):
        if not accept_missing_rois:
            raise ValueError(
                f"roi_ids from mask do not match provided roi_ids. "
                f"These ROIs are missing in the mask: {sorted(set(abs(roi_ids)) - set(abs(roi_ids_from_mask)))}.")

    roi2trace = roi_utils.get_roi2trace(traces=traces, traces_times=traces_times,
                                        roi_ids_traces=abs(roi_ids_from_mask), roi_ids_subset=abs(roi_ids))

    return roi2trace, frame_dt


def check_valid_triggers_rel_to_tracetime(
        trace_valid: int,
        trace_times: np.ndarray,
        triggertimes: np.ndarray,
) -> int:
    """Check whether all trigger times fall within the trace time window.

    Parameters
    ----------
    trace_valid : int
        1 if the trace is considered valid, 0 otherwise.
    trace_times : np.ndarray
        1-D array of trace sample timestamps in seconds.
    triggertimes : np.ndarray
        1-D array of trigger onset times in seconds.

    Returns
    -------
    int
        1 if the triggers are valid relative to the trace window,
        0 otherwise.
    """
    if len(triggertimes) == 0:
        return 1

    if trace_valid:
        if triggertimes[0] < trace_times[0]:
            return 0
        elif triggertimes[-1] > trace_times[-1]:
            return 0
        else:
            return 1
    else:
        return 0
