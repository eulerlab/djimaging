import numpy as np

from djimaging.utils.scanm import read_smp_utils, read_h5_utils, roi_utils, wparams_utils


def compute_triggers(stack: np.ndarray, frame_times: np.ndarray, frame_dt_offset: np.ndarray,
                     threshold: int = 30_000, stimulator_delay: float = 0.) \
        -> (np.ndarray, np.ndarray):
    """Extract triggertimes from stack"""
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
    if stack[0] >= threshold:
        first_trigger_end_idx = np.argmax(stack < threshold)
        mean_trigger_dur = int(np.median([np.argmax(stack[trigger_idx + 1:] < threshold)
                                          for trigger_idx in trigger_idxs]))
        first_trigger_start_idx = first_trigger_end_idx - mean_trigger_dur
        trigger_idxs = np.append(first_trigger_start_idx, trigger_idxs)

    triggertimes = stack_times[trigger_idxs] + stimulator_delay
    triggervalues = stack[trigger_idxs]

    return triggertimes, triggervalues


def compute_frame_times(n_frames: int, pix_dt: int, npix_x: int, npix_2nd: int,
                        npix_x_offset_left: int, npix_x_offset_right: int,
                        precision: str = 'line') -> (np.ndarray, np.ndarray):
    """Compute timepoints of frames and relative delay of individual pixels.
    npix_2nd can be npix_y (xy-scan) or npix_z (xz-scan)
    """
    frame_dt = pix_dt * npix_x * npix_2nd

    frame_dt_offset = (np.arange(npix_x * npix_2nd) * pix_dt).reshape(npix_2nd, npix_x).T

    if precision == 'line':
        frame_dt_offset = np.tile(frame_dt_offset[0, :], (npix_x, 1))

    frame_dt_offset = frame_dt_offset[npix_x_offset_left:-npix_x_offset_right]

    frame_times = np.arange(n_frames) * frame_dt

    return frame_times, frame_dt_offset


def compute_traces(stack: np.ndarray, roi_mask: np.ndarray, wparams: dict, precision: str = 'line') \
        -> (np.ndarray, np.ndarray):
    """Extract traces and tracetimes of stack"""
    # TODO: Decide if tracetimes should be numerically equal:
    # Igor uses different method of defining the central pixel, leading to slightly different results.
    # Igor uses GeoC, i.e. it defines a pixel for each ROI.

    if stack.ndim != 3:
        raise ValueError(f"stack must be 3d but ndim={stack.ndim}")
    roi_utils.check_if_scanm_roi_mask(roi_mask=roi_mask)
    if stack.shape[:2] != roi_mask.shape:
        raise ValueError(f"xy-dim of stack roi_mask must match but shapes are {stack.shape} and {roi_mask.shape}")

    frame_times, frame_dt_offset = wparams_utils.compute_frame_times(
        wparams=wparams, n_frames=stack.shape[2], precision=precision)

    n_frames = stack.shape[2]
    roi_idxs = roi_utils.extract_roi_idxs(roi_mask, npixartifact=0)

    traces_times = np.full((n_frames, roi_idxs.size), np.nan)
    traces = np.full((n_frames, roi_idxs.size), np.nan)

    for i, roi_idx in enumerate(roi_idxs):
        roi_mask_i = roi_mask == roi_idx
        traces_times[:, i] = frame_times + np.median(frame_dt_offset[roi_mask_i])
        traces[:, i] = np.mean(stack[roi_mask_i], axis=0)

    return traces, traces_times


def roi2trace_from_stack(filepath: str, roi_ids: np.ndarray, roi_mask: np.ndarray,
                         data_stack_name: str, precision: str, from_raw_data: bool = False):
    if not from_raw_data:
        ch_stacks, wparams = read_smp_utils.load_stacks_and_wparams(filepath, ch_names=(data_stack_name,))
    else:
        ch_stacks, wparams = read_h5_utils.load_stacks_and_wparams(filepath, ch_names=(data_stack_name,))

    traces, traces_times = compute_traces(
        stack=ch_stacks[data_stack_name], roi_mask=roi_mask, wparams=wparams, precision=precision)
    roi2trace = roi_utils.get_roi2trace(traces=traces, traces_times=traces_times, roi_ids=roi_ids)

    return roi2trace


def check_valid_triggers_rel_to_tracetime(trace_flag, trace_times, triggertimes):
    if len(triggertimes) == 0:
        return 1

    if trace_flag:
        if triggertimes[0] < trace_times[0]:
            return 0
        elif triggertimes[-1] > trace_times[-1]:
            return 0
        else:
            return 1
    else:
        return 0
