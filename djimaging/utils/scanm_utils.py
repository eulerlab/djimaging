import warnings

import h5py
import numpy as np

from djimaging.utils.data_utils import extract_h5_table
from djimaging.utils.misc_utils import CapturePrints
from djimaging.utils.trace_utils import find_closest


def get_stimulator_delay(date, setupid) -> float:
    raise NotImplementedError


def get_npixartifact(setupid):
    """Get number of lines that affected by the light artifact."""
    setupid = int(setupid)
    assert setupid in [1, 2, 3], setupid

    if setupid == 1:
        npixartifact = 1
    elif setupid == 3:
        npixartifact = 3
    elif setupid == 2:
        npixartifact = 4
    else:
        npixartifact = 0

    return npixartifact


def get_setup_xscale(setupid: int):
    """Get pixel scale in x and y for setup."""
    setupid = int(setupid)
    assert setupid in [1, 2, 3], setupid

    if setupid == 1:
        setup_xscale = 112.
    else:
        setup_xscale = 71.5

    return setup_xscale


def extract_roi_idxs(roi_mask, npixartifact=0):
    """Return roi idxs as in ROI mask (i.e. negative values)"""
    assert roi_mask.ndim == 2
    roi_idxs = np.unique(roi_mask[npixartifact:, :])
    roi_idxs = roi_idxs[roi_idxs < 0]  # remove background indexes (0 or 1)
    roi_idxs = roi_idxs[np.argsort(np.abs(roi_idxs))]  # Sort by value
    return roi_idxs.astype(int)


def get_pixel_size_xy_um(setupid: int, npix: int, zoom: float) -> float:
    """Get width / height of a pixel in um"""
    assert 0.15 <= zoom <= 4, zoom
    assert 1 <= npix < 5000, npix

    standard_pixel_size = get_setup_xscale(setupid) / npix
    pixel_size = standard_pixel_size / zoom

    return pixel_size


def get_retinal_position(rel_xcoord_um: float, rel_ycoord_um: float, rotation: float, eye: str) -> (float, float):
    """Get retinal position based on XCoord_um and YCoord_um relative to optic disk"""
    relx_rot = rel_xcoord_um * np.cos(np.deg2rad(rotation)) + rel_ycoord_um * np.sin(np.deg2rad(rotation))
    rely_rot = - rel_xcoord_um * np.sin(np.deg2rad(rotation)) + rel_ycoord_um * np.cos(np.deg2rad(rotation))

    # Get retinal position
    ventral_dorsal_pos_um = -relx_rot

    if eye == 'right':
        temporal_nasal_pos_um = rely_rot
    elif eye == 'left':
        temporal_nasal_pos_um = -rely_rot
    else:
        temporal_nasal_pos_um = np.nan

    return ventral_dorsal_pos_um, temporal_nasal_pos_um


def load_traces_from_h5(filepath):
    """Extract traces from ScanM h5 file"""
    with h5py.File(filepath, "r", driver="stdio") as h5_file:
        traces, traces_times = extract_traces(h5_file)
    return traces, traces_times


def get_roi2trace(traces, traces_times, roi_ids):
    """Get dict that holds traces and times accessible by roi_id"""
    assert np.all(roi_ids >= 1)

    roi2trace = dict()

    for roi_id in roi_ids:
        idx = roi_id - 1

        if traces.ndim == 3 and idx < traces.shape[-1]:
            trace = traces[:, :, idx]
            trace_times = traces_times[:, :, idx]
            valid_flag = 1
        elif traces.ndim == 2 and idx < traces.shape[-1]:
            trace = traces[:, idx]
            trace_times = traces_times[:, idx]
            valid_flag = 1
        else:
            valid_flag = 0
            trace = np.zeros(0)
            trace_times = np.zeros(0)

        if np.any(~np.isfinite(trace)) or np.any(~np.isfinite(trace_times)):
            warnings.warn(f'NaN trace or tracetime in for ROI{roi_id}.')
            valid_flag = 0

        roi2trace[roi_id] = dict(trace=trace, trace_times=trace_times, valid_flag=valid_flag)

    return roi2trace


def split_trace_by_reps(trace, times, triggertimes, ntrigger_rep, delay=0., atol=0.1, allow_drop_last=True):
    """Split trace in snippets, using triggertimes."""

    t_idxs = [find_closest(target=tt + delay, data=times, atol=atol, as_index=True)
              for tt in triggertimes[::ntrigger_rep]]

    assert len(t_idxs) > 1, 'Cannot split a single repetition'

    n_frames_per_rep = int(np.round(np.mean(np.diff(t_idxs))))

    assert trace.shape == times.shape, 'Shapes do not match'

    if times[t_idxs[-1]:].size < n_frames_per_rep:
        assert allow_drop_last, 'Data incomplete, allow to drop last repetition or fix data'
        # if there are not enough data points after the last trigger,
        # remove the last trigger (e.g. if a chirp was cancelled)
        droppedlastrep_flag = 1
        t_idxs.pop(-1)
    else:
        droppedlastrep_flag = 0

    snippets = np.zeros((n_frames_per_rep, len(t_idxs)))
    snippets_times = np.zeros((n_frames_per_rep, len(t_idxs)))
    triggertimes_snippets = np.zeros((ntrigger_rep, len(t_idxs)))

    # Frames may be reused, this is not a standard reshaping
    for i, idx in enumerate(t_idxs):
        snippets[:, i] = trace[idx:idx + n_frames_per_rep]
        snippets_times[:, i] = times[idx:idx + n_frames_per_rep]
        triggertimes_snippets[:, i] = triggertimes[i * ntrigger_rep:(i + 1) * ntrigger_rep]

    return snippets, snippets_times, triggertimes_snippets, droppedlastrep_flag


def load_stacks_from_h5(filepath, ch0_name='wDataCh0', ch1_name='wDataCh1') -> (dict, dict):
    """Load high resolution stack channel 0 and 1 from h5 file"""
    with h5py.File(filepath, 'r', driver="stdio") as h5_file:
        ch_stacks = extract_stacks_from_h5(h5_file, ch_names=(ch0_name, ch1_name))
        wparams = extract_wparams_from_h5(h5_file)

    for name, stack in ch_stacks.items():
        check_dims_ch_stack_wparams(ch_stack=stack, wparams=wparams)

    return ch_stacks, wparams


def check_dims_ch_stack_wparams(ch_stack, wparams):
    """Check if the dimensions of a stack match what is expected from wparams"""
    nxpix = wparams["user_dxpix"] - wparams["user_npixretrace"] - wparams["user_nxpixlineoffs"]
    nypix = wparams["user_dypix"]
    nzpix = wparams.get("user_dzpix", 0)

    if not (ch_stack.shape[:2] in [(nxpix, nypix), (nxpix, nzpix)]):
        ValueError(f'Stack shape error: {ch_stack.shape} not in [{(nxpix, nypix)}, {(nxpix, nzpix)}]')


def load_ch0_ch1_stacks_from_smp(filepath):
    """Load high resolution stack channel 0 and 1 from raw file"""
    try:
        from scanmsupport.scanm.scanm_smp import SMP
    except ImportError:
        print('Failed to load `scanmsupport`: Cannot load raw files.')
        return None, None, None

    scmf = SMP()

    with CapturePrints():
        scmf.loadSMH(filepath, verbose=False)
        scmf.loadSMP(filepath)

    ch0_stack = scmf.getData(ch=0, crop=True).T
    ch1_stack = scmf.getData(ch=1, crop=True).T

    wparams = dict()
    for k, v in scmf._kvPairDict.items():
        wparams[k.lower()] = v[2]

    wparams['user_dxpix'] = scmf.dxFr_pix
    wparams['user_dypix'] = scmf.dyFr_pix
    wparams['user_dzpix'] = scmf.dzFr_pix or 0
    wparams['user_npixretrace'] = scmf.dxRetrace_pix
    wparams['user_nxpixlineoffs'] = scmf.dxOffs_pix

    return ch0_stack, ch1_stack, wparams


def load_triggers_from_h5(filepath):
    with h5py.File(filepath, 'r', driver="stdio") as h5_file:
        triggertimes, triggervalues = extract_triggers(h5_file)
    return triggertimes, triggervalues


def extract_os_params(h5_file_open) -> dict:
    os_params = dict()
    os_params_key = [k for k in h5_file_open.keys() if k.lower() == 'os_parameters']
    if len(os_params_key) > 0:
        os_params.update(extract_h5_table(os_params_key[0], open_file=h5_file_open, lower_keys=True))
    return os_params


def extract_triggers(h5_file_open, check_triggervalues=False):
    key_triggertimes = [k for k in h5_file_open.keys() if k.lower() == 'triggertimes']

    if len(key_triggertimes) == 1:
        triggertimes = h5_file_open[key_triggertimes[0]][()]
    elif len(key_triggertimes) == 0:
        triggertimes = np.zeros(0)
    else:
        raise ValueError('Multiple triggertimes found')

    if "Tracetimes0" in h5_file_open.keys():  # Correct if triggertimes are in frames and not in time (old version)
        traces_times = np.asarray(h5_file_open["Tracetimes0"][()])
        correct_triggertimes = triggertimes[-1] > 2 * np.max(traces_times)
    else:
        correct_triggertimes = triggertimes[0] > 250 and triggertimes[-1] > 1000

    if correct_triggertimes:
        triggertimes = triggertimes / 500.

    key_triggervalues = [k for k in h5_file_open.keys() if k.lower() == 'triggervalues']

    if len(key_triggervalues) == 1:
        triggervalues = h5_file_open[key_triggervalues[0]][()]
    elif len(key_triggervalues) == 0:
        triggervalues = np.zeros(0)
    else:
        raise ValueError('Multiple triggervalues found')

    if check_triggervalues:
        assert len(triggertimes) == len(triggervalues), f'Mismatch: {len(triggertimes)} != {len(triggervalues)}'

    return triggertimes, triggervalues


def load_roi_mask_from_h5(filepath, ignore_not_found=False):
    with h5py.File(filepath, 'r', driver="stdio") as h5_file:
        roi_mask = extract_roi_mask(h5_file, ignore_not_found=ignore_not_found)
    return roi_mask


def extract_traces(h5_file_open):
    """Read all traces and their times from file"""
    if "Traces0_raw" in h5_file_open.keys() and "Tracetimes0" in h5_file_open.keys():
        traces = np.asarray(h5_file_open["Traces0_raw"][()])
        traces_times = np.asarray(h5_file_open["Tracetimes0"][()])
    else:
        raise ValueError('Traces not found in h5 file.')

    assert traces.shape == traces_times.shape, 'Inconsistent traces and tracetimes shapes'

    return traces, traces_times


def extract_roi_mask(h5_file_open, ignore_not_found=False):
    roi_keys = [k for k in h5_file_open.keys() if 'rois' in k.lower()]

    roi_mask = None
    if len(roi_keys) > 1:
        raise KeyError('Multiple ROI masks found in single file')
    elif len(roi_keys) == 1:
        roi_mask = np.copy(h5_file_open[roi_keys[0]])
        if np.all(roi_mask.astype(int) == roi_mask):
            roi_mask = roi_mask.astype(int)

        if roi_mask.size == 0:
            roi_mask = None

    if (roi_mask is None) and (not ignore_not_found):
        raise KeyError('No ROI mask found in single file')

    return roi_mask


def extract_wparams_from_h5(h5_file_open):
    wparams = dict()
    wparamsstr_key = [k for k in h5_file_open.keys() if k.lower() == 'wparamsstr']
    if len(wparamsstr_key) > 0:
        wparams.update(extract_h5_table(wparamsstr_key[0], open_file=h5_file_open, lower_keys=True))
    wparamsnum_key = [k for k in h5_file_open.keys() if k.lower() == 'wparamsnum']
    if len(wparamsnum_key) > 0:
        wparams.update(extract_h5_table(wparamsnum_key[0], open_file=h5_file_open, lower_keys=True))
    return wparams


def extract_stacks_from_h5(h5_file_open, ch_names=('wDataCh0', 'wDataCh1')) -> dict:
    ch_stacks = {ch_name: np.copy(h5_file_open[ch_name]) for ch_name in ch_names}
    for name, stack in ch_stacks.items():
        if stack.ndim != 3:
            raise ValueError(f"stack must be 3d but ndim={stack.ndim}")
        if stack.shape != ch_stacks[ch_names[0]].shape:
            raise ValueError('Stacks must be of equal size')
    return ch_stacks


def compute_frame_times_from_wparams(wparams: dict, n_frames: int, precision: str = 'line') \
        -> (np.ndarray, np.ndarray):
    """Compute timepoints of frames and relative delay of individual pixels. Extract relevant parameters from wparams"""
    if precision not in ['line', 'pixel']:
        raise ValueError(f"precision must be either 'line' or 'pixel' but was {precision}")

    wparams = {k.lower(): v for k, v in wparams.items()}

    npix_x_offset_left = int(wparams['user_nxpixlineoffs'])
    npix_x_offset_right = int(wparams['user_npixretrace'])
    npix_x = int(wparams['user_dxpix'])
    npix_y = int(wparams['user_dypix'])
    pix_dt = wparams['realpixdur'] * 1e-6

    frame_times, frame_dt_offset = compute_frame_times(
        n_frames=n_frames, pix_dt=pix_dt, npix_x=npix_x, npix_y=npix_y,
        npix_x_offset_left=npix_x_offset_left, npix_x_offset_right=npix_x_offset_right, precision=precision)

    return frame_times, frame_dt_offset


def compute_frame_times(n_frames: int, pix_dt: int, npix_x: int, npix_y: int,
                        npix_x_offset_left: int, npix_x_offset_right: int,
                        precision: str = 'line') -> (np.ndarray, np.ndarray):
    """Compute timepoints of frames and relative delay of individual pixels"""
    frame_dt = pix_dt * npix_x * npix_y

    frame_dt_offset = (np.arange(npix_x * npix_y) * pix_dt).reshape(npix_y, npix_x).T

    if precision == 'line':
        frame_dt_offset = np.tile(frame_dt_offset[0, :], (npix_x, 1))

    frame_dt_offset = frame_dt_offset[npix_x_offset_left:-npix_x_offset_right]

    frame_times = np.arange(n_frames) * frame_dt

    return frame_times, frame_dt_offset


def compute_triggertimes_from_wparams(
        stack: np.ndarray, wparams: dict, stimulator_delay: float,
        threshold: int = 30_000, precision: str = 'line') -> np.ndarray:
    """Extract triggertimes from stack, get parameters from wparams"""
    frame_times, frame_dt_offset = compute_frame_times_from_wparams(
        wparams=wparams, n_frames=stack.shape[2], precision=precision)
    triggertimes = compute_triggertimes(
        stack=stack, frame_times=frame_times, frame_dt_offset=frame_dt_offset, threshold=threshold)
    return triggertimes


def compute_triggertimes(stack: np.ndarray, frame_times: np.ndarray, frame_dt_offset: np.ndarray,
                         threshold: int = 30_000, stimulator_delay: float = 0.) -> np.ndarray:
    """Extract triggertimes from stack"""
    if stack.ndim != 3:
        raise ValueError(f"stack must be 3d but ndim={stack.ndim}")
    if stack.shape[2] != frame_times.size:
        raise ValueError(f"stack shape not not match frame_times {stack.shape} {frame_times.shape}")
    if stack.shape[:2] != frame_dt_offset.shape:
        raise ValueError(f"stack shape not not match frame_dt_offset {stack.shape} {frame_dt_offset.shape}")

    trigger_pixels = stack > threshold
    trigger_frame_idxs = np.where(np.diff(np.any(trigger_pixels, axis=(0, 1)).astype(int)) > 0)[0] + 1

    triggertimes = frame_times[trigger_frame_idxs] + np.array(
        [np.min(frame_dt_offset[trigger_pixels[:, :, idx]]) for idx in trigger_frame_idxs]) + stimulator_delay

    return triggertimes


def compute_traces(stack: np.ndarray, roi_mask: np.ndarray, wparams: dict, precision: str = 'line') \
        -> (np.ndarray, np.ndarray):
    """Extract traces and tracetimes of stack"""
    # TODO: Decide if tracetimes should be numerically equal:
    # Igor uses different method of defining the central pixel, leading to slightly different results.
    # Igor uses GeoC, i.e. it defines a pixel for each ROI.

    if stack.ndim != 3:
        raise ValueError(f"stack must be 3d but ndim={stack.ndim}")
    check_if_scanm_roi_mask(roi_mask=roi_mask)
    if stack.shape[:2] != roi_mask.shape:
        raise ValueError(f"xy-dim of stack roi_mask must match but shapes are {stack.shape} and {roi_mask.shape}")

    frame_times, frame_dt_offset = compute_frame_times_from_wparams(
        wparams=wparams, n_frames=stack.shape[2], precision=precision)

    n_frames = stack.shape[2]
    roi_idxs = extract_roi_idxs(roi_mask, npixartifact=0)

    traces_times = np.full((n_frames, roi_idxs.size), np.nan)
    traces = np.full((n_frames, roi_idxs.size), np.nan)

    for i, roi_idx in enumerate(roi_idxs):
        roi_mask_i = roi_mask == roi_idx
        traces_times[:, i] = frame_times + np.median(frame_dt_offset[roi_mask_i])
        traces[:, i] = np.mean(stack[roi_mask_i], axis=0)

    return traces, traces_times


def check_if_scanm_roi_mask(roi_mask: np.ndarray):
    """Test if ROI mask is in ScanM format, raise error otherwise"""
    if roi_mask.ndim != 2:
        raise ValueError(f"roi_mask must be 2d but ndim={roi_mask.ndim}")
    if np.max(roi_mask) != 1 or np.any(np.sum(roi_mask == 0)):
        raise ValueError('ROI mask contains unexpected values')


def get_roi_center(roi_mask: np.ndarray, roi_id: int) -> (float, float):
    binary_arr = -roi_mask == roi_id
    if not np.any(binary_arr):
        raise ValueError(f'roi_id={roi_id} not found in roi_mask with values {np.unique(roi_mask)}')
    x, y = np.mean(np.stack(np.where(binary_arr), axis=1), axis=0)
    return x, y


def get_roi_centers(roi_mask: np.ndarray, roi_ids: np.ndarray) -> np.ndarray:
    # TODO test if x, y should be swapped
    roi_centers = np.zeros((len(roi_ids), 2))
    for i, roi_id in enumerate(roi_ids):
        x, y = get_roi_center(roi_mask, roi_id)
        roi_centers[i, :] = (x, y)
    return roi_centers
