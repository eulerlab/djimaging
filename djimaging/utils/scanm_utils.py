import warnings

import os
import h5py
import numpy as np
import pandas as pd
from djimaging.utils.math_utils import truncated_vstack, padded_vstack

from djimaging.utils.data_utils import extract_h5_table
from djimaging.utils.misc_utils import CapturePrints
from djimaging.utils.trace_utils import find_closest


def get_stimulator_delay(date, setupid, file=None) -> float:
    """Get delay of stimulator which is the time that passes between the electrical trigger recorded in
    the third channel until the stimulus is actually displayed.
    For the light crafter these are 20-100 ms, for the Arduino it is zero."""
    if file is None:
        file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stimulator_delay.csv')

    eval_time = pd.Timestamp(date) + pd.to_timedelta(23, unit='h')  # Use end of day

    df = pd.read_csv(file, sep=';', parse_dates=['date'])
    df.loc[np.max(df.index) + 1] = {'date': eval_time}
    df.set_index('date', inplace=True)
    df.sort_index(ascending=True, inplace=True)
    df.ffill(inplace=True)
    stimulator_delay_ms = df.loc[eval_time, f"setup{setupid}"] / 1000.
    return stimulator_delay_ms


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
    # TODO: these values depend on the date. add date param and table, and read from table
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


def compute_t_idxs(trace, times, triggertimes, ntrigger_rep, delay=0., atol=0.1, allow_drop_last=True):
    t_idxs = [find_closest(target=tt + delay, data=times, atol=atol, as_index=True)
              for tt in triggertimes[::ntrigger_rep]]

    if len(t_idxs) > 1:
        n_frames_per_rep = int(np.median(np.diff(t_idxs)))  # Use median rep length to ignore outliers

        assert trace.shape == times.shape, 'Shapes do not match'

        if times[t_idxs[-1]:].size < n_frames_per_rep:
            assert allow_drop_last, 'Data incomplete, allow to drop last repetition or fix data'
            # if there are not enough data points after the last trigger,
            # remove the last trigger (e.g. if a chirp was cancelled)
            droppedlastrep_flag = True
            t_idxs.pop(-1)
        else:
            droppedlastrep_flag = False
    else:
        n_frames_per_rep = times.size - t_idxs[0]
        droppedlastrep_flag = False

    return t_idxs, n_frames_per_rep, droppedlastrep_flag


def split_trace_by_reps(trace, times, triggertimes, ntrigger_rep, delay=0., atol=0.1, allow_drop_last=True):
    """Split trace in snippets, using triggertimes."""

    t_idxs, n_frames_per_rep, droppedlastrep_flag = compute_t_idxs(
        trace, times, triggertimes, ntrigger_rep, delay=delay, atol=atol, allow_drop_last=allow_drop_last)

    snippets = np.zeros((n_frames_per_rep, len(t_idxs)))
    snippets_times = np.zeros((n_frames_per_rep, len(t_idxs)))
    snippets_triggertimes = np.zeros((ntrigger_rep, len(t_idxs)))

    # Frames may be reused, this is not a standard reshaping
    for i, idx in enumerate(t_idxs):
        snippets[:, i] = trace[idx:idx + n_frames_per_rep]
        snippets_times[:, i] = times[idx:idx + n_frames_per_rep]
        snippets_triggertimes[:, i] = triggertimes[i * ntrigger_rep:(i + 1) * ntrigger_rep]

    return snippets, snippets_times, snippets_triggertimes, droppedlastrep_flag


def group_reps_to_list(data, truncate_last_rep=True):
    rep_list = [sub_sub_rep for rep in data for sub_rep in rep for sub_sub_rep in sub_rep.T]
    if len(rep_list) > 1 and truncate_last_rep:
        rep_lens = [len(rep) for rep in rep_list]
        max_rep_len = max(rep_lens[:-1])
        if rep_lens[-1] > max_rep_len:
            rep_list[-1] = rep_list[-1][:max_rep_len]
    return rep_list


def split_trace_by_group_reps(
        trace, times, triggertimes, trial_info, delay=0., atol=0.1, rtol_trunc=0.1,
        allow_drop_last=True, squeeze_single_reps=False, stack_kind=None):
    """Split trace in snippets, using triggertimes. allow to have different stimulus groups"""

    ntrigger_rep = sum([trial_info_i['ntrigger'] for trial_info_i in trial_info])

    t_idxs, n_frames_per_rep, droppedlastrep_flag = compute_t_idxs(
        trace, times, triggertimes, ntrigger_rep, delay=delay, atol=atol, allow_drop_last=allow_drop_last)

    n_reps = len(t_idxs)

    names = np.unique([trial_info_i['name'] for trial_info_i in trial_info])

    # Create data structure
    group_snippets = {name: [[] for _ in range(n_reps)] for name in names}
    group_snippets_times = {name: [[] for _ in range(n_reps)] for name in names}
    group_snippets_triggertimes = {name: [[] for _ in range(n_reps)] for name in names}

    for rep in range(n_reps):
        trg_count = rep * ntrigger_rep
        for trial_info_i in trial_info:
            ntrigger_i = trial_info_i['ntrigger']
            ntrigger_split_i = trial_info_i.get('ntrigger_split', ntrigger_i)

            idx_start = find_closest(
                target=triggertimes[trg_count] + delay, data=times, atol=atol, as_index=True)

            try:
                idx_end = find_closest(
                    target=triggertimes[trg_count + ntrigger_split_i], data=times, as_index=True)
            except IndexError:
                idx_end = -1

            snippets_i, snippets_t_i, snippets_tt_i, _ = split_trace_by_reps(
                trace[idx_start:idx_end], times[idx_start:idx_end],
                triggertimes[trg_count:trg_count + ntrigger_i], ntrigger_split_i,
                delay=delay, atol=atol, allow_drop_last=False)

            group_snippets[trial_info_i['name']][rep].append(snippets_i)
            group_snippets_times[trial_info_i['name']][rep].append(snippets_t_i)
            group_snippets_triggertimes[trial_info_i['name']][rep].append(snippets_tt_i)

            trg_count += ntrigger_i

    # Stack repetitions
    if stack_kind is not None:
        for name in names:
            snippets_i = group_reps_to_list(group_snippets[name], truncate_last_rep=True)
            snippets_t_i = group_reps_to_list(group_snippets_times[name], truncate_last_rep=True)
            snippets_tt_i = group_reps_to_list(group_snippets_triggertimes[name], truncate_last_rep=False)

            stacked_snippets_triggertimes = truncated_vstack(snippets_tt_i, rtol=0.).T

            if stack_kind == 'pad':
                stacked_snippets = padded_vstack(snippets_i, cval=np.nan).T
                stacked_snippets_times = padded_vstack(snippets_t_i, cval=np.nan).T
            elif stack_kind == 'truncate':
                stacked_snippets = truncated_vstack(snippets_i, rtol=rtol_trunc).T
                stacked_snippets_times = truncated_vstack(snippets_t_i, rtol=rtol_trunc).T
            else:
                raise NotImplementedError(f'Unknown stack_kind: {stack_kind}')

            group_snippets[name] = stacked_snippets
            group_snippets_times[name] = stacked_snippets_times
            group_snippets_triggertimes[name] = stacked_snippets_triggertimes

    elif n_reps == 1 and squeeze_single_reps:
        # Squeeze single reps
        for name in names:
            group_snippets[name] = group_snippets[name][0]
            group_snippets_times[name] = group_snippets_times[name][0]
            group_snippets_triggertimes[name] = group_snippets_triggertimes[name][0]

    return group_snippets, group_snippets_times, group_snippets_triggertimes, droppedlastrep_flag


def load_stacks(filepath, from_raw_data, ch_names=('wDataCh0', 'wDataCh1')) -> (dict, dict):
    if from_raw_data:
        ch_stacks, wparams = load_stacks_from_smp(filepath, ch_names=ch_names)
    else:
        ch_stacks, wparams = load_stacks_from_h5(filepath, ch_names=ch_names)
    return ch_stacks, wparams


def load_stacks_from_h5(filepath, ch_names=('wDataCh0', 'wDataCh1')) -> (dict, dict):
    """Load high resolution stack channel 0 and 1 from h5 file"""
    with h5py.File(filepath, 'r', driver="stdio") as h5_file:
        ch_stacks = extract_stacks_from_h5(h5_file, ch_names=ch_names)
        wparams = extract_wparams_from_h5(h5_file)

    for name, stack in ch_stacks.items():
        check_dims_ch_stack_wparams(ch_stack=stack, wparams=wparams)

    return ch_stacks, wparams


def load_smp_file(raw_file_path):
    """Load smp file"""
    try:
        from scanmsupport.scanm.scanm_smp import SMP
    except ImportError:
        raise ImportError('Custom package `scanmsupport is not installed. Cannot load SMP files.')

    raw_file = SMP()

    with CapturePrints():
        raw_file.loadSMH(raw_file_path, verbose=False)
        raw_file.loadSMP(raw_file_path)

    return raw_file


def load_wparams_from_smp(raw_file, return_file=True):
    """Load wparams from raw file"""

    try:
        from scanmsupport.scanm.scanm_smp import SMP
    except ImportError:
        raise ImportError('Custom package `scanmsupport is not installed. Cannot load SMP files.')

    if not isinstance(raw_file, SMP):
        raw_file = load_smp_file(raw_file)

    wparams = dict()
    for k, v in raw_file._kvPairDict.items():
        wparams[k.lower()] = v[2]

    wparams['user_dxpix'] = raw_file.dxFr_pix
    wparams['user_dypix'] = raw_file.dyFr_pix
    wparams['user_dzpix'] = raw_file.dzFr_pix or 0
    wparams['user_npixretrace'] = raw_file.dxRetrace_pix
    wparams['user_nxpixlineoffs'] = raw_file.dxOffs_pix
    wparams['user_scantype'] = raw_file.scanType

    # Handle different naming conventions
    renaming_dict = {
        "realpixelduration_µs": 'realpixdur',
    }

    for k, v in renaming_dict.items():
        if k in wparams.keys():
            wparams[v] = wparams.pop(k)

    if return_file:
        return wparams, raw_file
    else:
        return wparams


def load_stacks_from_smp(raw_file, ch_names=('wDataCh0', 'wDataCh1')):
    """Load stacks channel 0 and 1 from raw file"""
    wparams, raw_file = load_wparams_from_smp(raw_file)

    ch_stacks = dict()
    ch_name_main = ch_names[0]
    ch_stacks[ch_name_main] = raw_file.getData(ch=int(ch_name_main[-1]), crop=True).T
    for ch_name in ch_names[1:]:
        try:
            ch_stacks[ch_name] = raw_file.getData(ch=int(ch_name[-1]), crop=True).T
        except IndexError:
            warnings.warn(f'Failed to load channel={ch_name}. Set values to zero.')
            ch_stacks[ch_name] = np.zeros_like(ch_stacks[ch_name_main])

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
    ch_stacks, wparams = load_stacks_from_smp(filepath, ch_names=('wDataCh0', 'wDataCh1'))
    return ch_stacks['wDataCh0'], ch_stacks['wDataCh1'], wparams


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


def get_scan_type_from_wparams(wparams: dict, assume_lower=False) -> str:
    # ToDo: Does not work for Z-Stacks.

    if assume_lower:
        wparams = {k.lower(): v for k, v in wparams.items()}

    npix_x = int(wparams['user_dxpix'])
    npix_y = int(wparams['user_dypix'])
    npix_z = int(wparams['user_dzpix'])

    if (npix_x > 1) and (npix_y > 1) and (npix_z <= 1):
        return 'xy'
    elif (npix_x > 1) and (npix_y <= 1) and (npix_z > 1):
        return 'xz'
    elif (npix_x > 1) and (npix_y > 1) and (npix_z > 1):
        return 'xyz'
    else:
        raise NotImplementedError(f"xyz = {npix_x}, {npix_y}, {npix_z}")


def compute_frame_times_from_wparams(wparams: dict, n_frames: int, precision: str = 'line') \
        -> (np.ndarray, np.ndarray):
    """Compute timepoints of frames and relative delay of individual pixels. Extract relevant parameters from wparams"""
    if precision not in ['line', 'pixel']:
        raise ValueError(f"precision must be either 'line' or 'pixel' but was {precision}")

    wparams = {k.lower(): v for k, v in wparams.items()}

    scan_type = get_scan_type_from_wparams(wparams=wparams, assume_lower=True)

    npix_x_offset_left = int(wparams['user_nxpixlineoffs'])
    npix_x_offset_right = int(wparams['user_npixretrace'])
    npix_x = int(wparams['user_dxpix'])
    pix_dt = wparams['realpixdur'] * 1e-6

    if scan_type == 'xy':
        npix_2nd = int(wparams['user_dypix'])
    elif scan_type == 'xz':
        npix_2nd = int(wparams['user_dzpix'])
    else:
        raise NotImplementedError(scan_type)

    frame_times, frame_dt_offset = compute_frame_times(
        n_frames=n_frames, pix_dt=pix_dt, npix_x=npix_x, npix_2nd=npix_2nd,
        npix_x_offset_left=npix_x_offset_left, npix_x_offset_right=npix_x_offset_right, precision=precision)

    return frame_times, frame_dt_offset


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


def compute_triggers_from_wparams(
        stack: np.ndarray, wparams: dict, stimulator_delay: float,
        threshold: int = 30_000, precision: str = 'line') -> (np.ndarray, np.ndarray):
    """Extract triggertimes from stack, get parameters from wparams"""
    frame_times, frame_dt_offset = compute_frame_times_from_wparams(
        wparams=wparams, n_frames=stack.shape[2], precision=precision)
    triggertimes, triggervalues = compute_triggers(
        stack=stack, frame_times=frame_times, frame_dt_offset=frame_dt_offset,
        threshold=threshold, stimulator_delay=stimulator_delay)
    return triggertimes, triggervalues


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
    roi_mask = np.asarray(roi_mask)

    if roi_mask.ndim != 2:
        raise ValueError(f"roi_mask must be 2d but ndim={roi_mask.ndim}")

    if np.any(roi_mask != roi_mask.astype(int)):
        raise ValueError(f'ROI mask contains non-integers: {np.unique(roi_mask)}')

    if np.max(roi_mask) == 0 and not np.any(roi_mask > 0) and np.any(roi_mask < 0):
        # Allow zero background mask
        return
    elif np.max(roi_mask) == 1 and not np.any(roi_mask == 0) and np.any(roi_mask < 0):
        # Allow one background mask, then zero should not be used
        return
    else:
        raise ValueError(f'ROI mask contains unexpected values {np.unique(roi_mask)}')


def compare_roi_masks(roi_mask1: np.ndarray, roi_mask2: np.ndarray, max_shift=4) -> str:
    """Test if two roi masks are the same"""
    check_if_scanm_roi_mask(roi_mask1)
    check_if_scanm_roi_mask(roi_mask2)

    if roi_mask1.shape != roi_mask2.shape:
        return 'different'
    if np.all(roi_mask1 == roi_mask2):
        return 'same'

    max_shift_x = np.minimum(max_shift, roi_mask1.shape[0] - 2)
    max_shift_y = np.minimum(max_shift, roi_mask1.shape[1] - 2)

    for dx in range(-max_shift_x, max_shift_x + 1):
        for dy in range(-max_shift_y, max_shift_y + 1):
            shifted1 = roi_mask1[dx:, dy:]
            dx = -roi_mask2.shape[0] if dx == 0 else dx  # Handle zero case
            dy = -roi_mask2.shape[0] if dy == 0 else dy
            shifted2 = roi_mask2[:-dx, :-dy]
            if np.all(shifted1 == shifted2):
                return 'shifted'
    return 'different'


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


def get_rel_roi_pos(roi_id, roi_mask, pixel_size_um, ang_deg=0.):
    """Get position relative to plotting axis"""
    # Get relative position in pixel space
    pix_x, pix_y = get_roi_center(roi_mask, roi_id)

    # Get offset to center in um
    dx_um = float((pix_x - roi_mask.shape[0] / 2) * pixel_size_um)
    dy_um = float((pix_y - roi_mask.shape[1] / 2) * pixel_size_um)

    # Rotate around center
    if ang_deg != 0.:
        ang_rad = ang_deg * np.pi / 180.
        dx_um_rot = dx_um * np.cos(ang_rad) - dy_um * np.sin(ang_rad)
        dy_um = dx_um * np.sin(ang_rad) + dy_um * np.cos(ang_rad)
        dx_um = dx_um_rot

    return dx_um, dy_um


def roi2trace_from_h5_file(filepath: str, roi_ids: np.ndarray):
    with h5py.File(filepath, "r", driver="stdio") as h5_file:
        traces, traces_times = extract_traces(h5_file)
    roi2trace = get_roi2trace(traces=traces, traces_times=traces_times, roi_ids=roi_ids)
    return roi2trace


def roi2trace_from_stack(filepath: str, roi_ids: np.ndarray, roi_mask: np.ndarray,
                         data_stack_name: str, precision: str, from_raw_data: bool = False):
    if not from_raw_data:
        ch_stacks, wparams = load_stacks_from_h5(filepath, ch_names=(data_stack_name,))
    else:
        ch_stacks, wparams = load_stacks_from_smp(filepath, ch_names=(data_stack_name,))

    traces, traces_times = compute_traces(
        stack=ch_stacks[data_stack_name], roi_mask=roi_mask, wparams=wparams, precision=precision)
    roi2trace = get_roi2trace(traces=traces, traces_times=traces_times, roi_ids=roi_ids)

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
