import warnings

import h5py
import numpy as np

from djimaging.utils.data_utils import extract_h5_table
from djimaging.utils.misc_utils import CapturePrints
from djimaging.utils.trace_utils import find_closest


def get_npixartifact(setupid):
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


def split_trace_by_reps(trace, times, triggertimes, ntrigger_rep, delay=0., atol=0.1, allow_drop_last=True):
    """Split trace in snippets, using triggertimes"""

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


def load_traces_from_h5_file(filepath, roi_ids):
    """Extract traces from ScanM h5 file"""

    with h5py.File(filepath, "r", driver="stdio") as h5_file:
        traces, traces_times = extract_traces(h5_file)

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
            warnings.warn(f'NaN trace or tracetime in {filepath} for ROI{roi_id}.')
            valid_flag = 0

        roi2trace[roi_id] = dict(trace=trace, trace_times=trace_times, valid_flag=valid_flag)

    return roi2trace


def load_ch0_ch1_stacks_from_h5(filepath, ch0_name='wDataCh0', ch1_name='wDataCh1'):
    """Load high resolution stack channel 0 and 1 from h5 file"""

    with h5py.File(filepath, 'r', driver="stdio") as h5_file:
        ch0_stack, ch1_stack = extract_ch0_ch1_stacks_from_h5(h5_file, ch0_name=ch0_name, ch1_name=ch1_name)
        wparams = extract_w_params_from_h5(h5_file)

    check_dims_ch_stack_wparams(ch_stack=ch0_stack, wparams=wparams)
    check_dims_ch_stack_wparams(ch_stack=ch1_stack, wparams=wparams)

    return ch0_stack, ch1_stack, wparams


def check_dims_ch_stack_wparams(ch_stack, wparams):
    nxpix = wparams["user_dxpix"] - wparams["user_npixretrace"] - wparams["user_nxpixlineoffs"]
    nypix = wparams["user_dypix"]
    nzpix = wparams.get("user_dzpix", 0)

    assert ch_stack.shape[:2] in [(nxpix, nypix), (nxpix, nzpix)], \
        f'Stack shape error: {ch_stack.shape} not in [{(nxpix, nypix)}, {(nxpix, nzpix)}]'


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


def get_triggers_and_data(filepath):
    with h5py.File(filepath, 'r', driver="stdio") as h5_file:
        triggertimes, triggervalues = extract_triggers(h5_file)
        ch0_stack, ch1_stack = extract_ch0_ch1_stacks_from_h5(h5_file, ch0_name='wDataCh0', ch1_name='wDataCh1')
        wparams = extract_w_params_from_h5(h5_file)

    check_dims_ch_stack_wparams(ch_stack=ch0_stack, wparams=wparams)
    check_dims_ch_stack_wparams(ch_stack=ch1_stack, wparams=wparams)

    return triggertimes, triggervalues, ch0_stack, ch1_stack, wparams


def extract_os_params(h5_file_open) -> dict:
    os_params = dict()
    os_params_key = [k for k in h5_file_open.keys() if k.lower() == 'os_parameters']
    if len(os_params_key) > 0:
        os_params.update(extract_h5_table(os_params_key[0], open_file=h5_file_open, lower_keys=True))
    return os_params


def extract_triggers(h5_file_open):
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
        assert len(triggertimes) == len(triggervalues), 'Trigger mismatch'
    elif len(key_triggervalues) == 0:
        triggervalues = np.zeros(0)
    else:
        raise ValueError('Multiple triggervalues found')

    return triggertimes, triggervalues


def extract_traces(h5_file_open):
    """Read all traces and their times from file"""
    if "Traces0_raw" in h5_file_open.keys() and "Tracetimes0" in h5_file_open.keys():
        traces = np.asarray(h5_file_open["Traces0_raw"][()])
        traces_times = np.asarray(h5_file_open["Tracetimes0"][()])
    else:
        raise ValueError('Traces not found in h5 file.')

    assert traces.shape == traces_times.shape, 'Inconsistent traces and tracetimes shapes'

    return traces, traces_times


def extract_w_params_from_h5(h5_file_open):
    wparams = dict()
    wparamsstr_key = [k for k in h5_file_open.keys() if k.lower() == 'wparamsstr']
    if len(wparamsstr_key) > 0:
        wparams.update(extract_h5_table(wparamsstr_key[0], open_file=h5_file_open, lower_keys=True))
    wparamsnum_key = [k for k in h5_file_open.keys() if k.lower() == 'wparamsnum']
    if len(wparamsnum_key) > 0:
        wparams.update(extract_h5_table(wparamsnum_key[0], open_file=h5_file_open, lower_keys=True))
    return wparams


def extract_ch0_ch1_stacks_from_h5(h5_file_open, ch0_name='wDataCh0', ch1_name='wDataCh1'):
    ch0_stack = np.copy(h5_file_open[ch0_name])
    ch1_stack = np.copy(h5_file_open[ch1_name])
    assert ch0_stack.shape == ch1_stack.shape, 'Stacks must be of equal size'
    assert ch0_stack.ndim == 3, 'Stack does not match expected shape'
    return ch0_stack, ch1_stack


def compute_triggertimes(w_params, data, threshold=30_000, os_params=None):
    npix_x_offset_left = int(w_params['User_nXPixLineOffs'])
    npix_x_offset_right = int(w_params['User_nPixRetrace'])
    npix_x = int(w_params['User_dxPix'])
    npix_y = int(w_params['User_dyPix'])
    pix_dt = w_params['RealPixDur'] * 1e-6
    line_dt = pix_dt * npix_x
    frame_dt = pix_dt * npix_x * npix_y

    if os_params is not None:
        assert np.isclose(line_dt, os_params['LineDuration'])
        assert np.isclose(1. / os_params['samp_rate_Hz'], frame_dt)

    frame_dt_offset = (np.arange(npix_x * npix_y) * pix_dt).reshape(npix_y, npix_x).T
    frame_dt_offset = frame_dt_offset[npix_x_offset_left:-npix_x_offset_right]

    frame_times = np.arange(data['wDataCh2'].shape[2]) * frame_dt

    trigger_pixels = data['wDataCh2'] > threshold
    trigger_frame_idxs = np.where(np.diff(np.any(trigger_pixels, axis=(0, 1)).astype(int)) > 0)[0] + 1

    triggertimes = frame_times[trigger_frame_idxs] + np.array(
        [np.min(frame_dt_offset[trigger_pixels[:, :, idx]]) for idx in trigger_frame_idxs])

    return triggertimes


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
