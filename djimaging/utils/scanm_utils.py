import h5py
import numpy as np

from djimaging.utils.data_utils import extract_h5_table
from djimaging.utils.misc_utils import CapturePrints


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


def get_pixel_size_xy_um(setupid: int, npix: int, zoom: float) -> float:
    """Get width / height of a pixel in um"""
    setupid = int(setupid)

    assert 0.15 <= zoom <= 4, zoom
    assert setupid in [1, 2, 3], setupid
    assert 1 <= npix < 5000, npix

    if setupid == 1:
        standard_pixel_size = 112. / npix
    else:
        standard_pixel_size = 71.5 / npix

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


def load_traces_from_h5_file(filepath, roi_ids):
    """Extract traces from ScanM h5 file"""

    with h5py.File(filepath, "r", driver="stdio") as h5_file:
        # read all traces and their times from file
        if "Traces0_raw" in h5_file.keys() and "Tracetimes0" in h5_file.keys():
            traces = np.asarray(h5_file["Traces0_raw"][()])
            traces_times = np.asarray(h5_file["Tracetimes0"][()])
        else:
            raise ValueError(f'Traces not found in {filepath}')

    assert np.all(roi_ids >= 1)
    assert traces.shape == traces_times.shape, f'Inconsistent traces and tracetimes in {filepath}'

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
            print(f'WARNING: NaN trace or tracetime in {filepath} for ROI{roi_id}.')
            valid_flag = 0

        roi2trace[roi_id] = dict(trace=trace, trace_times=trace_times, valid_flag=valid_flag)

    return roi2trace


def split_trace_by_reps(trace, times, triggertimes, ntrigger_rep, allow_drop_last=True):
    """Split trace in snippets, using triggertimes"""

    t_idxs = [np.argwhere(np.isclose(times, t, atol=1e-01))[0][0] for t in triggertimes[::ntrigger_rep]]

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


def load_ch0_ch1_stacks_from_h5(filepath, ch0_name='wDataCh0', ch1_name='wDataCh1'):
    """Load high resolution stack channel 0 and 1 from h5 file"""

    with h5py.File(filepath, 'r', driver="stdio") as h5_file:
        ch0_stack, ch1_stack, wparams = \
            extract_ch0_ch1_stacks_from_h5(h5_file, ch0_name=ch0_name, ch1_name=ch1_name)

    return ch0_stack, ch1_stack, wparams


def extract_ch0_ch1_stacks_from_h5(h5_file, ch0_name='wDataCh0', ch1_name='wDataCh1'):
    ch0_stack = np.copy(h5_file[ch0_name])
    ch1_stack = np.copy(h5_file[ch1_name])

    wparams = dict()
    if 'wParamsStr' in h5_file.keys():
        wparams.update(extract_h5_table('wParamsStr', open_file=h5_file, lower_keys=True))
        wparams.update(extract_h5_table('wParamsNum', open_file=h5_file, lower_keys=True))

    # Check stack average
    nxpix = wparams["user_dxpix"] - wparams["user_npixretrace"] - wparams["user_nxpixlineoffs"]
    nypix = wparams["user_dypix"]
    nzpix = wparams.get("user_dzpix", 0)

    assert ch0_stack.shape == ch1_stack.shape, 'Stacks must be of equal size'
    assert ch0_stack.ndim == 3, 'Stack does not match expected shape'
    assert ch0_stack.shape[:2] in [(nxpix, nypix), (nxpix, nzpix)], \
        f'Stack shape error: {ch0_stack.shape} not in [{(nxpix, nypix)}, {(nxpix, nzpix)}]'

    return ch0_stack, ch1_stack, wparams


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
        key_triggertimes = [k for k in h5_file.keys() if k.lower() == 'triggertimes']

        if len(key_triggertimes) == 1:
            triggertimes = h5_file[key_triggertimes[0]][()]
        elif len(key_triggertimes) == 0:
            triggertimes = np.zeros(0)
        else:
            raise ValueError('Multiple triggertimes found')

        key_triggervalues = [k for k in h5_file.keys() if k.lower() == 'triggervalues']

        if len(key_triggervalues) == 1:
            triggervalues = h5_file[key_triggervalues[0]][()]
            assert len(triggertimes) == len(triggervalues), 'Trigger mismatch'
        elif len(key_triggervalues) == 0:
            triggervalues = np.zeros(0)
        else:
            raise ValueError('Multiple triggervalues found')

        os_params = dict()
        if 'OS_Parameters' in h5_file.keys():
            os_params.update(extract_h5_table('OS_Parameters', open_file=h5_file, lower_keys=True))

        ch0_stack, ch1_stack, wparams = \
            extract_ch0_ch1_stacks_from_h5(h5_file, ch0_name='wDataCh0', ch1_name='wDataCh1')

    return triggertimes, triggervalues, ch0_stack, ch1_stack, wparams
