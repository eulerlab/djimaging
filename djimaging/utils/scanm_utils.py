import numpy as np
import h5py


def get_pixel_size_um(setupid: int, nypix: int, zoom: float) -> float:
    """Get width / height of a pixel in um"""
    setupid = int(setupid)

    assert 0.15 <= zoom <= 4, zoom
    assert setupid in [1, 2, 3], setupid
    assert 1 < nypix < 5000, nypix

    if setupid == 1:
        standard_pixel_size = 112. / nypix
    else:
        standard_pixel_size = 71.5 / nypix

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

    assert traces.shape == traces_times.shape

    roi2trace = dict()

    for roi_id in roi_ids:
        idx = roi_id - 1

        if traces.ndim == 3 and idx < traces.shape[-1]:
            trace = traces[:, :, idx]
            trace_times = traces_times[:, :, idx]
            trace_flag = 1
        elif traces.ndim == 2 and idx < traces.shape[-1]:
            trace = traces[:, idx]
            trace_times = traces_times[:, idx]
            trace_flag = 1
        else:
            trace_flag = 0
            trace = np.zeros(0)
            trace_times = np.zeros(0)

        roi2trace[roi_id] = dict(trace=trace, trace_times=trace_times, trace_flag=trace_flag)

    return roi2trace


def split_trace_by_reps(triggertimes, ntrigger_rep, times, trace_list, allow_drop_last=True):
    assert isinstance(trace_list, list)

    t_idxs = [np.argwhere(np.isclose(times, t, atol=1e-01))[0][0] for t in triggertimes[::ntrigger_rep]]

    assert len(t_idxs) > 1, 'Cannot split a single repetition'

    n_frames_per_rep = int(np.round(np.mean(np.diff(t_idxs))))

    for trace in trace_list:
        assert trace.shape == times.shape, 'Shapes do not match'

    if times[t_idxs[-1]:].size < n_frames_per_rep:
        assert allow_drop_last, 'Data incomplete, allow to drop last repetition or fix data'
        # if there are not enough data points after the last trigger,
        # remove the last trigger (e.g. if a chirp was cancelled)
        droppedlastrep_flag = 1
        t_idxs.pop(-1)
    else:
        droppedlastrep_flag = 0

    snippets_times = np.zeros((n_frames_per_rep, len(t_idxs)))
    triggertimes_snippets = np.zeros((ntrigger_rep, len(t_idxs)))
    snippets_list = [np.zeros((n_frames_per_rep, len(t_idxs))) for _ in range(len(trace_list))]

    for i, idx in enumerate(t_idxs):
        # Frames may be reused, this is not a standard reshaping
        snippets_times[:, i] = times[idx:idx + n_frames_per_rep]
        triggertimes_snippets[:, i] = triggertimes[i * ntrigger_rep:(i + 1) * ntrigger_rep]

        for j, (snippets, trace) in enumerate(zip(snippets_list, trace_list)):
            snippets_list[j][:, i] = trace[idx:idx + n_frames_per_rep]

    return snippets_times, triggertimes_snippets, snippets_list, droppedlastrep_flag
