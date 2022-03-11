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
