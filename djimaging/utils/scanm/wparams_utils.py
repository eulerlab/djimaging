import numpy as np

from djimaging.utils.scanm import traces_and_triggers_utils


def get_scan_type(wparams: dict, assume_lower=False) -> str:
    # ToDo: Does not work for Z-Stacks.

    if not assume_lower:
        wparams = {k.lower(): v for k, v in wparams.items()}

    npix_x = int(wparams.get('user_dxpix', 0))
    npix_y = int(wparams.get('user_dypix', 0))
    npix_z = int(wparams.get('user_dzpix', 0))

    if (npix_x > 1) and (npix_y > 1) and (npix_z <= 1):
        return 'xy'
    elif (npix_x > 1) and (npix_y <= 1) and (npix_z > 1):
        return 'xz'
    elif (npix_x > 1) and (npix_y > 1) and (npix_z > 1):
        return 'xyz'
    else:
        raise NotImplementedError(f"xyz = {npix_x}, {npix_y}, {npix_z}")


def compute_frame_times(wparams: dict, n_frames: int, precision: str = 'line') \
        -> (np.ndarray, np.ndarray):
    """Compute timepoints of frames and relative delay of individual pixels. Extract relevant parameters from wparams"""
    if precision not in ['line', 'pixel']:
        raise ValueError(f"precision must be either 'line' or 'pixel' but was {precision}")

    wparams = {k.lower(): v for k, v in wparams.items()}

    scan_type = get_scan_type(wparams=wparams, assume_lower=True)

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

    frame_times, frame_dt_offset, frame_dt = traces_and_triggers_utils.compute_frame_times(
        n_frames=n_frames, pix_dt=pix_dt, npix_x=npix_x, npix_2nd=npix_2nd,
        npix_x_offset_left=npix_x_offset_left, npix_x_offset_right=npix_x_offset_right, precision=precision)

    return frame_times, frame_dt_offset, frame_dt


def compute_triggers_from_wparams(
        stack: np.ndarray, wparams: dict, stimulator_delay: float,
        threshold: int = 30_000, precision: str = 'line') -> (np.ndarray, np.ndarray):
    """Extract triggertimes from stack, get parameters from wparams"""
    frame_times, frame_dt_offset, frame_dt = compute_frame_times(
        wparams=wparams, n_frames=stack.shape[2], precision=precision)
    triggertimes, triggervalues = traces_and_triggers_utils.compute_triggers(
        stack=stack, frame_times=frame_times, frame_dt_offset=frame_dt_offset,
        threshold=threshold, stimulator_delay=stimulator_delay)
    return triggertimes, triggervalues


def check_dims_ch_stack_wparams(ch_stack, wparams):
    """Check if the dimensions of a stack match what is expected from wparams"""
    nxpix = wparams["user_dxpix"] - wparams["user_npixretrace"] - wparams["user_nxpixlineoffs"]
    nypix = wparams["user_dypix"]
    nzpix = wparams.get("user_dzpix", 0)

    if not (ch_stack.shape[:2] in [(nxpix, nypix), (nxpix, nzpix)]):
        ValueError(f'Stack shape error: {ch_stack.shape} not in [{(nxpix, nypix)}, {(nxpix, nzpix)}]')
