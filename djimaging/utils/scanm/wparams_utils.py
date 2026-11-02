from typing import Tuple

import numpy as np

from djimaging.utils.scanm import traces_and_triggers_utils


def get_scan_type(wparams: dict, assume_lower: bool = False) -> str:
    """Determine the scan type from wparams.

    Parameters
    ----------
    wparams : dict
        Dictionary of scan parameters (wParamsNum / wParamsStr).
    assume_lower : bool, optional
        If True, assume keys are already lower-case. Default is False.

    Returns
    -------
    str
        One of ``'xy'``, ``'xz'``, or ``'xyz'``.

    Raises
    ------
    NotImplementedError
        If the pixel-dimension combination is not recognised.
    """
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
        -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute timepoints of frames and relative delay of individual pixels.

    Extracts the relevant timing parameters from ``wparams`` and delegates
    to :func:`traces_and_triggers_utils.compute_frame_times`.

    Parameters
    ----------
    wparams : dict
        Dictionary of scan parameters containing at minimum
        ``'user_nxpixlineoffs'``, ``'user_npixretrace'``, ``'user_dxpix'``,
        ``'realpixdur'``, and either ``'user_dypix'`` or ``'user_dzpix'``.
    n_frames : int
        Number of recorded frames.
    precision : str, optional
        Timing precision level: ``'line'`` (default) or ``'pixel'``.

    Returns
    -------
    frame_times : np.ndarray
        Start time of each frame in seconds, shape ``(n_frames,)``.
    frame_dt_offset : np.ndarray
        Per-pixel time offset within a frame in seconds,
        shape ``(npix_x, npix_2nd)``.
    frame_dt : float
        Duration of one frame in seconds.

    Raises
    ------
    ValueError
        If ``precision`` is not ``'line'`` or ``'pixel'``.
    NotImplementedError
        If the scan type derived from ``wparams`` is not ``'xy'`` or ``'xz'``.
    """
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
        threshold: int = 30_000, precision: str = 'line') -> Tuple[np.ndarray, np.ndarray]:
    """Extract trigger times from a stack using scan parameters from wparams.

    Parameters
    ----------
    stack : np.ndarray
        3-D array of shape ``(x, y, frames)`` representing the trigger channel.
    wparams : dict
        Dictionary of scan parameters passed to :func:`compute_frame_times`.
    stimulator_delay : float
        Delay in seconds to add to each detected trigger time.
    threshold : int, optional
        Signal threshold above which a trigger onset is detected. Default is 30 000.
    precision : str, optional
        Timing precision: ``'line'`` (default) or ``'pixel'``.

    Returns
    -------
    triggertimes : np.ndarray
        Array of trigger onset times in seconds.
    triggervalues : np.ndarray
        Signal amplitude at each trigger onset.
    """
    frame_times, frame_dt_offset, frame_dt = compute_frame_times(
        wparams=wparams, n_frames=stack.shape[2], precision=precision)
    triggertimes, triggervalues = traces_and_triggers_utils.compute_triggers(
        stack=stack, frame_times=frame_times, frame_dt_offset=frame_dt_offset,
        threshold=threshold, stimulator_delay=stimulator_delay)
    return triggertimes, triggervalues


def check_dims_ch_stack_wparams(ch_stack: np.ndarray, wparams: dict) -> None:
    """Check if the dimensions of a stack match what is expected from wparams.

    Parameters
    ----------
    ch_stack : np.ndarray
        3-D stack array whose first two axes are the spatial dimensions.
    wparams : dict
        Dictionary of scan parameters (lower-cased keys expected) containing
        ``'user_dxpix'``, ``'user_npixretrace'``, ``'user_nxpixlineoffs'``,
        ``'user_dypix'``, and optionally ``'user_dzpix'``.

    Raises
    ------
    ValueError
        If the spatial dimensions of ``ch_stack`` do not match the expected
        pixel counts derived from ``wparams``.
    """
    nxpix = wparams["user_dxpix"] - wparams["user_npixretrace"] - wparams["user_nxpixlineoffs"]
    nypix = wparams["user_dypix"]
    nzpix = wparams.get("user_dzpix", 0)

    if not (ch_stack.shape[:2] in [(nxpix, nypix), (nxpix, nzpix)]):
        raise ValueError(f'Stack shape error: {ch_stack.shape} not in [{(nxpix, nypix)}, {(nxpix, nzpix)}]')
