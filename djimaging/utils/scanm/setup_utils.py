import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def get_stimulator_delay(date: str, setupid: int, file: Optional[str] = None) -> float:
    """Get delay of stimulator which is the time that passes between the electrical trigger recorded in
    the third channel until the stimulus is actually displayed.
    For the light crafter these are 20-100 ms, for the Arduino it is zero.

    Parameters
    ----------
    date : str
        Date string in a format parseable by ``pandas.Timestamp``.
    setupid : int
        Integer identifier of the experimental setup.
    file : str, optional
        Path to the CSV file containing stimulator delay values.
        If None, uses the default file bundled with this module.

    Returns
    -------
    float
        Stimulator delay in seconds.
    """
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


def get_setup_xscale(setupid: int) -> float:
    """Get pixel scale in x and y for setup.

    Parameters
    ----------
    setupid : int
        Integer identifier of the experimental setup (1, 2, or 3).

    Returns
    -------
    float
        Full-field scan width in micrometres at zoom 1.
    """
    # TODO: these values depend on the date. add date param and table, and read from table
    setupid = int(setupid)
    assert setupid in [1, 2, 3], setupid

    if setupid == 1:
        setup_xscale = 112.
    else:
        setup_xscale = 71.5

    return setup_xscale


def get_pixel_size_xy_um(setupid: int, npix: int, zoom: float) -> float:
    """Get width / height of a pixel in micrometres.

    Parameters
    ----------
    setupid : int
        Integer identifier of the experimental setup.
    npix : int
        Number of pixels along one spatial axis.
    zoom : float
        Objective zoom factor (must be in [0.15, 4]).

    Returns
    -------
    float
        Size of a single pixel in micrometres.
    """
    assert 0.15 <= zoom <= 4, zoom
    assert 1 <= npix < 5000, npix

    standard_pixel_size = get_setup_xscale(setupid) / npix
    pixel_size = standard_pixel_size / zoom

    return pixel_size


def get_retinal_position(rel_xcoord_um: float, rel_ycoord_um: float, rotation: float, eye: str) -> Tuple[float, float]:
    """Get retinal position based on XCoord_um and YCoord_um relative to optic disk.

    Parameters
    ----------
    rel_xcoord_um : float
        X coordinate relative to the optic disk in micrometres.
    rel_ycoord_um : float
        Y coordinate relative to the optic disk in micrometres.
    rotation : float
        Rotation angle in degrees used to align the retinal axes.
    eye : str
        Eye identifier, either ``'left'`` or ``'right'``.

    Returns
    -------
    ventral_dorsal_pos_um : float
        Position along the ventral-dorsal axis in micrometres.
    temporal_nasal_pos_um : float
        Position along the temporal-nasal axis in micrometres.
        Returns ``np.nan`` if ``eye`` is not ``'left'`` or ``'right'``.
    """
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


def get_npixartifact(setupid: int) -> int:
    """Get number of lines affected by the light artifact.

    Parameters
    ----------
    setupid : int
        Integer identifier of the experimental setup (1, 2, or 3).

    Returns
    -------
    int
        Number of pixel lines at the top of the frame affected by
        the light artifact.
    """
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
