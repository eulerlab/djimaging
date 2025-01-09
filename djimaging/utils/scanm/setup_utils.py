import os

import numpy as np
import pandas as pd


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
