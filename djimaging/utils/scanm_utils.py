import numpy as np


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
