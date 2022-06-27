from numpy.testing import assert_almost_equal
from djimaging.utils.scanm_utils import get_pixel_size_xy_um, get_retinal_position


def test_get_pixel_size_um_64_default_scan1():
    setupid = 1
    nypix = 64
    zoom = 0.87
    exp = 2.0114942528735633
    obs = get_pixel_size_xy_um(setupid=setupid, npix=nypix, zoom=zoom)
    assert_almost_equal(obs, exp, decimal=4)


def test_get_pixel_size_um_64_default_scan3():
    setupid = 3
    nypix = 64
    zoom = 0.65
    exp = 1.71875
    obs = get_pixel_size_xy_um(setupid=setupid, npix=nypix, zoom=zoom)
    assert_almost_equal(obs, exp, decimal=4)


def test_get_pixel_size_um_128_default_scan1():
    setupid = 1
    nypix = 128
    zoom = 0.87
    exp = 1.0057471264367817
    obs = get_pixel_size_xy_um(setupid=setupid, npix=nypix, zoom=zoom)
    assert_almost_equal(obs, exp, decimal=4)


def test_get_pixel_size_um_512_default_scan1():
    setupid = 1
    nypix = 512
    zoom = 0.87
    exp = 0.2514367816091954
    obs = get_pixel_size_xy_um(setupid=setupid, npix=nypix, zoom=zoom)
    assert_almost_equal(obs, exp, decimal=4)


def test_get_pixel_size_um_128_wide_scan1():
    setupid = 1
    nypix = 128
    zoom = 0.15
    exp = 5.833333333333334
    obs = get_pixel_size_xy_um(setupid=setupid, npix=nypix, zoom=zoom)
    assert_almost_equal(obs, exp, decimal=4)


def test_get_retinal_position_right_dn():
    """The x stored in ScanM is dorsoventral axis (spatial y) and the y is nasotemporal axis (spatial x).
    The spatial y has to be multiplied with -1, to make dorsal positive.
    The spatial x: in RR x>0 means more nasal and in LR x<0 means more nasal,
    so the correct x in LR has to be multiplied with -1."""
    rel_xcoord_um = -1000.
    rel_ycoord_um = 100.
    rotation = 0.0
    eye = 'right'

    exp_ventral_dorsal_pos = 1000.
    exp_temporal_nasal_pos = 100.

    obs_ventral_dorsal_pos, obs_temporal_nasal_pos = get_retinal_position(
        rel_xcoord_um=rel_xcoord_um, rel_ycoord_um=rel_ycoord_um, rotation=rotation, eye=eye)

    assert exp_ventral_dorsal_pos == obs_ventral_dorsal_pos
    assert exp_temporal_nasal_pos == obs_temporal_nasal_pos


def test_get_retinal_position_left_vn():
    rel_xcoord_um = 1000.
    rel_ycoord_um = -100.
    rotation = 0.0
    eye = 'left'

    exp_ventral_dorsal_pos = -1000.
    exp_temporal_nasal_pos = 100.

    obs_ventral_dorsal_pos, obs_temporal_nasal_pos = get_retinal_position(
        rel_xcoord_um=rel_xcoord_um, rel_ycoord_um=rel_ycoord_um, rotation=rotation, eye=eye)

    assert exp_ventral_dorsal_pos == obs_ventral_dorsal_pos
    assert exp_temporal_nasal_pos == obs_temporal_nasal_pos
