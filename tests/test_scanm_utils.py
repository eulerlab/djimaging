from numpy.testing import assert_almost_equal
from djimaging.utils.scanm_utils import get_pixel_size_um


def test_get_pixel_size_um_64_default_scan1():
    setupid = 1
    nypix = 64
    zoom = 0.87
    exp = 2.0114942528735633
    obs = get_pixel_size_um(setupid=setupid, nypix=nypix, zoom=zoom)
    assert_almost_equal(obs, exp, decimal=4)


def test_get_pixel_size_um_64_default_scan3():
    setupid = 3
    nypix = 64
    zoom = 0.65
    exp = 1.71875
    obs = get_pixel_size_um(setupid=setupid, nypix=nypix, zoom=zoom)
    assert_almost_equal(obs, exp, decimal=4)


def test_get_pixel_size_um_128_default_scan1():
    setupid = 1
    nypix = 128
    zoom = 0.87
    exp = 1.0057471264367817
    obs = get_pixel_size_um(setupid=setupid, nypix=nypix, zoom=zoom)
    assert_almost_equal(obs, exp, decimal=4)


def test_get_pixel_size_um_512_default_scan1():
    setupid = 1
    nypix = 512
    zoom = 0.87
    exp = 0.2514367816091954
    obs = get_pixel_size_um(setupid=setupid, nypix=nypix, zoom=zoom)
    assert_almost_equal(obs, exp, decimal=4)


def test_get_pixel_size_um_128_wide_scan1():
    setupid = 1
    nypix = 128
    zoom = 0.15
    exp = 5.833333333333334
    obs = get_pixel_size_um(setupid=setupid, nypix=nypix, zoom=zoom)
    assert_almost_equal(obs, exp, decimal=4)


