import os.path

import pytest
import h5py
import numpy as np
from numpy.testing import assert_almost_equal

from djimaging.utils import scanm_utils


def test_get_pixel_size_um_64_default_scan1():
    setupid = 1
    nypix = 64
    zoom = 0.87
    exp = 2.0114942528735633
    obs = scanm_utils.get_pixel_size_xy_um(setupid=setupid, npix=nypix, zoom=zoom)
    assert_almost_equal(obs, exp, decimal=4)


def test_get_pixel_size_um_64_default_scan3():
    setupid = 3
    nypix = 64
    zoom = 0.65
    exp = 1.71875
    obs = scanm_utils.get_pixel_size_xy_um(setupid=setupid, npix=nypix, zoom=zoom)
    assert_almost_equal(obs, exp, decimal=4)


def test_get_pixel_size_um_128_default_scan1():
    setupid = 1
    nypix = 128
    zoom = 0.87
    exp = 1.0057471264367817
    obs = scanm_utils.get_pixel_size_xy_um(setupid=setupid, npix=nypix, zoom=zoom)
    assert_almost_equal(obs, exp, decimal=4)


def test_get_pixel_size_um_512_default_scan1():
    setupid = 1
    nypix = 512
    zoom = 0.87
    exp = 0.2514367816091954
    obs = scanm_utils.get_pixel_size_xy_um(setupid=setupid, npix=nypix, zoom=zoom)
    assert_almost_equal(obs, exp, decimal=4)


def test_get_pixel_size_um_128_wide_scan1():
    setupid = 1
    nypix = 128
    zoom = 0.15
    exp = 5.833333333333334
    obs = scanm_utils.get_pixel_size_xy_um(setupid=setupid, npix=nypix, zoom=zoom)
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

    obs_ventral_dorsal_pos, obs_temporal_nasal_pos = scanm_utils.get_retinal_position(
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

    obs_ventral_dorsal_pos, obs_temporal_nasal_pos = scanm_utils.get_retinal_position(
        rel_xcoord_um=rel_xcoord_um, rel_ycoord_um=rel_ycoord_um, rotation=rotation, eye=eye)

    assert exp_ventral_dorsal_pos == obs_ventral_dorsal_pos
    assert exp_temporal_nasal_pos == obs_temporal_nasal_pos


def test_compute_tracetimes(
        filepath="/gpfs01/euler/data/Data/DataJointTestData/20220304/1/Pre/SMP_M1_LR_GCL0_chirp.h5",
        stack_name='wDataCh0'):
    if not os.path.isfile(filepath):
        pytest.skip(f"File not found {filepath}")

    with h5py.File(filepath, 'r', driver="stdio") as h5_file:
        w_params = scanm_utils.extract_w_params_from_h5(h5_file)
        os_params = scanm_utils.extract_os_params(h5_file)
        _, igor_traces_times = scanm_utils.extract_traces(h5_file)
        stack = np.copy(h5_file[stack_name])
        roi_mask = scanm_utils.extract_roi_mask(h5_file)

    _, traces_times = scanm_utils.compute_traces(
        stack=stack, roi_mask=roi_mask, w_params=w_params, os_params=os_params)

    assert igor_traces_times.shape == traces_times.shape
    assert np.allclose(igor_traces_times, traces_times, atol=4e-3)


def test_compute_traces(
        filepath="/gpfs01/euler/data/Data/DataJointTestData/20220304/1/Pre/SMP_M1_LR_GCL0_chirp.h5",
        stack_name='wDataCh0'):
    if not os.path.isfile(filepath):
        pytest.skip(f"File not found {filepath}")

    with h5py.File(filepath, 'r', driver="stdio") as h5_file:
        w_params = scanm_utils.extract_w_params_from_h5(h5_file)
        os_params = scanm_utils.extract_os_params(h5_file)
        igor_traces, _ = scanm_utils.extract_traces(h5_file)
        stack = np.copy(h5_file[stack_name])
        roi_mask = scanm_utils.extract_roi_mask(h5_file)

    traces, _ = scanm_utils.compute_traces(
        stack=stack, roi_mask=roi_mask, w_params=w_params, os_params=os_params)

    assert igor_traces.shape == traces.shape
    assert np.allclose(igor_traces, traces)


def test_compute_triggertimes(filepath="/gpfs01/euler/data/Data/DataJointTestData/20201127/1/Pre/SMP_C1_d1_Chirp.h5"):
    if not os.path.isfile(filepath):
        pytest.skip(f"File not found {filepath}")

    with h5py.File(filepath, 'r', driver="stdio") as h5_file:
        w_params = scanm_utils.extract_w_params_from_h5(h5_file)
        os_params = scanm_utils.extract_os_params(h5_file)
        igor_triggertimes, _ = scanm_utils.extract_triggers(h5_file, check_triggervalues=False)
        ch2_stack = np.copy(h5_file['wDataCh2'])

    triggertimes = scanm_utils.compute_triggertimes(
        stack=ch2_stack, w_params=w_params, os_params=os_params)

    assert igor_triggertimes.shape == triggertimes.shape
    assert np.allclose(igor_triggertimes, triggertimes, atol=2e-3)


if __name__ == "__main__":
    test_compute_tracetimes()
    test_compute_triggertimes()
