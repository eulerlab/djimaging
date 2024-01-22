import os.path

import pytest
import h5py
import numpy as np
from numpy.testing import assert_almost_equal

from djimaging.utils.scanm import read_h5_utils, setup_utils, traces_and_triggers_utils, wparams_utils

__RAW_DATA_PATH = "/gpfs01/euler/data/Data/DataJointTestData/"


def test_get_pixel_size_um_64_default_scan1():
    setupid = 1
    nypix = 64
    zoom = 0.87
    exp = 2.0114942528735633
    obs = setup_utils.get_pixel_size_xy_um(setupid=setupid, npix=nypix, zoom=zoom)
    assert_almost_equal(obs, exp, decimal=4)


def test_get_pixel_size_um_64_default_scan3():
    setupid = 3
    nypix = 64
    zoom = 0.65
    exp = 1.71875
    obs = setup_utils.get_pixel_size_xy_um(setupid=setupid, npix=nypix, zoom=zoom)
    assert_almost_equal(obs, exp, decimal=4)


def test_get_pixel_size_um_128_default_scan1():
    setupid = 1
    nypix = 128
    zoom = 0.87
    exp = 1.0057471264367817
    obs = setup_utils.get_pixel_size_xy_um(setupid=setupid, npix=nypix, zoom=zoom)
    assert_almost_equal(obs, exp, decimal=4)


def test_get_pixel_size_um_512_default_scan1():
    setupid = 1
    nypix = 512
    zoom = 0.87
    exp = 0.2514367816091954
    obs = setup_utils.get_pixel_size_xy_um(setupid=setupid, npix=nypix, zoom=zoom)
    assert_almost_equal(obs, exp, decimal=4)


def test_get_pixel_size_um_128_wide_scan1():
    setupid = 1
    nypix = 128
    zoom = 0.15
    exp = 5.833333333333334
    obs = setup_utils.get_pixel_size_xy_um(setupid=setupid, npix=nypix, zoom=zoom)
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

    obs_ventral_dorsal_pos, obs_temporal_nasal_pos = setup_utils.get_retinal_position(
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

    obs_ventral_dorsal_pos, obs_temporal_nasal_pos = setup_utils.get_retinal_position(
        rel_xcoord_um=rel_xcoord_um, rel_ycoord_um=rel_ycoord_um, rotation=rotation, eye=eye)

    assert exp_ventral_dorsal_pos == obs_ventral_dorsal_pos
    assert exp_temporal_nasal_pos == obs_temporal_nasal_pos


def _test_compute_tracetimes(filepath, stack_name, precision, atol):
    if not os.path.isfile(filepath):
        pytest.skip(f"File not found {filepath}")

    with h5py.File(filepath, 'r', driver="stdio") as h5_file:
        wparams = read_h5_utils.extract_wparams(h5_file)
        os_params = read_h5_utils.extract_os_params(h5_file)
        _, igor_traces_times = read_h5_utils.extract_traces(h5_file)
        stack = np.copy(h5_file[stack_name])
        roi_mask = read_h5_utils.extract_roi_mask(h5_file)

        # In Igor this was wrongly added, so remove it again
        igor_traces_times -= os_params['stimulatordelay'] / 1000

    _, traces_times = traces_and_triggers_utils.compute_traces(
        stack=stack, roi_mask=roi_mask, wparams=wparams, precision=precision)

    assert igor_traces_times.shape == traces_times.shape
    assert np.allclose(igor_traces_times, traces_times,
                       atol=atol), f"{np.max(np.abs(igor_traces_times - traces_times))} > {atol}"


def test_compute_traces(
        filepath=os.path.join(__RAW_DATA_PATH, "xy-RGCs/20220304/1/Pre/SMP_M1_LR_GCL0_chirp.h5"),
        stack_name='wDataCh0'):
    if not os.path.isfile(filepath):
        pytest.skip(f"File not found {filepath}")

    with h5py.File(filepath, 'r', driver="stdio") as h5_file:
        wparams = read_h5_utils.extract_wparams(h5_file)
        igor_traces, _ = read_h5_utils.extract_traces(h5_file)
        stack = np.copy(h5_file[stack_name])
        roi_mask = read_h5_utils.extract_roi_mask(h5_file)

    traces, _ = traces_and_triggers_utils.compute_traces(
        stack=stack, roi_mask=roi_mask, wparams=wparams)

    assert igor_traces.shape == traces.shape
    assert np.allclose(igor_traces[0, 0], traces[0, 0])
    assert np.allclose(igor_traces[0, -1], traces[0, -1])
    assert np.allclose(igor_traces[-1, 0], traces[-1, 0])
    assert np.allclose(igor_traces, traces)


def _test_compute_triggertimes(filepath, precision, atol):
    """Don't add stimulator delay for comparison, because in Igor it was added to tracetimes instead"""
    if not os.path.isfile(filepath):
        pytest.skip(f"File not found {filepath}")

    with h5py.File(filepath, 'r', driver="stdio") as h5_file:
        wparams = read_h5_utils.extract_wparams(h5_file)
        igor_triggertimes, _ = read_h5_utils.extract_triggers(h5_file, check_triggervalues=False)
        ch2_stack = np.copy(h5_file['wDataCh2'])

    # Ignore 'stimulator_delay' here, because in Igor it's added to tracetimes instead
    triggertimes, triggervalues = wparams_utils.compute_triggers_from_wparams(
        stack=ch2_stack, wparams=wparams, precision=precision, stimulator_delay=0.)

    assert igor_triggertimes.shape == triggertimes.shape
    assert np.allclose(igor_triggertimes[0], triggertimes[0], atol=atol)
    assert np.allclose(igor_triggertimes[-1], triggertimes[-1], atol=atol)
    assert np.allclose(igor_triggertimes, triggertimes, atol=atol)


def test_compute_triggertimes_line_precision_xy_dendrites(
        filepath=os.path.join(__RAW_DATA_PATH, "xy-dendrites/20201127/1/Pre/SMP_C1_d1_Chirp.h5")):
    _test_compute_triggertimes(filepath=filepath, precision='line', atol=2.5e-3)


def test_compute_triggertimes_pixel_precision_xy_dendrites(
        filepath=os.path.join(__RAW_DATA_PATH, "xy-dendrites/20201127/1/Pre/SMP_C1_d1_Chirp.h5")):
    _test_compute_triggertimes(filepath=filepath, precision='pixel', atol=3e-3)


def test_compute_triggertimes_line_precision_xy_rgcs_mb(
        filepath=os.path.join(__RAW_DATA_PATH, "xy-RGCs/20220215/1/Pre/SMP_M1_LR_GCL1_MB_D.h5")):
    _test_compute_triggertimes(filepath=filepath, precision='line', atol=2.5e-3)


def test_compute_triggertimes_pixel_precision_xy_rgcs_mb(
        filepath=os.path.join(__RAW_DATA_PATH, "xy-RGCs/20220215/1/Pre/SMP_M1_LR_GCL1_MB_D.h5")):
    _test_compute_triggertimes(filepath=filepath, precision='pixel', atol=3e-3)


def test_compute_triggertimes_line_precision_xy_rgcs_noise(
        filepath=os.path.join(__RAW_DATA_PATH, "xy-RGCs/20220215/1/Pre/SMP_M1_LR_GCL1_MB_D.h5")):
    _test_compute_triggertimes(filepath=filepath, precision='line', atol=2.5e-3)


def test_compute_triggertimes_pixel_precision_xy_rgcs_noise(
        filepath=os.path.join(__RAW_DATA_PATH, "xy-RGCs/20220215/1/Pre/SMP_M1_LR_GCL3_DN_C1.h5")):
    _test_compute_triggertimes(filepath=filepath, precision='pixel', atol=3e-3)


def test_compute_triggertimes_line_precision_xz_bcs_chirp(
        filepath=os.path.join(__RAW_DATA_PATH, "xz-BCs/20220921/2/Pre/SMP_M1_RR_xz0_chirp_C1.h5")):
    _test_compute_triggertimes(filepath=filepath, precision='line', atol=2.5e-3)


def test_compute_triggertimes_line_precision_xz_bcs_noise(
        filepath=os.path.join(__RAW_DATA_PATH, "xz-BCs/20220921/2/Pre/SMP_M1_RR_xz3_BCnoise_C1.h5")):
    _test_compute_triggertimes(filepath=filepath, precision='line', atol=2.5e-3)


def test_compute_triggertimes_pixel_precision_xz_bcs_chirp(
        filepath=os.path.join(__RAW_DATA_PATH, "xz-BCs/20220921/2/Pre/SMP_M1_RR_xz0_chirp_C1.h5")):
    _test_compute_triggertimes(filepath=filepath, precision='pixel', atol=2e-3)


def test_compute_triggertimes_pixel_precision_xz_bcs_noise(
        filepath=os.path.join(__RAW_DATA_PATH, "xz-BCs/20220921/2/Pre/SMP_M1_RR_xz1_BCnoise_C1.h5")):
    _test_compute_triggertimes(filepath=filepath, precision='pixel', atol=2e-3)


def test_compute_tracetimes_line_precision(
        filepath=os.path.join(__RAW_DATA_PATH, "xy-RGCs/20220304/1/Pre/SMP_M1_LR_GCL0_chirp.h5"),
        stack_name='wDataCh0'):
    _test_compute_tracetimes(filepath=filepath, stack_name=stack_name, precision='line', atol=1e-3)


def test_compute_tracetimes_pixel_precision(
        filepath=os.path.join(__RAW_DATA_PATH, "xy-RGCs/20220304/1/Pre/SMP_M1_LR_GCL0_chirp.h5"),
        stack_name='wDataCh0'):
    _test_compute_tracetimes(filepath=filepath, stack_name=stack_name, precision='pixel', atol=3e-3)


def test_compute_tracetimes_line_precision_xy_rgcs(
        filepath=os.path.join(__RAW_DATA_PATH, "xy-RGCs/20220304/1/Pre/SMP_M1_LR_GCL0_chirp.h5"),
        stack_name='wDataCh0'):
    _test_compute_tracetimes(filepath=filepath, stack_name=stack_name, precision='line', atol=2.5e-3)


def test_compute_tracetimes_pixel_precision_xy_rgcs(
        filepath=os.path.join(__RAW_DATA_PATH, "xy-RGCs/20220304/1/Pre/SMP_M1_LR_GCL0_chirp.h5"),
        stack_name='wDataCh0'):
    _test_compute_tracetimes(filepath=filepath, stack_name=stack_name, precision='pixel', atol=3e-3)


def test_compute_tracetimes_line_precision_xz_bcs(
        filepath=os.path.join(__RAW_DATA_PATH, "xz-BCs/20220921/2/Pre/SMP_M1_RR_xz1_BCnoise_C1.h5"),
        stack_name='wDataCh0'):
    _test_compute_tracetimes(filepath=filepath, stack_name=stack_name, precision='line', atol=2.5e-3)


def test_compute_tracetimes_pixel_precision_xz_bcs(
        filepath=os.path.join(__RAW_DATA_PATH, "xz-BCs/20220921/2/Pre/SMP_M1_RR_xz1_BCnoise_C1.h5"),
        stack_name='wDataCh0'):
    _test_compute_tracetimes(filepath=filepath, stack_name=stack_name, precision='pixel', atol=3e-3)
