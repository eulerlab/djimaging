"""Tests of the ScanM .h5 readers with the synthetic recordings and small hand-written .h5 files."""
from pathlib import Path
from typing import Callable

import h5py
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from djimaging.utils.scanm import read_h5_utils, read_utils
from tests.fixtures.synthetic_scanm import (
    EXPERIMENT_DAY, EXPERIMENT_NUMBER, FRAME_DT, NX, NY, PIXEL_DURATION_US, STIMULATOR_DELAY_MS)


def write_igor_table(h5_file: h5py.File, name: str, params: dict, as_bytes: bool = False) -> None:
    """Igor wave with dimension labels, as wParamsNum / OS_Parameters."""
    dataset = h5_file.create_dataset(name, data=np.asarray(list(params.values()), dtype=np.float32))
    if as_bytes:
        labels = np.full((len(params) + 1, 1), b'', dtype=h5py.string_dtype(encoding='ascii'))
        labels[1:, 0] = [k.encode() for k in params]
    else:
        labels = np.full((len(params) + 1, 1), '', dtype=h5py.string_dtype(encoding='utf-8'))
        labels[1:, 0] = list(params)
    dataset.attrs['IGORWaveDimensionLabels'] = labels


def wparams_num(nx=NX, ny=NY, nz=0, offs=0, retrace=0, **kwargs) -> dict:
    params = dict(User_dxPix=nx, User_dyPix=ny, User_dzPix=nz, User_nPixRetrace=retrace, User_nXPixLineOffs=offs,
                  RealPixDur=PIXEL_DURATION_US, Zoom=1, Angle_deg=0, XCoord_um=1, YCoord_um=2, ZCoord_um=3,
                  ZStep_um=1, User_ScanType=10)
    params.update(kwargs)
    return params


def write_h5(path, stacks: dict, params: dict | None = None, **datasets) -> str:
    with h5py.File(path, 'w') as f:
        for name, stack in stacks.items():
            f.create_dataset(name, data=stack)
        if params is not None:
            write_igor_table(f, 'wParamsNum', params)
        for name, data in datasets.items():
            f.create_dataset(name, data=data)
    return str(path)


@pytest.fixture
def chirp_path(synthetic_recording_path: Callable[[str], Path]) -> str:
    return str(synthetic_recording_path('chirp'))


# Loading stacks

def test_load_stacks_and_wparams(chirp_path):
    ch_stacks, wparams = read_h5_utils.load_stacks_and_wparams(chirp_path, ch_names=('wDataCh0', 'wDataCh2'))
    assert sorted(ch_stacks) == ['wDataCh0', 'wDataCh2']
    assert ch_stacks['wDataCh0'].shape == (NX, NY, 900)
    assert wparams['user_dxpix'] == NX
    assert wparams['realpixdur'] == pytest.approx(PIXEL_DURATION_US)

    ch_stacks_ru, wparams_ru = read_utils.load_stacks(chirp_path, from_raw_data=False, ch_names=('wDataCh0',))
    assert_array_equal(ch_stacks_ru['wDataCh0'], ch_stacks['wDataCh0'])
    assert wparams_ru == wparams


def test_load_stacks_and_wparams_dim_mismatch(tmp_path):
    path = write_h5(tmp_path / 'a.h5', {'wDataCh0': np.zeros((4, 3, 2))}, wparams_num(nx=4, ny=4))
    with pytest.raises(ValueError, match='Stack shape'):
        read_h5_utils.load_stacks_and_wparams(path, ch_names=('wDataCh0',))


def test_load_stacks_and_wparams_with_line_offset_and_retrace(tmp_path):
    path = write_h5(tmp_path / 'a.h5', {'wDataCh0': np.zeros((4, 3, 2))}, wparams_num(nx=7, ny=3, offs=1, retrace=2))
    ch_stacks, _ = read_h5_utils.load_stacks_and_wparams(path, ch_names=('wDataCh0',))
    assert ch_stacks['wDataCh0'].shape == (4, 3, 2)


def test_load_missing_file(tmp_path):
    for loader in [read_h5_utils.load_traces, read_h5_utils.load_triggers, read_h5_utils.load_roi_mask,
                   read_h5_utils.load_stacks_and_wparams]:
        with pytest.raises(OSError, match='Error loading file'):
            loader(str(tmp_path / 'missing.h5'))


def test_extract_stacks_errors(tmp_path):
    path = write_h5(tmp_path / 'a.h5', {'wDataCh0': np.zeros((4, 3, 2)), 'wDataCh1': np.zeros((4, 3, 3)),
                                        'wDataCh2': np.zeros((4, 3))})
    with h5py.File(path, 'r') as f:
        with pytest.raises(ValueError, match='equal size'):
            read_h5_utils.extract_stacks(f, ch_names=('wDataCh0', 'wDataCh1'))
        with pytest.raises(ValueError, match='3d'):
            read_h5_utils.extract_stacks(f, ch_names=('wDataCh2',))
        with pytest.raises(KeyError):
            read_h5_utils.extract_stacks(f, ch_names=('wDataCh3',))


def test_extract_all_stack_from_h5(chirp_path):
    with h5py.File(chirp_path, 'r') as f:
        ch_stacks = read_h5_utils.extract_all_stack_from_h5(f)
    assert sorted(ch_stacks) == ['wDataCh0', 'wDataCh1', 'wDataCh2']


# Traces, triggers, ROI masks

def test_load_traces_and_triggers(chirp_path):
    traces, traces_times = read_h5_utils.load_traces(chirp_path)
    assert traces.shape == traces_times.shape == (900, 2)
    assert_allclose(np.diff(traces_times[:, 0]), FRAME_DT)

    triggertimes, triggervalues = read_h5_utils.load_triggers(chirp_path)
    assert triggertimes.size == triggervalues.size == 10
    assert triggertimes[0] == pytest.approx(7.0, abs=FRAME_DT)


def test_extract_traces_missing(tmp_path):
    path = write_h5(tmp_path / 'a.h5', {}, Traces0_raw=np.zeros((3, 2)))
    with pytest.raises(ValueError, match='Traces not found'):
        read_h5_utils.load_traces(path)


def test_load_roi_mask(chirp_path, tmp_path):
    roi_mask = read_h5_utils.load_roi_mask(chirp_path)
    assert roi_mask.dtype.kind == 'i'
    assert roi_mask.shape == (NX, NY)
    assert set(np.unique(roi_mask)) == {-2, -1, 1}

    no_mask = write_h5(tmp_path / 'none.h5', {})
    assert read_h5_utils.load_roi_mask(no_mask, ignore_not_found=True) is None
    with pytest.raises(KeyError, match='No ROI mask'):
        read_h5_utils.load_roi_mask(no_mask)

    empty_mask = write_h5(tmp_path / 'empty.h5', {}, ROIs=np.zeros((0, 0)))
    assert read_h5_utils.load_roi_mask(empty_mask, ignore_not_found=True) is None

    float_mask = write_h5(tmp_path / 'float.h5', {}, ROIs=np.array([[1., -1.], [-2., 1.]]))
    assert read_h5_utils.load_roi_mask(float_mask).dtype.kind == 'i'

    lower_case = write_h5(tmp_path / 'lower.h5', {}, rois=np.array([[1, -1]]))
    assert_array_equal(read_h5_utils.load_roi_mask(lower_case), [[1, -1]])

    two_masks = write_h5(tmp_path / 'two.h5', {}, ROIs=np.ones((2, 2)), ROIs_old=np.ones((2, 2)))
    with pytest.raises(KeyError, match='Multiple'):
        read_h5_utils.load_roi_mask(two_masks)


def test_extract_triggers_variants(tmp_path):
    no_triggers = write_h5(tmp_path / 'none.h5', {})
    with h5py.File(no_triggers, 'r') as f:
        triggertimes, triggervalues = read_h5_utils.extract_triggers(f)
        assert triggertimes.size == 0 and triggervalues.size == 0
        with pytest.raises(KeyError):
            read_h5_utils.extract_triggers(f, ignore_not_found=False)

    lower = write_h5(tmp_path / 'lower.h5', {}, triggertimes=np.array([1., 2.]), triggervalues=np.array([5.]))
    with h5py.File(lower, 'r') as f:
        triggertimes, triggervalues = read_h5_utils.extract_triggers(f)
        assert_array_equal(triggertimes, [1., 2.])
        with pytest.raises(AssertionError, match='Mismatch'):
            read_h5_utils.extract_triggers(f, check_triggervalues=True)

    duplicate = write_h5(tmp_path / 'dup.h5', {}, triggertimes=np.array([1.]), Triggertimes=np.array([1.]))
    with h5py.File(duplicate, 'r') as f:
        with pytest.raises(ValueError, match='Multiple triggertimes'):
            read_h5_utils.extract_triggers(f)

    duplicate_values = write_h5(tmp_path / 'dupv.h5', {}, Triggervalues=np.array([1.]), triggervalues=np.array([1.]))
    with h5py.File(duplicate_values, 'r') as f:
        with pytest.raises(ValueError, match='Multiple triggervalues'):
            read_h5_utils.extract_triggers(f)


def test_extract_triggers_converts_old_frame_based_triggers(tmp_path):
    """Old files stored triggers in samples of 2 ms (500 Hz)."""
    in_frames = np.array([3500., 5500., 7500.])
    path = write_h5(tmp_path / 'old.h5', {}, Triggertimes=in_frames)
    triggertimes, _ = read_h5_utils.load_triggers(path)
    assert_allclose(triggertimes, in_frames / 500.)

    with_tracetimes = write_h5(tmp_path / 'old_tt.h5', {}, Triggertimes=in_frames,
                               Tracetimes0=np.linspace(0, 20, 100)[:, None])
    triggertimes, _ = read_h5_utils.load_triggers(with_tracetimes)
    assert_allclose(triggertimes, in_frames / 500.)

    # In seconds already: not changed
    in_seconds = write_h5(tmp_path / 'new_tt.h5', {}, Triggertimes=np.array([7., 11.]),
                          Tracetimes0=np.linspace(0, 20, 100)[:, None])
    triggertimes, _ = read_h5_utils.load_triggers(in_seconds)
    assert_allclose(triggertimes, [7., 11.])


@pytest.mark.xfail(strict=True, reason="BUG: load_roi2trace compares positive roi_ids with the negative ids of "
                                       "extract_roi_ids (ValueError) and passes negative ids to get_roi2trace "
                                       "(AssertionError): Traces with compute_from_stack=False always fails")
def test_load_roi2trace(chirp_path):
    roi2trace, frame_dt = read_h5_utils.load_roi2trace(chirp_path, roi_ids=np.array([1, 2]))
    assert sorted(roi2trace) == [1, 2]
    assert frame_dt == pytest.approx(FRAME_DT)


def test_load_roi2trace_rejects_unknown_rois(chirp_path):
    with pytest.raises(ValueError, match='do not match'):
        read_h5_utils.load_roi2trace(chirp_path, roi_ids=np.array([3]))


# Parameter tables

def test_extract_wparams_and_os_params(chirp_path):
    with h5py.File(chirp_path, 'r') as f:
        wparams = read_h5_utils.extract_wparams(f, lower_keys=False)
        os_params = read_h5_utils.extract_os_params(f)
    assert wparams['User_dxPix'] == NX
    assert 'user_dxpix' not in wparams
    assert os_params == {'stimulatordelay': pytest.approx(STIMULATOR_DELAY_MS)}


def test_extract_wparams_with_str_table_and_bytes_labels(tmp_path):
    path = tmp_path / 'a.h5'
    with h5py.File(path, 'w') as f:
        write_igor_table(f, 'wparamsnum', {'Zoom': 2.}, as_bytes=True)
        dataset = f.create_dataset('wParamsStr', data=np.array([b'PC1', b'', b'me']))
        labels = np.full((4, 1), '', dtype=h5py.string_dtype(encoding='utf-8'))
        labels[1:, 0] = ['ComputerName', '', 'UserName']  # empty labels are skipped
        dataset.attrs['IGORWaveDimensionLabels'] = labels
    with h5py.File(path, 'r') as f:
        wparams = read_h5_utils.extract_wparams(f)
        assert read_h5_utils.extract_os_params(f) == {}
    assert wparams == {'zoom': 2., 'computername': b'PC1', 'username': b'me'}
    assert read_h5_utils.load_h5_table('wparamsnum', 'wParamsStr', filename=str(path)) == {
        'Zoom': 2., 'ComputerName': b'PC1', 'UserName': b'me'}


def test_load_h5_data(tmp_path):
    path = write_h5(tmp_path / 'a.h5', {'wDataCh0': np.ones((2, 2, 2))}, Traces0_raw=np.zeros((3, 1)))
    data = read_h5_utils.load_h5_data(path)
    assert sorted(data) == ['Traces0_raw', 'wDataCh0']
    assert sorted(read_h5_utils.load_h5_data(path, lower_keys=True)) == ['traces0_raw', 'wdatach0']


def test_read_config_dict(tutorial_data_dir: Path):
    ini = tutorial_data_dir / EXPERIMENT_DAY / EXPERIMENT_NUMBER / 'synthetic__left.ini'
    config = read_h5_utils.read_config_dict(str(ini))
    assert config['projname'] == 'Synthetic tutorial data'
    assert config['setupid'] == '3'
    assert config['prepwmopticdiscpos'] == '111;222;3'
    assert config['pharmdrugconc_um'] == ''  # only the type prefix is removed
    assert config['prepwmorient'] == '0'


def test_read_config_dict_key_without_prefix(tmp_path):
    ini = tmp_path / 'h.ini'
    ini.write_text('[A]\nnoprefix=1\nstring_x_y=2\n', encoding='utf-8')
    assert read_h5_utils.read_config_dict(str(ini)) == {'noprefix': '1', 'x_y': '2'}
