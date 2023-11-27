import pytest

from djimaging.utils.datafile_utils import as_pre_filepath


def test_as_pre_filepath():
    raw_filepath = '/some/path/20190515/Raw/M1_LR_chirp.smp'
    pre_filepath = '/some/path/20190515/Pre/SMP_M1_LR_chirp.h5'

    obs_filepath = as_pre_filepath(
        raw_filepath, raw_data_dir='Raw', pre_data_dir='Pre',
        raw_suffix='.smp', pre_suffix='.h5', pre_prefix='SMP_')

    assert obs_filepath == pre_filepath


def test_as_pre_filepath_no_change():
    pre_filepath = '/some/path/20190515/Pre/SMP_M1_LR_chirp.h5'

    obs_filepath = as_pre_filepath(
        pre_filepath, raw_data_dir='Raw', pre_data_dir='Pre',
        raw_suffix='.smp', pre_suffix='.h5', pre_prefix='SMP_')

    assert obs_filepath == pre_filepath


def test_as_pre_filepath_error_raw_path():
    raw_filepath = '/some/path/20190515/Pre/M1_LR_chirp.smp'

    with pytest.raises(ValueError):
        as_pre_filepath(
            raw_filepath, raw_data_dir='Raw', pre_data_dir='Pre',
            raw_suffix='.smp', pre_suffix='.h5', pre_prefix='SMP_')


def test_as_pre_filepath_error_suffix():
    raw_filepath = '/some/path/20190515/Raw/M1_LR_chirp.smp'

    with pytest.raises(ValueError):
        as_pre_filepath(
            raw_filepath, raw_data_dir='Raw', pre_data_dir='Pre',
            raw_suffix='.h5', pre_suffix='.h5', pre_prefix='SMP_')
