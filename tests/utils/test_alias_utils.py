"""Tests of the filename alias matching (djimaging.utils.alias_utils) and of misc_utils.NoPrints."""
import sys

import pytest

from djimaging.utils import alias_utils as au
from djimaging.utils.misc_utils import NoPrints


@pytest.mark.parametrize('str1, str2, kwargs, expected', [
    ('hr_highres', 'M1_LR_HR', {}, True),
    ('hr_highres', 'M1_LR_HR', dict(case_sensitive=True), False),
    ('hr_highres', 'M1_LR_hr2', {}, False),  # tokens must be equal
    ('a__b', 'x__y', {}, False),  # empty tokens never match
    ('_', '_', {}, False),
    ('hr-highres', 'm1-hr', dict(sep='-'), True),
])
def test_check_shared_alias_str(str1, str2, kwargs, expected):
    assert au.check_shared_alias_str(str1, str2, **kwargs) is expected


def test_check_shared_alias_list():
    assert au.check_shared_alias_list(['a', ''], ['', 'a'])
    assert not au.check_shared_alias_list(['', 'b'], ['', 'c'])
    assert not au.check_shared_alias_list([], [])


@pytest.mark.parametrize('file, pattern, kwargs, expected', [
    ('SMP_M1_GCL0_chirp.h5', 'chirp', {}, True),
    ('SMP_M1_GCL0_chirp.h5', 'CHIRP', {}, True),
    ('SMP_M1_GCL0_chirp.h5', 'CHIRP', dict(case_sensitive=True), False),
    ('/some/dir_with_chirp/SMP_M1_GCL0_mb.h5', 'chirp', {}, False),  # directories are ignored
    ('SMP_M1_GCL0_chirp.h5', 'mb_chirp', {}, True),  # any of the pattern tokens
    ('SMP_M1_GCL0_chirp.h5', 'chirp', dict(pattern_loc=3), True),
    ('SMP_M1_GCL0_chirp.h5', 'chirp', dict(pattern_loc=2), False),
    ('SMP_M1_GCL0_chirp.h5', 'chirp', dict(pattern_loc=4), False),  # beyond the last token
    ('SMP_M1_GCL0_chirp.h5', 'chirp', dict(pattern_loc=-1), True),
    ('SMP_M1_GCL0_chirp2.h5', 'chirp', {}, False),
    ('SMP_M1_GCL0_chirp2.h5', 'chirp', dict(mode='contains'), True),
    ('SMP_M1_GCL0_chirp.h5', 'chirp', dict(mode='unknown'), False),
    ('SMP_M1_GCL0_chirp.smp.h5', 'chirp', dict(ftype='smp.h5'), True),
    ('M1_GCL0_chirp', 'chirp', {}, True),  # no extension
    ('M1-GCL0-chirp.h5', 'gcl0', dict(sep='-'), True),
])
def test_match_file(file, pattern, kwargs, expected):
    assert au.match_file(file, pattern, **kwargs) is expected


def test_match_file_case_insensitive_ftype():
    assert au.match_file('SMP_M1_GCL0_chirp.H5', 'chirp', ftype='H5')


def test_match_files():
    files = ['SMP_M1_GCL0_chirp.h5', 'SMP_M1_GCL1_chirp.h5', 'SMP_M1_GCL0_MB.h5']
    assert au.match_files(files, 'gcl0') == ['SMP_M1_GCL0_chirp.h5', 'SMP_M1_GCL0_MB.h5']
    assert au.match_files(files, 'GCL', pattern_loc=2, mode='contains') == files
    assert au.match_files([], 'x') == []


def test_get_field_files(tmp_path):
    for name in ['SMP_M1_GCL1_chirp.h5', 'SMP_M1_GCL0_MB.h5', 'SMP_M1_GCL0_chirp.h5', '.SMP_M1_GCL0_dn.h5',
                 'SMP_M1_GCL0_DN.smp', 'SMP_M1_GCL01_chirp.h5']:
        (tmp_path / name).write_text('')
    assert au.get_field_files(str(tmp_path), 'gcl0') == ['SMP_M1_GCL0_MB.h5', 'SMP_M1_GCL0_chirp.h5']  # sorted
    assert au.get_field_files(str(tmp_path), 'GCL0', incl_hidden=True) == [
        '.SMP_M1_GCL0_dn.h5', 'SMP_M1_GCL0_MB.h5', 'SMP_M1_GCL0_chirp.h5']
    assert au.get_field_files(str(tmp_path), 'GCL0', ftype='smp') == ['SMP_M1_GCL0_DN.smp']
    assert au.get_field_files(str(tmp_path), 'GCL0', field_loc=1) == []
    assert au.get_field_files(str(tmp_path), 'GCL0', field_loc=2) == ['SMP_M1_GCL0_MB.h5', 'SMP_M1_GCL0_chirp.h5']


def test_no_prints(capsys):
    stdout = sys.stdout
    with NoPrints():
        print('hidden')
    print('visible')
    assert capsys.readouterr().out == 'visible\n'
    assert sys.stdout is stdout


def test_no_prints_restores_stdout_on_error():
    stdout = sys.stdout
    with pytest.raises(RuntimeError):
        with NoPrints():
            raise RuntimeError('x')
    assert sys.stdout is stdout
