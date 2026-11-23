"""Tests of filename parsing and file discovery in djimaging.utils.filesystem_utils (with files in tmp_path)."""
import os

import pytest

from djimaging.utils import filesystem_utils as fu

# Positions of the tokens in Pre file names, e.g. SMP_M1_LR_GCL0_chirp_C1.h5 (as in UserInfo)
USER_DICT = dict(
    opticdisk_alias='od_opticdisk_optic',
    highres_alias='hr_highres',
    outline_alias='outline_edge',
    mask_alias='chirp_mb_dn',
    animal_loc=1, region_loc=2, field_loc=3, stimulus_loc=4, cond1_loc=5, cond2_loc=None,
)
ALIASES = dict(od_aliases=['od', 'opticdisk'], hr_aliases=['hr', 'highres'], ol_aliases=['outline', 'edge'],
               rm_aliases=['chirp', 'mb', 'dn'])


def _row(filename, loc_mapper=None):
    loc_mapper = dict(animal=1, region=2, field=3, stimulus=4, cond1=5) if loc_mapper is None else loc_mapper
    return fu.get_file_info_row('folder', filename, loc_mapper, **ALIASES)


def test_get_file_info_row_response():
    row = _row('SMP_M1_LR_GCL0_chirp.h5')
    assert (row['animal'], row['region'], row['field'], row['stimulus']) == ('M1', 'LR', 'GCL0', 'chirp')
    assert row['cond1'] is None
    assert row['kind'] == 'response'
    assert row['mask_order'] == 0.
    assert row['filepath'] == os.path.join('folder', 'SMP_M1_LR_GCL0_chirp.h5')


@pytest.mark.parametrize('stimulus, expected', [('chirp', 0.), ('MB', 1.), ('dn', 2.), ('other', 3.)])
def test_get_file_info_row_mask_order(stimulus, expected):
    assert _row(f'SMP_M1_LR_GCL0_{stimulus}.h5')['mask_order'] == expected


@pytest.mark.parametrize('cond, penalty', [('control', 0.), ('C1', 0.), ('none', 0.), ('Cntr', 0.), ('drug', 0.1)])
def test_get_file_info_row_condition_penalty(cond, penalty):
    assert _row(f'SMP_M1_LR_GCL0_chirp_{cond}.h5')['mask_order'] == pytest.approx(penalty)


def test_get_file_info_row_condition_penalties_add_up():
    loc_mapper = dict(animal=1, region=2, field=3, stimulus=4, cond1=5, cond2=6, cond3=7)
    row = _row('SMP_M1_LR_GCL0_mb_drug_wash_control.h5', loc_mapper=loc_mapper)
    assert row['mask_order'] == pytest.approx(1.2)


@pytest.mark.parametrize('filename, kind', [
    ('SMP_M1_LR_OD_x.h5', 'od'),  # optic disk in field
    ('SMP_M1_LR_GCL0_opticdisk.h5', 'od'),  # optic disk in stimulus
    ('SMP_M1_OD.h5', 'od'),  # no stimulus at all
    ('SMP_M1_LR_GCL0_HR.h5', 'hr'),
    ('SMP_M1_LR_outline3.h5', 'outline'),  # numbered outline
    ('SMP_M1_LR_GCL0_Edge.h5', 'outline'),
    ('SMP_M1_LR_GCL0_outlines.h5', 'response'),  # suffix that is not a number
    ('SMP_M1_LR_GCL0_hr2.h5', 'response'),  # numbers only allowed for outlines
])
def test_get_file_info_row_kind(filename, kind):
    row = _row(filename)
    assert row['kind'] == kind
    if kind != 'response':
        assert row['mask_order'] == 4.  # after all mask aliases


def test_get_file_info_row_without_stimulus():
    with pytest.raises(ValueError, match='valid stimulus'):
        _row('SMP_M1_LR_GCL0.h5')


def test_get_file_info_row_with_dots_in_name():
    assert _row('SMP_M1_LR_GCL0_chirp.v2.h5')['stimulus'] == 'chirp.v2'


def test_get_file_info_df_pre(tmp_path):
    for name in ['SMP_M1_LR_GCL0_chirp.h5', 'SMP_M1_LR_GCL0_MB_drug.H5', 'SMP_M1_LR_OD.h5', '.SMP_M1_LR_GCL0_dn.h5',
                 'SMP_M1_LR_GCL1.h5', 'M1_LR_GCL0_chirp.smp', 'notes.txt']:
        (tmp_path / name).write_text('')
    with pytest.warns(UserWarning, match='SMP_M1_LR_GCL1.h5'):
        df = fu.get_file_info_df(str(tmp_path), USER_DICT, from_raw_data=False)
    df = df.sort_values('filepath').reset_index(drop=True)
    assert sorted(os.path.basename(f) for f in df.filepath) == [
        'SMP_M1_LR_GCL0_MB_drug.H5', 'SMP_M1_LR_GCL0_chirp.h5', 'SMP_M1_LR_OD.h5']
    by_name = {os.path.basename(r.filepath): r for r in df.itertuples()}
    assert by_name['SMP_M1_LR_GCL0_chirp.h5'].mask_order == 0.
    assert by_name['SMP_M1_LR_GCL0_MB_drug.H5'].mask_order == pytest.approx(1.1)
    assert by_name['SMP_M1_LR_OD.h5'].kind == 'od'
    assert 'cond2' not in df.columns


def test_get_file_info_df_raw_shifts_locations(tmp_path):
    for name in ['M1_LR_GCL0_chirp.smp', 'M1_LR_GCL0_MB.SMP', 'SMP_M1_LR_GCL0_chirp.h5']:
        (tmp_path / name).write_text('')
    df = fu.get_file_info_df(str(tmp_path), USER_DICT, from_raw_data=True)
    assert sorted(df.stimulus) == ['MB', 'chirp']
    assert set(df.field) == {'GCL0'}
    assert set(df.animal) == {'M1'}


def test_get_file_info_df_empty(tmp_path):
    assert fu.get_file_info_df(str(tmp_path), USER_DICT, from_raw_data=False).empty


def test_is_alias_number_match():
    assert fu.is_alias_number_match('outline', 'outline')
    assert fu.is_alias_number_match('outline12', 'outline')
    assert not fu.is_alias_number_match('outlinex', 'outline')
    assert not fu.is_alias_number_match('outlin', 'outline')
    assert not fu.is_alias_number_match('xoutline1', 'outline')


@pytest.mark.parametrize('loc, expected', [(None, None), (-1, None), (0, 'a'), (2, 'c'), (3, None)])
def test_get_file_info_at_loc(loc, expected):
    assert fu.get_file_info_at_loc(['a', 'b', 'c'], loc) == expected


def test_is_file_info_alias_match():
    info = dict(region='LR', field='OD', stimulus=None)
    assert fu.is_file_info_alias_match(info, aliases=['od'])
    assert not fu.is_file_info_alias_match(info, aliases=['od'], key_list=['region', 'stimulus'])
    assert not fu.is_file_info_alias_match(info, aliases=['od'], key_list=['missing'])
    assert fu.is_file_info_alias_match(dict(field='OD2'), aliases=['od'], allow_num_suffix=True)
    assert not fu.is_file_info_alias_match(dict(field='OD2'), aliases=['od'])


def test_find_folders_with_file_of_type(tmp_path):
    (tmp_path / 'a' / 'b').mkdir(parents=True)
    (tmp_path / 'c').mkdir()
    (tmp_path / 'd').mkdir()
    (tmp_path / 'a' / 'b' / 'header.ini').write_text('')
    (tmp_path / 'c' / '.hidden.ini').write_text('')
    (tmp_path / 'd' / 'x.ini.bak').write_text('')
    found = fu.find_folders_with_file_of_type(str(tmp_path), ending='.ini')
    assert sorted(found) == sorted([str(tmp_path / 'a' / 'b'), str(tmp_path / 'c')])
    found = fu.find_folders_with_file_of_type(str(tmp_path), ending='.ini', ignore_hidden=True)
    assert found == [str(tmp_path / 'a' / 'b')]


def _make_tree(root):
    (root / 'exp' / 'Pre').mkdir(parents=True)
    (root / 'exp' / 'Raw').mkdir()
    (root / 'exp' / 'header.ini').write_text('')
    (root / 'exp' / 'Pre' / 'SMP_M1_chirp.h5').write_text('')
    (root / 'exp' / 'Raw' / 'M1_chirp.SMP').write_text('')
    (root / 'exp' / 'Raw' / 'M1_chirp.smh').write_text('')


def test_print_tree(tmp_path, capsys):
    _make_tree(tmp_path)
    fu.print_tree(str(tmp_path), include_types=['.h5', 'SMP'])
    out = capsys.readouterr().out
    assert 'SMP_M1_chirp.h5' in out
    assert 'M1_chirp.SMP' in out
    assert 'header.ini' not in out
    assert 'M1_chirp.smh' not in out
    assert 'Pre/:  [1 files]' in out

    fu.print_tree(str(tmp_path), exclude_types=['h5'])
    out = capsys.readouterr().out
    assert 'SMP_M1_chirp.h5' not in out
    assert 'header.ini' in out

    fu.print_tree(str(tmp_path), nmax=1)
    out = capsys.readouterr().out
    assert out.rstrip().endswith('...')
    assert 'Pre' not in out


def test_print_tree_file_without_extension(tmp_path, capsys):
    _make_tree(tmp_path)
    (tmp_path / 'exp' / 'README').write_text('')
    (tmp_path / 'exp' / 'Pre' / 'SMP_M1_chirp.v2.h5').write_text('')
    fu.print_tree(str(tmp_path), include_types=['h5'])
    assert 'SMP_M1_chirp.v2.h5' in capsys.readouterr().out


def test_as_pre_filepath_custom_dirs():
    raw = os.path.join('root', 'exp', 'raw_data', 'M1_chirp.smp')
    assert fu.as_pre_filepath(raw, raw_data_dir='raw_data', pre_data_dir='pre_data') == os.path.join(
        'root', 'exp', 'pre_data', 'SMP_M1_chirp.h5')
    with pytest.raises(ValueError, match='not in pre dir'):
        fu.as_pre_filepath(os.path.join('root', 'Raw', 'SMP_M1_chirp.h5'))
