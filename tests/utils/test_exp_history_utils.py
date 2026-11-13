import os
from datetime import datetime, time, timedelta

import h5py
import numpy as np
import pytest

from djimaging.utils import exp_history_utils as ehu
from djimaging.utils.scanm import read_header_utils as rhu
from tests.fixtures import synthetic_scanm

USER = dict(
    datatype_loc=0, animal_loc=1, region_loc=2, field_loc=3, stimulus_loc=4, cond1_loc=5,
    cond2_loc=None, cond3_loc=None,
    opticdisk_alias='od_opticdisk_opticdisc', highres_alias='hq_hr_highresolution_512',
    outline_alias='outline_edge', mask_alias='chirp_mb_movingbar')


# ----------------------------------------------------------------------------------------------------------------------
# Header utils
# ----------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('date_stamp, time_stamp, expected', [
    ('2019-1-30', '19-44-4-6', datetime(2019, 1, 30, 19, 44, 4, 6000)),
    ('2022-01-25', '14-54-39-944', datetime(2022, 1, 25, 14, 54, 39, 944000)),
    ('2022-01-25', '14:54:39', datetime(2022, 1, 25, 14, 54, 39)),
    ('', '14-54-39-944', None),
    ('2022-01-25', None, None),
    ('n/a', '1-2-3-4', None),
])
def test_parse_scanm_datetime(date_stamp, time_stamp, expected):
    assert rhu.parse_scanm_datetime(date_stamp, time_stamp) == expected


@pytest.mark.parametrize('ny, nz, expected', [
    (64, 0, 'xy'), (64, 1, 'xy'), (1, 56, 'xz'), (64, 10, 'xyz'), (1, 0, 'line'), (None, None, 'unknown')])
def test_scan_type_from_dims(ny, nz, expected):
    assert rhu.scan_type_from_dims(ny, nz) == expected


def test_normalize_header_smh_and_h5_give_same_keys():
    smh = {'FrameWidth': 80, 'FrameHeight': 64, 'dZPixels': 0, 'PixRetraceLen': 10, 'XPixLineOffs': 6,
           'NumberOfFrames': 4000, 'FrameCounter': 2887, 'RealPixelDuration_\ufffds': 25.0, 'Zoom': 0.87,
           'SetupID': 1, 'DateStamp': '2022-1-25', 'TimeStamp': '14-52-6-877', 'Objective': 'n/a'}
    h5 = {'User_dxPix': np.float32(80), 'User_dyPix': np.float32(64), 'User_dzPix': np.float32(0),
          'User_nPixRetrace': 10., 'User_nXPixLineOffs': 6., 'RealPixDur': 25., 'Zoom': np.float32(0.87),
          'User_SetupID': 1., 'DateStamp_d_m_y': b'2022-1-25', 'TimeStamp_h_m_s_ms': b'14-52-6-877'}
    a, b = rhu.normalize_header(smh, 'smh'), rhu.normalize_header(h5, 'h5')
    assert set(a) == set(b) == set(rhu.HEADER_KEYS)
    for k in ('nxpix_full', 'nypix', 'nzpix', 'nxpix_retrace', 'nxpix_offset', 'real_pix_dur_us', 'setupid',
              'date_stamp', 'time_stamp'):
        assert a[k] == b[k], k
    assert a['zoom'] == pytest.approx(b['zoom'])
    assert a['n_frames_set'] == 4000 and b['n_frames_set'] is None
    assert rhu.n_frames_from_smh(a) == 4000 - 2887
    assert rhu.frame_duration_s(a, 'xy') == pytest.approx(80 * 64 * 25e-6)  # 7.8125 Hz
    with pytest.raises(ValueError):
        rhu.normalize_header(smh, 'igor')


@pytest.mark.parametrize('n_set, counter, expected', [(4000, 2887, 1113), (10, 0, 10), (10, None, 10), (None, 0, None)])
def test_n_frames_from_smh(n_set, counter, expected):
    assert rhu.n_frames_from_smh(dict(n_frames_set=n_set, frame_counter=counter)) == expected


def test_pixel_size_is_none_for_unknown_setup_or_zoom():
    header = dict(nxpix_full=80, nxpix_retrace=10, nxpix_offset=6, setupid=1, zoom=0.87)
    assert rhu.pixel_size_um(header) == pytest.approx(112. / 64 / 0.87)
    assert rhu.pixel_size_um(dict(header, setupid=4)) is None
    assert rhu.pixel_size_um(dict(header, zoom=10)) is None
    assert rhu.pixel_size_um(dict(header, zoom=None)) is None


def test_pixel_size_from_scan_width():
    """A scan width (e.g. from ExperimentImagingSetup) replaces the setupid constant, also for unknown setups."""
    header = dict(nxpix_full=80, nxpix_retrace=10, nxpix_offset=6, setupid=1, zoom=0.87)
    assert rhu.pixel_size_um(header, scan_width_um=100.) == pytest.approx(100. / 64 / 0.87)
    assert rhu.pixel_size_um(dict(header, setupid=4), scan_width_um=100.) == pytest.approx(100. / 64 / 0.87)
    assert rhu.pixel_size_um(dict(header, setupid=None), scan_width_um=100.) == pytest.approx(100. / 64 / 0.87)
    assert rhu.pixel_size_um(dict(header, zoom=10), scan_width_um=100.) is None
    assert rhu.setup_scan_width_um(1) == 112. and rhu.setup_scan_width_um(3) == 71.5
    assert rhu.setup_scan_width_um(4) is None and rhu.setup_scan_width_um(None) is None


def test_count_triggers_in_flat_stack():
    assert rhu.count_triggers_in_stack(np.zeros((4, 4, 10)), {}) == 0
    assert rhu.count_triggers_in_stack(np.zeros((0, 4, 10)), {}) is None


@pytest.fixture(scope='module')
def synthetic_pre_dir(tmp_path_factory):
    root = synthetic_scanm.generate_tutorial_dataset(tmp_path_factory.mktemp('scanm'))
    return root / synthetic_scanm.EXPERIMENT_DAY / synthetic_scanm.EXPERIMENT_NUMBER / 'Pre'


def test_read_h5_header(synthetic_pre_dir):
    filepath = sorted(synthetic_pre_dir.glob('*.h5'))[0]
    info = rhu.read_h5_header(str(filepath))
    with h5py.File(filepath, 'r') as f:
        n_frames = f['wDataCh0'].shape[-1]
        n_triggers = f['Triggertimes'].size
    assert info['is_scanm']
    assert info['n_frames'] == n_frames
    assert info['n_triggers'] == n_triggers > 0
    assert info['ch_names'] == ['wDataCh0', 'wDataCh1', 'wDataCh2']
    assert info['header']['nxpix_full'] == synthetic_scanm.NX
    assert info['header']['zoom'] == 1


def test_read_recording_scan_width(synthetic_pre_dir):
    """`scan_width_um` is stored with the recording; without an imaging setup it comes from the header setupid."""
    rec = ehu.find_recordings([str(synthetic_pre_dir)])[0]
    header = rec.h5.header
    nxpix = header['nxpix_full'] - (header['nxpix_retrace'] or 0) - (header['nxpix_offset'] or 0)

    entry = ehu.read_recording(rec, scan_width_um=150.)
    assert entry['scan_width_um'] == 150.
    assert entry['pixel_size_um'] == pytest.approx(150. / nxpix / header['zoom'])

    entry = ehu.read_recording(rec)  # synthetic files have no setupid
    assert header['setupid'] is None
    assert entry['scan_width_um'] is None and entry['pixel_size_um'] is None


def test_exp_history_uses_imaging_setup_table():
    from djimaging.tables.history import ExpHistoryTemplate

    class FakeImagingSetup:
        def __call__(self):
            return self

        @staticmethod
        def get_recording_params(key):
            return dict(scan_width_um=95.5, npixartifact=2)

    without = type('T', (ExpHistoryTemplate,), dict(experiment_table=None, userinfo_table=None))()
    with_setup = type('T', (ExpHistoryTemplate,), dict(
        experiment_table=None, userinfo_table=None, experiment_imaging_setup_table=FakeImagingSetup()))()
    assert without._scan_width_um(dict(experimenter='x')) is None
    assert with_setup._scan_width_um(dict(experimenter='x')) == 95.5
    assert 'scan_width_um = NULL' in ExpHistoryTemplate.Recording().definition


def test_read_h5_header_counts_triggers_without_triggertimes(synthetic_pre_dir, tmp_path):
    src = sorted(synthetic_pre_dir.glob('*.h5'))[0]
    dst = tmp_path / 'SMP_no_triggertimes.h5'
    with h5py.File(src, 'r') as f_in, h5py.File(dst, 'w') as f_out:
        n_triggers = f_in['Triggertimes'].size
        for k in f_in.keys():
            if k not in ('Triggertimes', 'Triggervalues'):
                f_in.copy(k, f_out)
    assert rhu.read_h5_header(str(dst))['n_triggers'] == n_triggers
    assert rhu.read_h5_header(str(dst), count_triggers=False)['n_triggers'] is None


def test_read_h5_header_of_non_scanm_file(tmp_path):
    filepath = tmp_path / 'root.h5'
    with h5py.File(filepath, 'w') as f:
        f.create_dataset('x', data=np.zeros(3))
    assert rhu.read_h5_header(str(filepath)) == dict(is_scanm=False)


# ----------------------------------------------------------------------------------------------------------------------
# Finding and linking files
# ----------------------------------------------------------------------------------------------------------------------
def test_recording_stem():
    assert ehu.recording_stem('/a/Pre/SMP_M1_LR_GCL0_chirp.h5') == 'M1_LR_GCL0_chirp'
    assert ehu.recording_stem('/a/Pre/smp_M1_LR_GCL0_chirp.H5') == 'M1_LR_GCL0_chirp'
    assert ehu.recording_stem('/a/Raw/M1_LR_GCL0_chirp.smp') == 'M1_LR_GCL0_chirp'
    assert ehu.recording_stem('/a/Raw/SMP_x.smp') == 'SMP_x'  # prefix only removed from .h5


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'')


def test_list_scanm_files_and_unscanned(tmp_path):
    exp = tmp_path / '20220125' / '1'
    for f in ['Raw/M1_a.smp', 'Raw/M1_a.smh', 'Raw/M1_b.smp', 'Raw/._M1_c.smp', 'Raw/notes.txt',
              'Pre/SMP_M1_a.h5', 'Pre/root.h5', 'Stacks/M1_zstack.smp', 'Stacks/M1_zstack.smh',
              'Morph/F_Morph.h5', 'Morph/SMP_M1_morph.h5']:
        _touch(exp / f)

    folders = ehu.resolve_data_dirs(str(exp), ['Pre', 'Raw', 'Raw', 'Missing'])
    assert [os.path.basename(f) for f in folders] == ['Pre', 'Raw']

    smps, h5s = ehu.list_scanm_files(folders)
    assert sorted(f.filename for f in smps) == ['M1_a.smp', 'M1_b.smp']
    assert sorted(f.filename for f in h5s) == ['SMP_M1_a.h5', 'root.h5']
    smh = {f.filename: f.smh for f in smps}
    assert os.path.basename(smh['M1_a.smp']) == 'M1_a.smh' and smh['M1_b.smp'] is None

    unscanned = [os.path.relpath(f, exp).replace('\\', '/') for f in ehu.find_unscanned_recording_files(
        str(exp), folders)]
    assert unscanned == ['Morph/SMP_M1_morph.h5', 'Stacks/M1_zstack.smp']


def _file(path, date='2022-1-25', time_stamp=None, orig=None, is_scanm=True):
    header = dict(date_stamp=date, time_stamp=time_stamp, orig_file_name=orig)
    f = ehu.ScanMFile(path=path, header=header if is_scanm else None)
    if not path.endswith('.smp'):
        f.h5_info = dict(is_scanm=is_scanm)
    return f


def test_link_smp_h5():
    smps = [
        _file('/e/Raw/20220111_M1_RR_xz0_LC.smp', time_stamp='15-29-8-117'),  # h5 was renamed: orig_file_name
        _file('/e/Raw/M1_RR_GCL0_chirp.smp', time_stamp='14-54-39-944'),  # same name, no orig in h5
        _file('/e/Raw/M1_odd_name.smp', time_stamp='16-00-00-001'),  # different name, same time stamp
        _file('/e/Raw/M1_RR_GCL0_HR.smp', time_stamp='15-01-57-120'),  # no h5
    ]
    h5s = [
        _file('/e/Pre/SMP_M1_RR_xz0_Lchirp_C1.h5', time_stamp='15-29-8-117',
              orig=r'C:\ScanM_Data\20220111_M1_RR_xz0_LC.smp'),
        _file('/e/Pre/SMP_M1_RR_GCL0_chirp.h5', time_stamp='14-54-39-944'),
        _file('/e/Pre/SMP_M1_other.h5', time_stamp='16-00-00-001'),
        _file('/e/Pre/SMP_M1_RR_GCL2_MB.h5', time_stamp='15-15-06-267'),  # no smp
        _file('/e/Pre/root.h5', is_scanm=False),  # not a recording
    ]
    recs = {r.rec_name: r for r in ehu.link_smp_h5(smps, h5s)}
    assert set(recs) == {'20220111_M1_RR_xz0_LC', 'M1_RR_GCL0_chirp', 'M1_odd_name', 'M1_RR_GCL0_HR',
                         'M1_RR_GCL2_MB'}
    assert recs['20220111_M1_RR_xz0_LC'].link_method == 'orig_file_name'
    assert recs['20220111_M1_RR_xz0_LC'].h5.filename == 'SMP_M1_RR_xz0_Lchirp_C1.h5'
    assert recs['M1_RR_GCL0_chirp'].link_method == 'file_name'
    assert recs['M1_odd_name'].link_method == 'time_stamp'
    assert recs['M1_odd_name'].h5.filename == 'SMP_M1_other.h5'
    assert (recs['M1_RR_GCL0_HR'].source, recs['M1_RR_GCL0_HR'].link_method) == ('smp', 'none')
    assert (recs['M1_RR_GCL2_MB'].source, recs['M1_RR_GCL2_MB'].link_method) == ('h5', 'none')
    assert recs['M1_RR_GCL0_chirp'].source == 'smp+h5'


def test_link_smp_h5_ambiguous_time_stamp_and_duplicate_names():
    smps = [_file('/e/Raw/a.smp', time_stamp='1-1-1-1'), _file('/e/Extra/a.smp', time_stamp='2-2-2-2')]
    h5s = [_file('/e/Pre/SMP_x.h5', time_stamp='3-3-3-3'), _file('/e/Pre/SMP_y.h5', time_stamp='3-3-3-3')]
    recs = ehu.link_smp_h5(smps, h5s)
    assert sorted(r.rec_name for r in recs) == ['Extra/a', 'Raw/a', 'x', 'y']
    assert all(r.link_method == 'none' for r in recs)


# ----------------------------------------------------------------------------------------------------------------------
# File names and fields
# ----------------------------------------------------------------------------------------------------------------------
def test_parse_recording_name():
    h5 = ehu.parse_recording_name('/e/Pre/SMP_M1_LR_GCL0_chirp_C1.h5', USER)
    smp = ehu.parse_recording_name('/e/Raw/M1_LR_GCL0_chirp_C1.smp', USER)
    for info in (h5, smp):
        assert (info['kind'], info['field'], info['stimulus'], info['region'], info['cond1']) == \
               ('response', 'GCL0', 'chirp', 'LR', 'C1')
    assert ehu.parse_recording_name('/e/Raw/M1_LR_GCL0_HR.smp', USER)['kind'] == 'hr'
    assert ehu.parse_recording_name('/e/Raw/M1_RR_OpticDisc.smp', USER)['kind'] == 'od'
    unknown = ehu.parse_recording_name('/e/Raw/M1_LR.smp', USER)  # no field or stimulus: does not raise
    assert unknown['kind'] == 'unknown' and unknown['field'] is None and unknown['region'] == 'LR'


def test_match_field_recordings():
    infos = {
        'chirp': dict(field='GCL0', region='LR', cond1='C1'),
        'mb_no_region': dict(field='GCL0', region=None, cond1=None),
        'other_field': dict(field='GCL1', region='LR', cond1=None),
        'hr': dict(field='GCL0', region='LR', cond1=None, kind='hr'),
        'no_field': dict(field=None, region='LR'),
        'none': None,
    }
    assert ehu.match_field_recordings(infos, dict(field='GCL0', region='LR'), ['field', 'region']) == ['chirp', 'hr']
    assert ehu.match_field_recordings(infos, dict(field='GCL0', region='N/A'), ['field', 'region']) == \
           ['mb_no_region']
    assert ehu.match_field_recordings(infos, dict(field='GCL0', region='LR', cond1='control'),
                                      ['field', 'region', 'cond1']) == ['hr']


# ----------------------------------------------------------------------------------------------------------------------
# Stimulus logs
# ----------------------------------------------------------------------------------------------------------------------
def test_seconds_of_day():
    assert ehu.seconds_of_day(timedelta(hours=1, seconds=2)) == 3602
    assert ehu.seconds_of_day(time(1, 0, 2)) == 3602
    assert ehu.seconds_of_day(datetime(2020, 1, 1, 1, 0, 2, 500000)) == 3602.5
    assert ehu.seconds_of_day(None) is None
    with pytest.raises(TypeError):
        ehu.seconds_of_day('01:00:02')


def test_match_stimuli_to_recordings_recovers_clock_offset():
    true_offset = 73.  # ScanM clock = QDSpy clock + 73 s
    rec_starts = np.array([1000., 1300., 1600., 2500.])
    rec_ends = rec_starts + 200.
    lags = np.array([5., 4., 6.])
    stim_starts_scanm = list(rec_starts[:3] + lags) + [2000.]  # the last stimulus: no recording (focus scan)
    stim_starts = np.array(stim_starts_scanm) - true_offset

    offset, matches = ehu.match_stimuli_to_recordings(rec_starts, rec_ends, stim_starts)
    assert true_offset - 6 <= offset <= true_offset  # lower estimate, within the smallest lag
    assert [m[0] for m in matches] == [0, 1, 2, None]
    assert matches[3] == (None, None)
    assert all(0 <= m[1] <= 6 + 1e-6 for m in matches[:3])


def test_match_stimuli_to_recordings_without_match():
    assert ehu.match_stimuli_to_recordings([], [], [1., 2.]) == (None, [(None, None), (None, None)])
    assert ehu.match_stimuli_to_recordings([0.], [10.], []) == (None, [])
    # Too far apart for the maximum clock offset
    offset, matches = ehu.match_stimuli_to_recordings([0.], [10.], [5000.], max_offset_s=900)
    assert offset is None and matches == [(None, None)]
