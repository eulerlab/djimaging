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


# Real session (2022-02-15, xy-RGCs test data): per field the recordings MB, chirp, DN follow each other within ~15 s,
# the stimuli MovingBar (~101 s), Chirp (165 s), Noise (302 s) are shown during them. The QDSpy clock is ~14-20 s ahead.
_SESSION_RECS = [  # (recording, start in s of day (ScanM clock), scan duration in s)
    ('OpticDisc', 43514., 2.56), ('GCL0_MB_C1', 44053., 129.664), ('GCL0_chirp_C1', 44200., 199.04),
    ('GCL0_DN_C1', 44415., 327.936), ('GCL0_HR_C1', 44762., 10.015), ('GCL1_MB_C1', 44927., 152.704),
    ('GCL1_chirp_C1', 45093., 194.56), ('GCL1_DN_C1', 45301., 336.), ('GCL1_HR_C1', 45659., 10.015),
    ('GCL2_MB_C1', 45834., 127.744), ('GCL2_chirp_C1', 45975., 195.712), ('GCL2_DN_C1', 46190., 346.496),
    ('GCL2_HR_C1', 46552., 10.015), ('GCL3_MB_C1', 46676., 140.8), ('GCL3_chirp_C1', 46830., 193.92),
    ('GCL3_DN_C1', 47034., 336.768), ('GCL3_HR_C1', 47391., 10.015), ('GCL0_MB_C2', 47556., 129.92),
    ('GCL0_chirp_C2', 47697., 195.968), ('GCL0_DN_C2', 47907., 347.264), ('GCL0_HR_C2', 48269., 10.015),
    ('GCL1_MB_D', 49791., 0.), ('GCL1_chirp_D', 50055., 193.408), ('GCL1_DN_D', 50268., 339.712),
    ('GCL1_MB1_D', 50620., 128.), ('GCL1_chirp1_D', 50759., 192.768), ('GCL1_HR_D', 50977., 10.015),
    ('GCL2_MB_D', 51175., 153.344), ('GCL2_chirp_D', 51354., 190.72), ('GCL2_DN_D', 51558., 332.288),
    ('GCL2_HR_D', 51906., 10.015), ('GCL3_MB_D', 52084., 124.928), ('GCL3_chirp_D', 52218., 205.056),
    ('GCL3_DN_D', 52442., 337.024), ('GCL3_HR_D', 52794., 10.015),
]
_SESSION_STIMS = [  # (stimulus, start in s of day (QDSpy clock), duration in s); first 12: tests before recording
    ('MovingBar', 43225., 7.), ('MovingBar', 43268., 11.), ('Noise', 43281., 32.), ('MovingBar', 43316., 10.),
    ('MovingBar', 43353., 22.), ('MovingBar', 43401., 72.), ('MovingBar', 43554., 7.), ('MovingBar', 43673., 19.),
    ('MovingBar', 43770., 36.), ('MovingBar', 43810., 6.), ('MovingBar', 43868., 7.), ('MovingBar', 43958., 7.),
    ('MovingBar', 44073., 102.), ('Chirp', 44219., 165.), ('Noise', 44431., 302.),
    ('MovingBar', 44944., 101.), ('Chirp', 45113., 165.), ('Noise', 45324., 302.),
    ('MovingBar', 45852., 101.), ('Chirp', 45995., 165.), ('Noise', 46209., 302.),
    ('MovingBar', 46694., 101.), ('Chirp', 46847., 165.), ('Noise', 47059., 303.),
    ('MovingBar', 47574., 101.), ('Chirp', 47717., 165.), ('Noise', 47927., 303.),
    ('MovingBar', 49814., 101.), ('Chirp', 50073., 165.), ('Noise', 50294., 302.),
    ('MovingBar', 50637., 101.), ('Chirp', 50776., 165.), ('Noise', 51099., 6.),
    ('MovingBar', 51191., 102.), ('Chirp', 51369., 165.), ('Noise', 51580., 302.),
    ('MovingBar', 52098., 102.), ('Chirp', 52243., 165.), ('Noise', 52461., 302.),
]


def test_match_stimuli_to_recordings_uses_durations_for_back_to_back_recordings():
    """Regression: with start times only, shifting all stimuli by one recording (+289 s) put more stimulus starts
    into recordings than the true offset (~-14 s), but the stimuli then overran their recordings."""
    rec_starts = [r[1] for r in _SESSION_RECS]
    rec_ends = [r[1] + r[2] for r in _SESSION_RECS]
    stim_starts = [s[1] for s in _SESSION_STIMS]
    stim_durs = [s[2] for s in _SESSION_STIMS]

    offset, matches = ehu.match_stimuli_to_recordings(rec_starts, rec_ends, stim_starts, stim_durs_s=stim_durs)
    assert -20. <= offset <= -14.
    expected_kind = {'MovingBar': '_MB', 'Chirp': '_chirp', 'Noise': '_DN'}
    n_matched = 0
    for (stim, _, _), (idx, lag) in zip(_SESSION_STIMS, matches):
        if idx is None:
            continue
        n_matched += 1
        assert expected_kind[stim] in _SESSION_RECS[idx][0]
        assert 0. <= lag <= 20.
    # Not matched: 12 tests before the first recording, MovingBar of GCL1_MB_D (no scan time), 6 s Noise (no recording)
    assert n_matched == len(_SESSION_STIMS) - 14

    # Durations given via assign_stimuli_to_experiments as well; unknown durations (None, NaN, < 0) are allowed
    offset_a, _ = ehu.assign_stimuli_to_experiments(
        {'A': (rec_starts, rec_ends)}, 'A', stim_starts, stim_durs_s=stim_durs[:-2] + [None, -1.])
    assert -20. <= offset_a <= -14.


def test_name_keys_match_recording_tokens_to_qdspy_file_names():
    def match(token, file_name):
        return not ehu.recording_name_keys(token).isdisjoint(ehu.stimulus_name_keys(file_name))

    assert match('MB', 'RGC_MovingBar') and match('mb1', 'RGC_MovingBar') and match('movingbar', 'RGC_MovingBar')
    assert match('chirp', 'Chirp') and match('chirp2', 'Chirp')
    assert match('CSL', 'Contrast Saturation Local')
    assert match('noise', 'Noise_RGC') and match('DN', 'DenseNoise')
    assert not match('chirp', 'RGC_MovingBar') and not match('MB', 'Chirp') and not match('HR', 'Chirp')
    assert not match('DN', 'Noise')  # needs the Stimulus table, see below
    assert not match(None, 'Chirp') and not match('M', 'Moving') and not match('MB', None)
    # Stimulus table: recording listed for stimulus 'noise' (alias 'dn'), log entry identified as 'noise'
    assert not ehu.recording_name_keys('DN', ['noise']).isdisjoint(ehu.stimulus_name_keys('Noise', 'noise'))


def test_logged_stimulus_name():
    stimuli = [('chirp', 'h1', {'stim_name_list': ['Chirp']}), ('noise', 'h2', {'stim_hash_list': ['h3']}),
               ('noise2', 'h4', None)]
    assert ehu.logged_stimulus_name('h3', 'whatever', stimuli) == 'noise'
    assert ehu.logged_stimulus_name('x', 'Chirp', stimuli) == 'chirp'
    assert ehu.logged_stimulus_name('x', 'Other', stimuli) is None
    assert ehu.logged_stimulus_name(None, None, []) is None


# Real session (2022-01-25): per field the recordings MB, chirp, CSL and a 10 s HR scan follow each other within
# ~10 s, the stimuli RGC_MovingBar (101 s), Chirp (165 s), Contrast Saturation Local (182 s) start right after the
# recording start. Before the first field the moving bar was run 8 times as a test (aborted). The true clock offset
# is ~+250 s; shifting by one recording (+484 s) puts the stimuli into the next recording (MB -> chirp -> CSL -> HR).
_SESSION2_RECS = [  # (recording, start in s of day (ScanM clock), scan duration in s)
    ('OpticDisc', 52916., 2.56), ('GCL0_MB_C1', 53527., 142.464), ('GCL0_chirp_C1', 53680., 195.456),
    ('GCL0_CSL_C1', 53886., 215.168), ('GCL0_HR_C1', 54117., 10.0147), ('GCL1_MB_C1', 54208., 144.384),
    ('GCL1_chirp_C1', 54364., 200.192), ('GCL1_CSL_C1', 54576., 212.352), ('GCL1_HR_C1', 54804., 10.0147),
    ('GCL2_MB_C1', 54906., 135.04), ('GCL2_chirp_C1', 55056., 197.12), ('GCL2_CSL_C1', 55264., 219.52),
    ('GCL2_HR_C1', 55499., 10.0147), ('GCL3_MB_C1', 55595., 130.56), ('GCL3_chirp_C1', 55735., 192.512),
    ('GCL3_CSL_C1', 55941., 213.248), ('GCL3_HR_C1', 56173., 10.0147), ('GCL0_MB_C2', 56283., 136.832),
    ('GCL0_chirp_C2', 56430., 200.448), ('GCL0_CSL_C2', 56644., 214.528), ('GCL0_HR_C2', 56875., 10.0147),
    ('GCL1_MB_D', 58183., 131.584), ('GCL1_chirp_D', 58326., 201.728), ('GCL1_CSL_D', 58540., 220.416),
    ('GCL1_HR_D', 58774., 10.0147), ('GCL2_MB_D', 59176., 135.04), ('GCL2_chirp_D', 59320., 207.104),
    ('GCL2_CSL_D', 59543., 209.92), ('GCL2_HR_D', 59768., 10.0147), ('GCL3_MB_D', 59874., 160.768),
    ('GCL3_chirp_D', 60048., 195.328), ('GCL3_CSL_D', 60257., 210.816), ('GCL3_HR_D', 60486., 10.0147),
]
_MB, _CH, _CSL = 'RGC_MovingBar', 'Chirp', 'Contrast Saturation Local'
_SESSION2_STIMS = [  # (stimulus file, start in s of day (QDSpy clock), duration in s, aborted)
    (_MB, 52512., 13., True), (_MB, 52541., 11., True), (_MB, 52577., 32., True), (_MB, 52765., 12., True),
    (_MB, 52816., 11., True), (_MB, 52837., 11., True), (_MB, 52999., 20., True), (_MB, 53084., 10., True),
    (_MB, 53280., 101., False), (_CH, 53431., 165., False), (_CSL, 53637., 182., False),
    (_MB, 53959., 101., False), (_CH, 54116., 165., False), (_CSL, 54328., 182., False),
    (_MB, 54655., 101., False), (_CH, 54806., 165., False), (_CSL, 55015., 182., False),
    (_MB, 55347., 101., False), (_CH, 55484., 165., False), (_CSL, 55692., 182., False),
    (_MB, 56031., 101., False), (_CH, 56186., 165., False), (_CSL, 56396., 183., False),
    (_MB, 57934., 101., False), (_CH, 58078., 165., False), (_CSL, 58292., 182., False),
    (_MB, 58923., 101., False), (_CH, 59070., 165., False), (_CSL, 59290., 182., False),
    (_MB, 59625., 101., False), (_CH, 59799., 165., False), (_CSL, 60008., 182., False),
]


def _session2_inputs(tokens=None):
    tokens = tokens or [(r[0].split('_') + [''])[1] for r in _SESSION2_RECS]
    return dict(
        rec_starts_s=[r[1] for r in _SESSION2_RECS], rec_ends_s=[r[1] + r[2] for r in _SESSION2_RECS],
        stim_starts_s=[s[1] for s in _SESSION2_STIMS],
        rec_names=[ehu.recording_name_keys(t) if t else set() for t in tokens],
        stim_names=[ehu.stimulus_name_keys(s[0]) for s in _SESSION2_STIMS],
        stim_aborted=[s[3] for s in _SESSION2_STIMS])


def _assert_session2_fit(offset, matches):
    assert 245. <= offset <= 255.
    expected = {_MB: '_MB_', _CH: '_chirp_', _CSL: '_CSL_'}
    matched = [(s, _SESSION2_RECS[idx][0]) for s, (idx, _) in zip(_SESSION2_STIMS, matches) if idx is not None]
    assert len(matched) == 24  # all but the 8 test runs
    assert all(expected[s[0]] in rec for s, rec in matched)


def test_match_stimuli_to_recordings_uses_names():
    inputs = _session2_inputs()
    # Start times only: one recording too late (+484 s) puts more stimulus starts into recordings
    names = {k: inputs.pop(k) for k in ('rec_names', 'stim_names', 'stim_aborted')}
    offset, _ = ehu.match_stimuli_to_recordings(**inputs)
    assert offset > 450.
    # With the names the true shift wins, also without durations
    _assert_session2_fit(*ehu.match_stimuli_to_recordings(**inputs, **names))
    # ... and with durations
    durs = [s[2] for s in _SESSION2_STIMS]
    _assert_session2_fit(*ehu.match_stimuli_to_recordings(**inputs, **names, stim_durs_s=durs))


def test_match_stimuli_to_recordings_tolerates_a_wrong_name():
    tokens = [(r[0].split('_') + [''])[1] for r in _SESSION2_RECS]
    tokens[2], tokens[7] = 'MB', 'chirp'  # GCL0_chirp_C1 named MB, GCL1_CSL_C1 named chirp
    _assert_session2_fit(*ehu.match_stimuli_to_recordings(**_session2_inputs(tokens)))


def test_match_stimuli_to_recordings_ignores_aborted_test_runs_for_names():
    # One recording 'MB' at 1000-1100 s; a full moving bar at 500 s (QDSpy) and aborted test runs at 300 and 350 s.
    # Only the full run may name the recording: shift +500 s, not +700 s or +650 s.
    kw = dict(rec_names=[{'mb'}], stim_names=[ehu.stimulus_name_keys('RGC_MovingBar')] * 3)
    offset, matches = ehu.match_stimuli_to_recordings(
        [1000.], [1100.], [300., 350., 500.], stim_aborted=[True, True, False], stim_durs_s=[10., 10., 100.], **kw)
    assert offset == 500. and matches[2] == (0, 0.) and matches[0] == (None, None)


def test_assign_stimuli_to_experiments_uses_names():
    inputs = _session2_inputs()
    recs = {'A': (inputs['rec_starts_s'], inputs['rec_ends_s'], inputs['rec_names'])}
    offset, assigned = ehu.assign_stimuli_to_experiments(
        recs, 'A', inputs['stim_starts_s'], stim_names=inputs['stim_names'], stim_aborted=inputs['stim_aborted'])
    _assert_session2_fit(offset, [(a[1], a[2]) for a in assigned])
    with pytest.raises(ValueError):
        ehu.assign_stimuli_to_experiments({'A': ([1.], [2.], [])}, 'A', [1.], stim_names=[set()])


def test_match_stimuli_to_recordings_without_match():
    assert ehu.match_stimuli_to_recordings([], [], [1., 2.]) == (None, [(None, None), (None, None)])
    assert ehu.match_stimuli_to_recordings([0.], [10.], []) == (None, [])
    # Too far apart for the maximum clock offset
    offset, matches = ehu.match_stimuli_to_recordings([0.], [10.], [5000.], max_offset_s=900)
    assert offset is None and matches == [(None, None)]


def test_assign_stimuli_to_experiments_two_retinas_one_log():
    # Retina A recorded 1000-1100 and 1200-1300, retina B 3000-3100 (ScanM clock); QDSpy clock is 40 s behind.
    recs = {'A': ([1000., 1200.], [1100., 1300.]), 'B': ([3000.], [3100.])}
    scanm_times = [1005., 1030., 1205., 1230., 1150., 3005., 3050., 3080., 5000.]
    stims = [t - 40 for t in scanm_times]
    offset_a, a = ehu.assign_stimuli_to_experiments(recs, 'A', stims)
    offset_b, b = ehu.assign_stimuli_to_experiments(recs, 'B', stims)
    assert offset_a == offset_b
    assert 35 <= offset_a <= 40
    assert [x[0] for x in a] == ['recording'] * 4 + ['in_session'] + ['outside'] * 4
    assert [x[1] for x in a[:4]] == [0, 0, 1, 1]
    assert [x[0] for x in b] == ['outside'] * 5 + ['recording'] * 3 + ['outside']
    assert [x[1] for x in b[5:8]] == [0, 0, 0]


def test_assign_stimuli_to_experiments_between_goes_to_nearest():
    recs = {'A': ([1000.], [1100.]), 'B': ([1500.], [1600.])}
    stims = [1005., 1030., 1505., 1530., 1180., 1420.]
    _, a = ehu.assign_stimuli_to_experiments(recs, 'A', stims, pad_s=600.)
    _, b = ehu.assign_stimuli_to_experiments(recs, 'B', stims, pad_s=600.)
    assert [x[0] for x in a] == ['recording'] * 2 + ['outside'] * 2 + ['in_session', 'outside']
    assert [x[0] for x in b] == ['outside'] * 2 + ['recording'] * 2 + ['outside', 'in_session']


def test_assign_stimuli_to_experiments_single_experiment_and_no_match():
    _, a = ehu.assign_stimuli_to_experiments({'A': ([1000.], [1100.])}, 'A', [1005., 1030., 1200., 9000.], pad_s=600.)
    assert [x[0] for x in a] == ['recording', 'recording', 'in_session', 'outside']
    offset, a = ehu.assign_stimuli_to_experiments({'A': ([], [])}, 'A', [1., 2.])
    assert offset is None and [x[0] for x in a] == ['in_session'] * 2
