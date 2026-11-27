import numpy as np
import pytest

from djimaging.tables import receptivefield, stimulator
from djimaging.tables.receptivefield.rf_offset import compute_rf_roi_offset
from djimaging.tables.stimulator.mb_direction import compute_mb_direction_map, compute_retinal_pref_dir, \
    fetch_stim_log
from djimaging.utils import stimulator as st
from djimaging.utils.dj_utils import get_class_attributes, get_input_tables


def _test_definition(djclass):
    tables = get_input_tables(djclass().definition)
    class_attrs = get_class_attributes(djclass)
    for table in tables:
        table = table.split('.', 1)[0].removesuffix('()')
        assert table == 'master' or table in class_attrs, table


@pytest.mark.parametrize('djclass', [
    stimulator.StimulatorTemplate, stimulator.ExperimentStimulatorTemplate, stimulator.MovingBarVersionTemplate,
    stimulator.ImagingSetupTemplate, stimulator.ExperimentImagingSetupTemplate,
    stimulator.MbDirectionMapTemplate, stimulator.MbRetinalPrefDirTemplate, stimulator.MouseCamGeometryTemplate,
    receptivefield.RfRoiOffsetTemplate])
def test_definitions(djclass):
    _test_definition(djclass)


def _geometry(setupid, confidence='high', reason=''):
    row = {r['setupid']: r for r in st.DEFAULT_STIMULATORS}[setupid]
    return dict(screen_x_axis=row['screen_x_axis'], screen_y_axis=row['screen_y_axis'], offset_relx_um=0.,
                offset_rely_um=0., confidence=confidence, reason=reason)


# Known cases: (setup, stimulator confidence, md5, file name, eye, prepwmorient) -> confidence, retinal 0° direction
KNOWN_CASES = [
    # Documented setup era, documented version, eye and orientation known
    ('1', 'high', 'a574d90497d97ff9b93381f6b2eb207d', 'RGC_MovingBar', 'right', 0, 'high', 0.),
    ('1', 'high', 'a574d90497d97ff9b93381f6b2eb207d', 'RGC_MovingBar', 'left', 0, 'high', 180.),
    ('3', 'high', '00bdaae87a8a50500c2063a68bfb3c1e', 'DS', 'right', 0, 'high', 270.),  # S3 DS 0°: back -> front
    ('1', 'high', '00bdaae87a8a50500c2063a68bfb3c1e', 'DS', 'right', 180, 'high', 270.),  # S1 DS 0° front->back, rot.
    # Undocumented era (stimulator assignment medium)
    ('3', 'medium', 'a574d90497d97ff9b93381f6b2eb207d', 'RGC_MovingBar', 'right', 0, 'medium', 0.),
    # Unknown md5, convention from file name
    ('1', 'high', '31ca3c0d0ce464357ee7a7f5b57849c5', 'RGC_MovingBar_2', 'right', 0, 'medium', 0.),
    # Unknown md5 and file name
    ('1', 'high', 'ffffffffffffffffffffffffffffffff', 'bar', 'right', 0, 'low', np.nan),
    # No log
    ('1', 'high', None, None, 'right', 0, 'low', np.nan),
    # Eye unknown: no retinal angle
    ('1', 'high', 'a574d90497d97ff9b93381f6b2eb207d', 'RGC_MovingBar', 'unknown', 0, 'low', np.nan),
    # Orientation unknown
    ('1', 'high', 'a574d90497d97ff9b93381f6b2eb207d', 'RGC_MovingBar', 'right', -1, 'low', np.nan),
]


@pytest.mark.parametrize('setupid,stim_conf,md5,name,eye,orient,expected_conf,expected_deg', KNOWN_CASES)
def test_mb_direction_map_known_cases(setupid, stim_conf, md5, name, eye, orient, expected_conf, expected_deg):
    reason = 'outside documented period' if stim_conf != 'high' else ''
    row = compute_mb_direction_map(
        script_dirs_deg=[0, 180, 45, 225, 90, 270, 135, 315], stim_md5=md5, stim_file_name=name,
        log_reason='' if md5 else 'no QDSpy log tables in this schema', versions=st.DEFAULT_MB_VERSIONS,
        geometry=_geometry(setupid, stim_conf, reason), eye=eye, prepwmorient=orient)
    assert row['confidence'] == expected_conf
    assert (row['reason'] == '') == (expected_conf == 'high')
    if np.isnan(expected_deg):
        assert np.isnan(row['retinal_dirs_deg'][0])
    else:
        assert abs(st.wrap_deg(row['retinal_dirs_deg'][0] - expected_deg)) < 1e-6


def test_retinal_pref_dir():
    row = compute_mb_direction_map([0, 180, 45, 225, 90, 270, 135, 315], 'a574d90497d97ff9b93381f6b2eb207d',
                                   'RGC_MovingBar', '', st.DEFAULT_MB_VERSIONS, _geometry('1'), 'right', 0)
    # Script 90° on setup 1 (RGC_MovingBar) moves to the back: dorsal. Field 500 µm dorsal of the optic disc.
    out = compute_retinal_pref_dir(np.deg2rad(90), row['script_to_retina'], row['confidence'], row['reason'],
                                   od_pos=(0., 500.))
    assert out['pref_dir_cardinal'] == 'dorsal' and out['confidence'] == 'high'
    assert abs(out['pref_dir_retinal_deg'] - 90) < 1e-4 and abs(out['pref_dir_dorsal'] - 1) < 1e-9
    assert abs(out['pref_dir_rel_od_deg']) < 1e-4  # centrifugal

    # Eye unknown: only dorsal component
    row = compute_mb_direction_map([0, 90], 'a574d90497d97ff9b93381f6b2eb207d', 'RGC_MovingBar', '',
                                   st.DEFAULT_MB_VERSIONS, _geometry('1'), 'unknown', 0)
    out = compute_retinal_pref_dir(np.deg2rad(90), row['script_to_retina'], row['confidence'], row['reason'])
    assert out['pref_dir_retinal_deg'] is None and out['pref_dir_cardinal'] is None
    assert abs(out['pref_dir_dorsal'] - 1) < 1e-9 and out['confidence'] == 'low'

    # Unknown convention: nothing
    out = compute_retinal_pref_dir(np.deg2rad(90), None, 'low', 'unknown')
    assert out['pref_dir_retinal_deg'] is None and out['pref_dir_dorsal'] is None


@pytest.mark.parametrize('setupid', ['1', '3'])
def test_rf_roi_offset_matches_former_implementation(setupid):
    rng = np.random.default_rng(0)
    for rf_dx, rf_dy, relx, rely in rng.normal(size=(20, 4)) * 30:
        new = compute_rf_roi_offset(rf_dx, rf_dy, relx, rely, st.NOISE_ARRAY_TO_SCREEN, _geometry(setupid))
        if setupid == '1':
            old = (+rf_dy - relx, -rf_dx - rely)
        else:
            old = (-rf_dy - relx, -rf_dx - rely)
        np.testing.assert_allclose(new, old)


def test_rf_roi_offset_stimulator_offset():
    geometry = dict(_geometry('1'), offset_relx_um=5., offset_rely_um=-3.)
    assert compute_rf_roi_offset(0., 0., 1., 1., st.NOISE_ARRAY_TO_SCREEN, geometry) == (4., -4.)


def test_fetch_stim_log_without_tables():
    log, reason = fetch_stim_log(None, None, {})
    assert log is None and 'no QDSpy log' in reason


@pytest.mark.parametrize('kwargs,expected', [
    (dict(setup_id=3), (71.5, 3)),
    (dict(setup_id=3, scan_width_um=100., npixartifact=6), (100., 6)),
    (dict(scan_width_um=100., npixartifact=0), (100., 0)),
])
def test_scanm_recording_setup_params(synthetic_recording_path, kwargs, expected):
    """Values of the setup tables (scan width, light artifact rows) override the setupid constants."""
    from djimaging.utils.scanm.recording import ScanMRecording
    rec = ScanMRecording(filepath=str(synthetic_recording_path("chirp")), **kwargs)
    scan_width_um, npixartifact = expected
    assert rec.pix_n_artifact == npixartifact
    assert rec.pix_dx_um == pytest.approx(scan_width_um / rec.pix_nx / rec.obj_zoom)


def test_recording_setup_mixin():
    from djimaging.tables.core.setup_params import RecordingSetupMixin

    class ExpInfo:
        def __and__(self, key):
            return self

        def fetch1(self, attr):
            return '1'

    class Experiment:
        def __init__(self):
            self.ExpInfo = ExpInfo()

    class ExpStim:
        def get_stimulator_delay_s(self, key):
            return 0.064

    class ExpImaging:
        def get_recording_params(self, key):
            return dict(scan_width_um=112., npixartifact=1)

    class Legacy(RecordingSetupMixin):
        experiment_table = Experiment

    class WithTables(Legacy):
        experiment_stimulator_table = ExpStim
        experiment_imaging_setup_table = ExpImaging

    key = dict(date='2023-01-01')
    assert Legacy().recording_setup_kwargs(key) == dict(setup_id='1', date='2023-01-01')
    assert WithTables().recording_setup_kwargs(key) == dict(
        setup_id='1', date='2023-01-01', stimulator_delay=0.064, scan_width_um=112., npixartifact=1)
