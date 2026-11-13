"""Known cases for stimulus geometry and moving bar direction mapping.

The expected moving bar arrows and array transforms are copied from QDSpy-stimuli-Documentation
(stimuli/MovingBar/README_MovingBar.md, README.md "Orientation", stimuli/*/plot_*.ipynb).
"""
import numpy as np
import pytest

from djimaging.utils import stimulator as st
from djimaging.utils.scanm.setup_utils import get_retinal_position

SETUPS = {row['setupid']: row for row in st.DEFAULT_STIMULATORS}
SCRIPT_DIRS = [0, 180, 45, 225, 90, 270, 135, 315]

# Arrows in the chamber top view (dorsal/back up, curtain down) as (right, up)
ARROWS = {'→': (1, 0), '←': (-1, 0), '↑': (0, 1), '↓': (0, -1),
          '↗': (1, 1), '↙': (-1, -1), '↖': (-1, 1), '↘': (1, -1)}

# README_MovingBar.md: "played on Setup 1/3"
DOCUMENTED_MB = {
    ('rgc_movingbar', '1'): '→ ← ↗ ↙ ↑ ↓ ↖ ↘',
    ('rgc_movingbar', '3'): '→ ← ↘ ↖ ↓ ↑ ↙ ↗',
    ('ds', '1'): '↑ ↓ ↖ ↘ ← → ↙ ↗',
    ('ds', '3'): '↓ ↑ ↙ ↗ ← → ↖ ↘',
}
# README_MovingBar.md: "local QDSpy" (screen, x right, y up)
DOCUMENTED_LOCAL = {'rgc_movingbar': '↓ ↑ ↙ ↗ ← → ↖ ↘', 'ds': '← → ↖ ↘ ↑ ↓ ↗ ↙'}


def _arrow_to_field_deg(arrow):
    right, up = ARROWS[arrow]
    return np.mod(np.rad2deg(np.arctan2(right, -up)), 360)  # field (XCoord, YCoord) = (-up, right)


@pytest.mark.parametrize('convention', ['rgc_movingbar', 'ds'])
def test_mb_local_qdspy_directions(convention):
    v = st.angles_to_vectors(SCRIPT_DIRS) @ st.script_to_screen_matrix(convention).T
    expected = [ARROWS[a] for a in DOCUMENTED_LOCAL[convention].split()]
    np.testing.assert_allclose(np.sign(np.round(v, 6)), np.sign(expected))


@pytest.mark.parametrize('convention,setupid', list(DOCUMENTED_MB))
def test_mb_documented_field_directions(convention, setupid):
    s = SETUPS[setupid]
    res = st.map_mb_directions(SCRIPT_DIRS, convention, s['screen_x_axis'], s['screen_y_axis'],
                               prepwmorient=0, eye='right')
    expected = [_arrow_to_field_deg(a) for a in DOCUMENTED_MB[(convention, setupid)].split()]
    np.testing.assert_allclose(st.wrap_deg(res['field_dirs_deg'] - expected), 0, atol=1e-6)


def test_readme_test_stimuli_mb0_mb90():
    """README.md: S1 MB0 front -> back, MB90 right -> left; S3 MB0 back -> front, MB90 right -> left (DS convention)."""
    for setupid, mb0, mb90 in [('1', '↑', '←'), ('3', '↓', '←')]:
        s = SETUPS[setupid]
        res = st.map_mb_directions([0, 90], 'ds', s['screen_x_axis'], s['screen_y_axis'], 0, 'right')
        np.testing.assert_allclose(res['field_dirs_deg'], [_arrow_to_field_deg(mb0), _arrow_to_field_deg(mb90)])


@pytest.mark.parametrize('eye,expected', [('right', [0, 180, 90, 270]), ('left', [180, 0, 90, 270])])
def test_mb_retinal_directions_setup1(eye, expected):
    # Setup 1, RGC_MovingBar, dorsal away from the experimenter: 0° -> right (YCoord+), 90° -> back (dorsal).
    # YCoord+ is nasal for the right eye and temporal for the left eye (README.md, SetupOrientationRetina.png).
    s = SETUPS['1']
    res = st.map_mb_directions([0, 180, 90, 270], 'rgc_movingbar', s['screen_x_axis'], s['screen_y_axis'], 0, eye)
    np.testing.assert_allclose(st.wrap_deg(res['retinal_dirs_deg'] - expected), 0, atol=1e-6)
    assert [st.cardinal(a) for a in res['retinal_dirs_deg']] == [st.cardinal(a) for a in expected]


def test_mb_retinal_unknown_eye_keeps_dorsal_component():
    s = SETUPS['3']
    res = st.map_mb_directions([0, 90, 270], 'rgc_movingbar', s['screen_x_axis'], s['screen_y_axis'], 0, 'unknown')
    assert np.all(np.isnan(res['retinal_dirs_deg']))
    np.testing.assert_allclose(res['dorsal_components'], [0, -1, 1], atol=1e-9)  # S3: 90° -> ↓ = ventral


def test_mb_unknown_orientation_or_convention():
    s = SETUPS['1']
    res = st.map_mb_directions([0, 90], 'ds', s['screen_x_axis'], s['screen_y_axis'], -1, 'right')
    assert np.all(np.isfinite(res['field_dirs_deg'])) and res['script_to_retina'] is None
    assert np.all(np.isnan(res['dorsal_components']))
    res = st.map_mb_directions([0, 90], None, s['screen_x_axis'], s['screen_y_axis'], 0, 'right')
    assert np.all(np.isnan(res['field_dirs_deg'])) and res['script_to_field'] is None


@pytest.mark.parametrize('prepwmorient', [0, 30, 90, 135, 180, 270])
@pytest.mark.parametrize('eye', ['left', 'right'])
def test_retinal_direction_consistent_with_location_tables(prepwmorient, eye):
    """A displacement in the field maps like the difference of two RetinalFieldLocation positions."""
    rng = np.random.default_rng(prepwmorient)
    p0, d = rng.normal(size=2) * 100, rng.normal(size=2) * 50
    vd0, tn0 = get_retinal_position(*p0, rotation=prepwmorient, eye=eye)
    vd1, tn1 = get_retinal_position(*(p0 + d), rotation=prepwmorient, eye=eye)
    nasal_dorsal = st.field_to_retina(d, prepwmorient, eye)[0]
    np.testing.assert_allclose(nasal_dorsal, [tn1 - tn0, vd1 - vd0], atol=1e-9)


@pytest.mark.parametrize('convention', ['rgc_movingbar', 'ds'])
@pytest.mark.parametrize('setupid', ['1', '3'])
def test_pref_dir_maps_like_directions(convention, setupid):
    s = SETUPS[setupid]
    res = st.map_mb_directions(SCRIPT_DIRS, convention, s['screen_x_axis'], s['screen_y_axis'], 45, 'left')
    for script_deg, retinal_deg in zip(SCRIPT_DIRS, res['retinal_dirs_deg']):
        angle, dorsal = st.map_pref_dir(np.deg2rad(script_deg), res['script_to_retina'])
        assert abs(st.wrap_deg(angle - retinal_deg)) < 1e-6
        np.testing.assert_allclose(dorsal, np.sin(np.deg2rad(retinal_deg)), atol=1e-9)


def test_rf_geometry_reproduces_former_rf_roi_offset():
    """Former RfRoiOffset: setup 1: rely = -rf_dx, relx = +rf_dy; setup 3: rely = -rf_dx, relx = -rf_dy."""
    rf_dx, rf_dy = 13., -7.
    for setupid, expected_relx, expected_rely in [('1', rf_dy, -rf_dx), ('3', -rf_dy, -rf_dx)]:
        s = SETUPS[setupid]
        m = st.array_to_field_matrix(st.NOISE_ARRAY_TO_SCREEN, s['screen_x_axis'], s['screen_y_axis'])
        np.testing.assert_allclose(m @ [rf_dx, rf_dy], [expected_relx, expected_rely])


def test_array_to_field_matches_documentation_notebooks():
    """plot_noise/plot_mousecam: setup1 = flip(rot90(x, 1, (1, 2)), 1), setup3 = rot90(x, 1, (1, 2))."""
    movie = np.arange(2 * 4 * 6).reshape(2, 4, 6)
    s1, s3 = SETUPS['1'], SETUPS['3']
    out1 = st.array_to_field(movie, st.QDSPY_EXPORT_ARRAY_TO_SCREEN, s1['screen_x_axis'], s1['screen_y_axis'])
    out3 = st.array_to_field(movie, st.QDSPY_EXPORT_ARRAY_TO_SCREEN, s3['screen_x_axis'], s3['screen_y_axis'])
    np.testing.assert_array_equal(out1, np.flip(np.rot90(movie, k=1, axes=(1, 2)), axis=1))
    np.testing.assert_array_equal(out3, np.rot90(movie, k=1, axes=(1, 2)))


def test_axes_to_matrix_validates():
    with pytest.raises(ValueError):
        st.axes_to_matrix('+x', '-x')
    with pytest.raises(ValueError):
        st.axes_to_matrix('+x', 'y')
    with pytest.raises(ValueError):
        st.array_to_field_matrix(None, '+x', '-y')


def test_ellipse_to_frame():
    m = st.array_to_field_matrix(st.NOISE_ARRAY_TO_SCREEN, '+x', '-y')
    w, h, a = st.ellipse_to_frame(4., 2., 0., m)
    assert (w, h) == (4., 2.) and abs(st.wrap_deg(a - np.rad2deg(np.arctan2(*(m @ [1, 0])[::-1])))) < 1e-9


# Stimulator assignment -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('setupid,date,expected_id,expected_conf', [
    ('1', '2023-05-01', 'setup1_2019-12-12', 'high'),
    (1, '2026-05-31', 'setup1_2025-02-17', 'high'),
    ('3', '2019-02-01', 'setup3_2016-11-14', 'medium'),  # before documented period
    ('1', '2026-09-01', 'setup1_2025-02-17', 'medium'),  # after documented period
    ('2', '2023-01-01', 'setup2_2018-11-16', 'medium'),  # assumed like setup 1
    ('4', '2023-01-01', None, 'low'),
    ('', '2023-01-01', None, 'low'),
])
def test_assign_stimulator(setupid, date, expected_id, expected_conf):
    stim_id, conf, reason = st.assign_stimulator(setupid, date, st.DEFAULT_STIMULATORS)
    assert (stim_id, conf) == (expected_id, expected_conf)
    assert (reason == '') == (conf == 'high')


def test_assign_stimulator_ambiguous_and_periods():
    rows = [dict(st.DEFAULT_STIMULATORS[0], stimulator_id='a', valid_from='2000-01-01', valid_to='2020-12-31'),
            dict(st.DEFAULT_STIMULATORS[0], stimulator_id='b', valid_from='2020-06-01', valid_to=None)]
    assert st.assign_stimulator('1', '2019-01-01', rows)[0] == 'a'
    assert st.assign_stimulator('1', '2022-01-01', rows)[0] == 'b'
    stim_id, conf, reason = st.assign_stimulator('1', '2020-07-01', rows)
    assert stim_id is None and conf == 'low' and 'Several' in reason


# Moving bar versions --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('md5,name,expected', [
    ('a574d90497d97ff9b93381f6b2eb207d', 'RGC_MovingBar', ('rgc_movingbar', 'high')),
    ('00bdaae87a8a50500c2063a68bfb3c1e', 'DS', ('ds', 'high')),
    ('31ca3c0d0ce464357ee7a7f5b57849c5', 'RGC_MovingBar_2', ('rgc_movingbar', 'medium')),
    ('bc083a0411701c4141e7b55f6a9c898b', 'RGC_MovingBar', ('rgc_movingbar', 'medium')),
    ('ffffffffffffffffffffffffffffffff', 'DS', ('ds', 'medium')),
    ('ffffffffffffffffffffffffffffffff', 'bar_test', (None, 'low')),
    (None, None, (None, 'low')),
])
def test_infer_mb_convention(md5, name, expected):
    convention, conf, reason = st.infer_mb_convention(md5, name, st.DEFAULT_MB_VERSIONS)
    assert (convention, conf) == expected
    assert (reason == '') == (conf == 'high')


# Retina status, cardinals, optic disc ---------------------------------------------------------------------------------
def test_retina_status():
    assert st.retina_status('right', 0)[:3] == (True, True, 'high')
    assert st.retina_status('unknown', 0)[:3] == (True, False, 'low')
    assert st.retina_status('left', -1)[:3] == (False, True, 'low')


def test_cardinal_and_optic_disc_angle():
    assert [st.cardinal(a) for a in (0, 44, 46, 180, 269, 316)] == \
           ['nasal', 'nasal', 'dorsal', 'temporal', 'ventral', 'nasal']
    assert st.cardinal(np.nan) is None
    # Field dorsal of the optic disc: dorsal motion is centrifugal, ventral motion centripetal
    assert st.angle_rel_optic_disc(90., 0., 500.) == 0.
    assert abs(st.angle_rel_optic_disc(270., 0., 500.)) == 180.
    assert st.angle_rel_optic_disc(0., 0., 500.) == -90.
    assert np.isnan(st.angle_rel_optic_disc(0., 10., 10.))


def test_combine_confidence():
    assert st.combine_confidence() == 'high'
    assert st.combine_confidence('high', 'medium') == 'medium'
    assert st.combine_confidence('medium', 'low', 'high') == 'low'
    with pytest.raises(ValueError):
        st.combine_confidence('unsure')


# Mouse cam ------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('params,version', [
    (dict(movName_Test='C:\\x\\test_images_rand_right.jpg', movName_Train='train_images_right.jpg'), 'v1'),
    (dict(movName_Test='MC2_test_rgb_72x64_right.jpg', movName_Train='MC2_train_rgb2_72x64_right.jpg'), 'v2'),
    (dict(movName_Test='MC2_test_rgb_64x72_left-new.jpg', movName_Train='MC2_train_rgb2_64x72_left-new.jpg'), 'v2new'),
    (dict(movName_Test='MC3_test_rgb_64x72_left.jpg'), 'v3'),
    (dict(movName_Test='MC3_test_rgb_64x72_left.jpg', movName_Train='MC2_train_rgb2_72x64_right.jpg'), None),
    ({}, None),
    (None, None),
])
def test_mc_version_from_params(params, version):
    assert st.mc_version_from_params(params) == version


@pytest.mark.parametrize('version,setupid,conf', [
    ('v1', '3', 'high'), ('v1', '1', 'low'), ('v2', '1', 'medium'), ('v2new', '1', 'high'), ('v3', '1', 'high'),
    ('v2new', '3', 'low'), (None, '1', 'low'),
])
def test_mc_confidence(version, setupid, conf):
    assert st.mc_confidence(version, setupid)[0] == conf


# Stimulator delay and imaging setup (formerly setupid constants of setup_utils) ----------------------------------------
@pytest.mark.parametrize('setupid', ['1', '2', '3'])
def test_default_stimulators_reproduce_stimulator_delay_csv(setupid):
    from djimaging.utils.scanm.setup_utils import get_stimulator_delay
    for date in np.arange(np.datetime64('2005-01-01'), np.datetime64('2027-06-01'), np.timedelta64(11, 'D')):
        stim_id, _, _ = st.assign_stimulator(setupid, str(date), st.DEFAULT_STIMULATORS)
        row = next(r for r in st.DEFAULT_STIMULATORS if r['stimulator_id'] == stim_id)
        assert row['stimulator_delay_ms'] / 1000. == pytest.approx(get_stimulator_delay(str(date), setupid))


def test_default_stimulators_periods_are_contiguous():
    for setupid, periods in st.stimulator_delay_periods().items():
        assert periods[-1][1] is None
        for (_, end, _), (start, _, _) in zip(periods[:-1], periods[1:]):
            assert np.datetime64(end, 'D') + np.timedelta64(1, 'D') == np.datetime64(start, 'D')


@pytest.mark.parametrize('setupid', ['1', '2', '3'])
def test_default_imaging_setups_reproduce_setup_utils(setupid):
    from djimaging.utils.scanm import setup_utils
    imaging_id, conf, reason = st.assign_imaging_setup(setupid, '2020-01-01', st.DEFAULT_IMAGING_SETUPS)
    row = next(r for r in st.DEFAULT_IMAGING_SETUPS if r['imaging_setup_id'] == imaging_id)
    assert row['scan_width_um'] == setup_utils.get_setup_xscale(setupid)
    assert row['npixartifact'] == setup_utils.get_npixartifact(setupid)
    assert setup_utils.pixel_size_from_scan_width(row['scan_width_um'], 64, 0.87) == \
        pytest.approx(setup_utils.get_pixel_size_xy_um(setupid, 64, 0.87))
    assert (conf == 'high') == (setupid != '2')
    assert (reason == '') == (conf == 'high')
