"""Tests for the field/ROI location tables (relative to the optic disk and retinal coordinates),
computed without a database."""

import numpy as np
import pytest

from djimaging.tables.location.location import (
    RelativeFieldLocationTemplate, RetinalFieldLocationTemplate,
    retinal_position_or_none)
from djimaging.tables.location.roi_location import RelativeRoiLocationTemplate, RetinalRoiLocationTemplate
from tests.tables._fake_dj import FakeRel, make_fake


# ---------------------------------------------------------------- retinal_position_or_none

@pytest.mark.parametrize('relx,rely,rotation,eye,expected', [
    # rotation 0: -XCoord is dorsal, +YCoord is nasal for the right eye and temporal for the left eye
    (-100., 0., 0., 'right', (100., 0.)),
    (0., 50., 0., 'right', (0., 50.)),
    (0., 50., 0., 'left', (0., -50.)),
    # rotation 90: the retina is rotated by 90 deg, so +YCoord is ventral
    (0., 100., 90., 'right', (-100., 0.)),
    (0., 100., 90., 'left', (-100., 0.)),
    (100., 0., 90., 'right', (0., -100.)),
    (100., 0., 90., 'left', (0., 100.)),
])
def test_retinal_position_known_geometry(relx, rely, rotation, eye, expected):
    vd, tn = retinal_position_or_none(relx, rely, rotation, eye)
    assert vd == pytest.approx(expected[0], abs=1e-9)
    assert tn == pytest.approx(expected[1], abs=1e-9)


@pytest.mark.parametrize('rotation', [0., 33., 90., 180., 270.])
def test_retinal_position_left_right_mirror_and_distance(rotation):
    relx, rely = 120., -70.
    vd_r, tn_r = retinal_position_or_none(relx, rely, rotation, 'right')
    vd_l, tn_l = retinal_position_or_none(relx, rely, rotation, 'left')
    assert vd_r == pytest.approx(vd_l)
    assert tn_r == pytest.approx(-tn_l)
    # Rotations preserve the distance to the optic disk
    assert np.hypot(vd_r, tn_r) == pytest.approx(np.hypot(relx, rely))


def test_retinal_position_full_rotation_is_identity():
    assert retinal_position_or_none(30., 40., 360., 'right') == pytest.approx(
        retinal_position_or_none(30., 40., 0., 'right'))


def test_retinal_position_unknown_eye_or_orientation():
    assert retinal_position_or_none(30., 40., None, 'right') == (None, None)
    vd, tn = retinal_position_or_none(30., 40., 0., 'unknown')
    assert vd == pytest.approx(-30.)
    assert tn is None
    assert isinstance(vd, float)


# ---------------------------------------------------------------- RelativeFieldLocation

def _rel_field_table(field_xyz, od_xyz=(1000., 2000., 50.), pres_xyz=None):
    pres_table = None
    if pres_xyz is not None:
        pres_table = FakeRel(rows=[dict(absx=x, absy=y, absz=z) for x, y, z in pres_xyz])
    return make_fake(RelativeFieldLocationTemplate, ['make'],
                     opticdisk_table=FakeRel(dict(odx=od_xyz[0], ody=od_xyz[1], odz=od_xyz[2])),
                     field_table=FakeRel(dict(absx=field_xyz[0], absy=field_xyz[1], absz=field_xyz[2])),
                     presentation_table=pres_table)


def _rel(entry):
    return entry['relx'], entry['rely'], entry['relz']


def test_relative_field_location_subtracts_optic_disk():
    table = _rel_field_table((1100., 1950., 60.))
    table.make(dict(field='f1', experimenter='x'))
    entry = table.inserted[0]
    assert _rel(entry) == pytest.approx((100., -50., 10.))
    assert entry['field'] == 'f1'


def test_relative_field_location_consistent_presentations():
    table = _rel_field_table((1100., 1950., 60.), pres_xyz=[(1101., 1951., 61.), (1099., 1949., 59.)])
    table.make(dict(field='f1'))
    assert _rel(table.inserted[0]) == pytest.approx((100., -50., 10.))


def test_relative_field_location_uses_non_outlier_presentations():
    pres = [(1300., 1950., 70.), (1302., 1952., 72.), (50_000., 1950., 60.)]
    table = _rel_field_table((0., 0., 0.), pres_xyz=pres)
    with pytest.warns(UserWarning, match='different from the presentation'):
        table.make(dict(field='f1'))
    assert _rel(table.inserted[0]) == pytest.approx((301., -49., 21.))


def test_relative_field_location_all_presentations_outliers_uses_median():
    pres = [(30_000., 30_000., 1.), (30_002., 30_002., 3.), (30_004., 30_004., 5.)]
    table = _rel_field_table((0., 0., 0.), pres_xyz=pres)
    with pytest.warns(UserWarning):
        table.make(dict(field='f1'))
    assert _rel(table.inserted[0]) == pytest.approx((29_002., 28_002., -47.))


# ---------------------------------------------------------------- RetinalFieldLocation / RetinalRoiLocation

class _Orientation(FakeRel):
    def __init__(self, eye, orientation_deg):
        super().__init__()
        self.eye, self.orientation_deg = eye, orientation_deg

    def get_eye_and_orientation(self, key):
        return self.eye, self.orientation_deg


@pytest.mark.parametrize('template,table_attr', [
    (RetinalFieldLocationTemplate, 'relativefieldlocation_table'),
    (RetinalRoiLocationTemplate, 'relative_roi_location_table'),
])
@pytest.mark.parametrize('eye,orientation,expected', [
    ('right', 0., (-100., 50.)),
    ('left', 0., (-100., -50.)),
    ('unknown', 0., (-100., None)),
    ('right', None, (None, None)),
])
def test_retinal_location_make(template, table_attr, eye, orientation, expected):
    table = make_fake(template, ['make'], retina_orientation_table=_Orientation(eye, orientation),
                      **{table_attr: FakeRel(dict(relx=100., rely=50.))})
    table.make(dict(field='f1'))
    entry = table.inserted[0]
    for name, exp in zip(['ventral_dorsal_pos_um', 'temporal_nasal_pos_um'], expected):
        if exp is None:
            assert entry[name] is None
        else:
            assert entry[name] == pytest.approx(exp)


# ---------------------------------------------------------------- RelativeRoiLocation

def test_relative_roi_location_adds_field_offset():
    table = make_fake(RelativeRoiLocationTemplate, ['make'],
                      field_table=FakeRel(dict(pixel_size_um=0.5, scan_type='xy')),
                      relative_field_location_wrt_field_table=FakeRel(dict(relx_wrt_field=10., rely_wrt_field=-5.)),
                      relative_field_location_table=FakeRel(dict(relx=100., rely=200., relz=30.)))
    table.make(dict(field='f1', roi_id=3))
    entry = table.inserted[0]
    assert (entry['relx'], entry['rely'], entry['relz']) == pytest.approx((110., 195., 30.))


def test_relative_roi_location_xz_not_implemented():
    table = make_fake(RelativeRoiLocationTemplate, ['make'],
                      field_table=FakeRel(dict(pixel_size_um=0.5, scan_type='xz')),
                      relative_field_location_wrt_field_table=FakeRel(dict(relx_wrt_field=10., rely_wrt_field=-5.)),
                      relative_field_location_table=FakeRel(dict(relx=100., rely=200., relz=30.)))
    with pytest.raises(NotImplementedError):
        table.make(dict(field='f1', roi_id=3))
