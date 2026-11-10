"""Regression tests for the ROI location templates (scalar field attributes must be fetched with fetch1)."""
import numpy as np
import pytest

from djimaging.tables.location.roi_location import RelativeRoiLocationTemplate, RelativeRoiLocationWrtFieldTemplate


class _FakeRestriction:
    """Restricted table with one row: fetch1 returns scalars, to_arrays returns 1-element arrays (DataJoint 2)."""

    def __init__(self, row):
        self.row = row

    def __call__(self):
        return self

    def __and__(self, other):
        return self

    def fetch1(self, *attrs):
        values = tuple(self.row[a] for a in attrs)
        return values[0] if len(values) == 1 else values

    def to_arrays(self, *attrs):
        values = tuple(np.array([self.row[a]]) for a in attrs)
        return values[0] if len(values) == 1 else values


FIELD_ROW = dict(pixel_size_um=2.0, z_step_um=1.0, scan_type='xy')


def _wrt_field_table(inserted):
    roi_mask = np.zeros((8, 8), dtype=int)
    roi_mask[1, 2] = -3  # ROI 3 (stored negative, as in Igor masks): one pixel at (1, 2), centre is (4, 4)

    class _Presentation(_FakeRestriction):
        ScanInfo = _FakeRestriction(dict(angle_deg=0.0))

    class Table(RelativeRoiLocationWrtFieldTemplate):
        roi_table = _FakeRestriction(dict(roi_id=3))
        roi_mask_table = _FakeRestriction(dict(roi_mask=roi_mask))
        field_table = _FakeRestriction(FIELD_ROW)
        presentation_table = _Presentation({})

        def insert1(self, row, **kwargs):
            inserted.append(row)

    return Table.__new__(Table)


def test_relative_roi_location_wrt_field_make_inserts_floats():
    inserted = []
    _wrt_field_table(inserted).make(dict(roi_id=3))

    assert len(inserted) == 1
    row = inserted[0]
    for attr in ('relx_wrt_field', 'rely_wrt_field', 'relz_wrt_field'):
        assert np.ndim(row[attr]) == 0
    # d1 = (1 - 4) * 2 um = -6, d2 = (2 - 4) * 2 um = -4; relx = -d2, rely = -d1
    assert row['relx_wrt_field'] == pytest.approx(4.0)
    assert row['rely_wrt_field'] == pytest.approx(6.0)
    assert row['relz_wrt_field'] == 0.


def test_relative_roi_location_make_adds_field_offset():
    inserted = []

    class Table(RelativeRoiLocationTemplate):
        relative_field_location_wrt_field_table = _FakeRestriction(dict(relx_wrt_field=-5.0, rely_wrt_field=3.0))
        relative_field_location_table = _FakeRestriction(dict(relx=100.0, rely=-200.0, relz=10.0))
        field_table = _FakeRestriction(FIELD_ROW)

        def insert1(self, row, **kwargs):
            inserted.append(row)

    Table.__new__(Table).make(dict(roi_id=3))

    assert inserted[0]['relx'] == pytest.approx(95.0)
    assert inserted[0]['rely'] == pytest.approx(-197.0)
    assert inserted[0]['relz'] == pytest.approx(10.0)
