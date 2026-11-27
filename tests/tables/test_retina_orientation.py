import numpy as np
import pytest

from djimaging.tables.core.retina_orientation import header_has_orientation, orientation_from_header
from djimaging.tables.location.location import retinal_position_or_none


@pytest.mark.parametrize('eye,prepwmorient,has_orientation,expected', [
    ('right', 90, True, dict(eye='right', orientation_deg=90., source='header')),
    ('Left', 0, True, dict(eye='left', orientation_deg=0., source='header')),
    ('unknown', 180, True, dict(eye=None, orientation_deg=180., source='header')),
    ('right', -1, True, dict(eye='right', orientation_deg=None, source='header')),  # 'unknown' or 111 in header
    ('right', 0, False, dict(eye='right', orientation_deg=0., source='header_default')),  # not in header
    ('right', 0, None, dict(eye='right', orientation_deg=0., source='header')),  # header not readable: trust ExpInfo
])
def test_orientation_from_header(eye, prepwmorient, has_orientation, expected):
    assert orientation_from_header(eye, prepwmorient, has_orientation) == expected


def test_orientation_from_header_missing_as_unknown():
    out = orientation_from_header('left', 0, header_has_orientation=False, orientation_if_missing=None)
    assert out == dict(eye='left', orientation_deg=None, source='header_default')


@pytest.mark.parametrize('line,expected', [
    ('prepwmorient=90', True), ('prepwmorient=', False), (None, False)])
def test_header_has_orientation(tmp_path, line, expected):
    header = tmp_path / 'header.ini'
    header.write_text('[animal]\nstring_eye=left\n' + (f'[prep]\nint_{line}\n' if line else ''))
    assert header_has_orientation(str(header)) is expected


def test_header_has_orientation_unreadable(tmp_path):
    with pytest.warns(UserWarning):
        assert header_has_orientation(str(tmp_path / 'missing.ini')) is None


def test_retinal_position_or_none():
    assert retinal_position_or_none(100., 50., None, 'right') == (None, None)
    vd, tn = retinal_position_or_none(100., 50., 0., None)
    assert vd == pytest.approx(-100.) and tn is None
    vd, tn = retinal_position_or_none(100., 50., 0., 'right')
    assert (vd, tn) == (pytest.approx(-100.), pytest.approx(50.))
    assert all(isinstance(v, float) and np.isfinite(v) for v in (vd, tn))
