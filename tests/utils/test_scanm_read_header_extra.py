"""Additional tests of read_header_utils (type conversion, plain values, trigger counting of .smp files)."""
import os
from datetime import datetime

import numpy as np
import pytest

from djimaging.utils.scanm import read_header_utils as rhu
from djimaging.utils.scanm import read_smp_utils
from tests.utils.test_scanm_recording import FakeSMP


@pytest.mark.parametrize('key, expected', [
    ('RealPixelDuration_µs', 'RealPixelDuration_us'),
    ('RealPixelDuration_�s', 'RealPixelDuration_us'),
    ('TargetedPixelDuration_Âµs', 'TargetedPixelDuration_us'),
    ('Zoom', 'Zoom'),
])
def test_clean_key(key, expected):
    assert rhu._clean_key(key) == expected


@pytest.mark.parametrize('value, kind, expected', [
    ('3.6', int, 4), (np.float32(2.4), int, 2), ('2.5', float, 2.5), (b'7', float, 7.), (' n/a', float, None),
    ('', int, None), (None, int, None), ('abc', float, None), (np.nan, float, None), (np.inf, int, None),
    ([1, 2], float, None),
])
def test_to_number(value, kind, expected):
    assert rhu._to_number(value, kind) == expected


def test_to_plain():
    assert rhu.to_plain(b'abc') == 'abc'
    assert rhu.to_plain(np.bytes_(b'\xffx')) == '�x'
    assert rhu.to_plain(np.array([1, 2])) == [1, 2]
    value = rhu.to_plain(np.float32(1.5))
    assert value == 1.5 and type(value) is float
    assert rhu.to_plain('x') == 'x'


def test_normalize_header_types():
    header = rhu.normalize_header({'User_dxPix': '64.0', 'Zoom': b'1.5', 'User_Comment': b'hi', 'User_SetupID': 'x'},
                                  'h5')
    assert header['nxpix_full'] == 64 and isinstance(header['nxpix_full'], int)
    assert header['zoom'] == 1.5
    assert header['user_comment'] == 'hi'
    assert header['setupid'] is None
    assert header['objective'] is None


@pytest.mark.parametrize('header, scan_type, expected', [
    (dict(nxpix_full=10, real_pix_dur_us=2., nypix=5, nzpix=3), 'xz', 10 * 3 * 2e-6),
    (dict(nxpix_full=10, real_pix_dur_us=2., nypix=5, nzpix=3), 'xyz', 10 * 15 * 2e-6),
    (dict(nxpix_full=10, real_pix_dur_us=2.), 'line', 10 * 2e-6),
    (dict(nxpix_full=10, real_pix_dur_us=2.), 'unknown', None),
    (dict(nxpix_full=None, real_pix_dur_us=2.), 'xy', None),
    (dict(nxpix_full=10, real_pix_dur_us=0.), 'xy', None),
])
def test_frame_duration_s(header, scan_type, expected):
    assert rhu.frame_duration_s(header, scan_type) == (pytest.approx(expected) if expected else None)


def test_pixel_size_um_missing_nx_or_too_many_pixels():
    assert rhu.pixel_size_um(dict(nxpix_full=None, zoom=1.), scan_width_um=100.) is None
    assert rhu.pixel_size_um(dict(nxpix_full=10, nxpix_retrace=10, zoom=1.), scan_width_um=100.) is None
    assert rhu.pixel_size_um(dict(nxpix_full=10, zoom=1.), scan_width_um=100.) == pytest.approx(10.)


def test_count_triggers_in_stack_thresholds():
    nx, ny, nf = 4, 4, 10
    wparams = dict(user_dxpix=nx, user_dypix=ny, user_dzpix=0, user_nxpixlineoffs=0, user_npixretrace=0,
                   realpixdur=1000.)
    flat = np.zeros(nx * ny * nf)
    flat[[20, 60, 100]] = 40_000
    stack = flat.reshape(nf, ny, nx).T
    assert rhu.count_triggers_in_stack(stack, wparams) == 3
    # Low amplitude pulses: midpoint threshold
    assert rhu.count_triggers_in_stack(stack / 4, wparams) == 3
    # High baseline (uint16): no overflow in the midpoint
    assert rhu.count_triggers_in_stack((stack / 2 + 30_000).astype(np.uint16), wparams) == 3
    # Cannot be computed (z-stack): None
    assert rhu.count_triggers_in_stack(stack, dict(wparams, user_dzpix=5)) is None


def test_count_triggers_smp(monkeypatch):
    monkeypatch.setattr(read_smp_utils, 'SMP', FakeSMP)
    assert rhu.count_triggers_smp('x.smp') == 0


def test_count_triggers_smp_unreadable(tmp_path):
    with pytest.warns(UserWarning, match='Could not load'):
        assert rhu.count_triggers_smp(str(tmp_path / 'missing.smp')) is None


def test_file_size_and_mtime(tmp_path):
    path = tmp_path / 'a.bin'
    path.write_bytes(b'12345')
    os.utime(path, (0, 1_700_000_000.75))
    size, mtime = rhu.file_size_and_mtime(str(path))
    assert size == 5
    assert mtime == datetime.fromtimestamp(1_700_000_000)
