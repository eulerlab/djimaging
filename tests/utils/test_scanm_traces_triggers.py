"""Hand-crafted tests of ScanM frame timing, trigger detection and trace extraction.

Stacks are (x, y, frames); in time x is the fast axis:
sample (x, y, f) is at ``f * frame_dt + (y * nx_full + x) * pix_dt``.
"""
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from djimaging.utils.scanm import traces_and_triggers_utils as ttu
from djimaging.utils.scanm import wparams_utils
from tests.fixtures.synthetic_scanm import PIXEL_DURATION_US, STIMULATOR_DELAY_MS


def _stack_from_time_series(flat: np.ndarray, nx: int, ny: int) -> np.ndarray:
    """Inverse of `stack.T.flatten()`: samples ordered in time -> (x, y, frames)."""
    return flat.reshape(-1, ny, nx).T


def _wparams(nx=4, ny=3, nz=0, offs=0, retrace=0, pix_dur_us=10.):
    return dict(User_dxPix=nx, User_dyPix=ny, User_dzPix=nz, User_nXPixLineOffs=offs, User_nPixRetrace=retrace,
                RealPixDur=pix_dur_us)


# compute_frame_times

def test_compute_frame_times_pixel_precision():
    frame_times, offset, frame_dt = ttu.compute_frame_times(
        n_frames=3, pix_dt=1., npix_x=5, npix_2nd=2, npix_x_offset_left=1, npix_x_offset_right=2, precision='pixel')
    assert frame_dt == 10.
    assert_array_equal(frame_times, [0., 10., 20.])
    assert offset.shape == (2, 2)  # 5 - 1 - 2 active pixels, 2 lines
    assert_array_equal(offset, [[1., 6.], [2., 7.]])  # x=1,2 of line 0 and 1


def test_compute_frame_times_line_precision():
    _, offset, _ = ttu.compute_frame_times(
        n_frames=1, pix_dt=1., npix_x=5, npix_2nd=2, npix_x_offset_left=1, npix_x_offset_right=2, precision='line')
    assert_array_equal(offset, [[0., 5.], [0., 5.]])  # start of the line


def test_compute_frame_times_no_offsets():
    _, offset, _ = ttu.compute_frame_times(
        n_frames=1, pix_dt=1., npix_x=3, npix_2nd=2, npix_x_offset_left=0, npix_x_offset_right=0, precision='pixel')
    assert_array_equal(offset, [[0., 3.], [1., 4.], [2., 5.]])


@pytest.mark.parametrize('left, right', [(-1, 0), (0, -1)])
def test_compute_frame_times_negative_offsets(left, right):
    with pytest.raises(ValueError):
        ttu.compute_frame_times(n_frames=1, pix_dt=1., npix_x=3, npix_2nd=2,
                                npix_x_offset_left=left, npix_x_offset_right=right)


# compute_triggers

def _frame_timing(nx=4, ny=4, nf=10, precision='pixel'):
    return ttu.compute_frame_times(n_frames=nf, pix_dt=1e-3, npix_x=nx, npix_2nd=ny, npix_x_offset_left=0,
                                   npix_x_offset_right=0, precision=precision)


def test_compute_triggers_onsets_and_delay():
    nx, ny, nf = 4, 4, 10
    frame_times, offset, _ = _frame_timing(nx, ny, nf)
    flat = np.full(nx * ny * nf, 1000.)
    flat[17:25] = 40_000
    flat[60:70] = 50_000
    flat[159] = 45_000  # last sample
    triggertimes, triggervalues = ttu.compute_triggers(
        _stack_from_time_series(flat, nx, ny), frame_times, offset, threshold=30_000, stimulator_delay=0.5)
    assert_allclose(triggertimes, np.array([17, 60, 159]) * 1e-3 + 0.5)
    assert_array_equal(triggervalues, [40_000, 50_000, 45_000])


def test_compute_triggers_threshold_is_inclusive():
    nx, ny, nf = 4, 4, 4
    frame_times, offset, _ = _frame_timing(nx, ny, nf)
    flat = np.zeros(nx * ny * nf)
    flat[20:30] = 30_000
    triggertimes, _ = ttu.compute_triggers(_stack_from_time_series(flat, nx, ny), frame_times, offset)
    assert_allclose(triggertimes, [20e-3])


def test_compute_triggers_line_precision_uses_line_start():
    nx, ny, nf = 4, 4, 10
    frame_times, offset, _ = _frame_timing(nx, ny, nf, precision='line')
    flat = np.zeros(nx * ny * nf)
    flat[18:25] = 60_000  # line 4 (pixels 16..19)
    triggertimes, _ = ttu.compute_triggers(_stack_from_time_series(flat, nx, ny), frame_times, offset)
    assert_allclose(triggertimes, [16e-3])


def test_compute_triggers_glitches_closer_than_two_lines_are_merged():
    nx, ny, nf = 4, 4, 10
    frame_times, offset, _ = _frame_timing(nx, ny, nf)
    flat = np.zeros(nx * ny * nf)
    flat[[20, 22, 24]] = 60_000  # bouncing onset (< 2 lines = 8 samples)
    flat[40] = 60_000  # 20 samples later: separate trigger
    flat[48] = 60_000  # exactly 2 lines later: separate trigger
    triggertimes, _ = ttu.compute_triggers(_stack_from_time_series(flat, nx, ny), frame_times, offset)
    assert_allclose(triggertimes, np.array([20, 40, 48]) * 1e-3)


def test_compute_triggers_no_triggers():
    nx, ny, nf = 4, 4, 5
    frame_times, offset, _ = _frame_timing(nx, ny, nf)
    triggertimes, triggervalues = ttu.compute_triggers(np.zeros((nx, ny, nf)), frame_times, offset)
    assert triggertimes.size == 0 and triggervalues.size == 0


def test_compute_triggers_input_validation():
    frame_times, offset, _ = _frame_timing(4, 4, 5)
    with pytest.raises(ValueError, match='3d'):
        ttu.compute_triggers(np.zeros((4, 4)), frame_times, offset)
    with pytest.raises(ValueError, match='frame_times'):
        ttu.compute_triggers(np.zeros((4, 4, 6)), frame_times, offset)
    with pytest.raises(ValueError, match='frame_dt_offset'):
        ttu.compute_triggers(np.zeros((4, 3, 5)), frame_times, offset)
    with pytest.raises(ValueError, match='stack_times'):
        ttu.compute_triggers(np.zeros((4, 4, 5)), frame_times[::-1], offset)


def test_compute_triggers_trigger_on_at_start_is_extrapolated():
    nx, ny, nf = 4, 4, 10
    frame_times, offset, _ = _frame_timing(nx, ny, nf)
    flat = np.zeros(nx * ny * nf)
    flat[0:15] = 60_000  # trigger already on, 10 samples long, started 5 samples before recording... -> clipped
    flat[40:50] = 60_000
    flat[100:110] = 60_000
    triggertimes, _ = ttu.compute_triggers(_stack_from_time_series(flat, nx, ny), frame_times, offset)
    assert triggertimes.size == 3
    assert triggertimes[0] < triggertimes[1]
    assert_allclose(triggertimes[1:], [40e-3, 100e-3])


def test_compute_triggers_trigger_on_at_start_duration_off_by_one():
    nx, ny, nf = 4, 4, 10
    frame_times, offset, _ = _frame_timing(nx, ny, nf)
    flat = np.zeros(nx * ny * nf)
    flat[0:10] = 60_000  # same duration as the other triggers, started exactly at the first sample
    flat[40:50] = 60_000
    triggertimes, _ = ttu.compute_triggers(_stack_from_time_series(flat, nx, ny), frame_times, offset)
    assert_allclose(triggertimes, [0., 40e-3])


def test_compute_triggers_trigger_on_at_start_shorter_than_others():
    nx, ny, nf = 4, 4, 10
    frame_times, offset, _ = _frame_timing(nx, ny, nf)
    flat = np.zeros(nx * ny * nf)
    flat[0:2] = 60_000
    flat[40:50] = 60_000
    flat[100:110] = 60_000
    triggertimes, _ = ttu.compute_triggers(_stack_from_time_series(flat, nx, ny), frame_times, offset)
    assert np.all(np.diff(triggertimes) > 0)
    assert triggertimes[0] <= 0.


def test_compute_triggers_from_wparams_with_offsets():
    nx_full, ny, nf, offs, retrace = 6, 2, 5, 1, 1
    wparams = _wparams(nx=nx_full, ny=ny, offs=offs, retrace=retrace, pix_dur_us=1000.)
    stack = np.zeros((nx_full - offs - retrace, ny, nf))
    stack[2, 1, 3] = 60_000  # active x=2 -> full x=3, line 1, frame 3
    triggertimes, _ = wparams_utils.compute_triggers_from_wparams(stack, wparams, stimulator_delay=0.,
                                                                  precision='pixel')
    assert_allclose(triggertimes, [(3 * nx_full * ny + nx_full + 3) * 1e-3])
    triggertimes, _ = wparams_utils.compute_triggers_from_wparams(stack, wparams, stimulator_delay=0.,
                                                                  precision='line')
    assert_allclose(triggertimes, [(3 * nx_full * ny + nx_full) * 1e-3])


# wparams_utils

@pytest.mark.parametrize('ny, nz, expected', [(4, 0, 'xy'), (4, 1, 'xy'), (1, 5, 'xz'), (0, 5, 'xz'), (4, 5, 'xyz')])
def test_get_scan_type(ny, nz, expected):
    assert wparams_utils.get_scan_type(_wparams(ny=ny, nz=nz)) == expected
    lower = {k.lower(): v for k, v in _wparams(ny=ny, nz=nz).items()}
    assert wparams_utils.get_scan_type(lower, assume_lower=True) == expected


@pytest.mark.parametrize('nx, ny, nz', [(1, 4, 0), (4, 1, 1), (4, 0, 0)])
def test_get_scan_type_unknown(nx, ny, nz):
    with pytest.raises(NotImplementedError):
        wparams_utils.get_scan_type(_wparams(nx=nx, ny=ny, nz=nz))


def test_compute_frame_times_from_wparams_xz():
    wparams = _wparams(nx=5, ny=1, nz=3, offs=1, retrace=1, pix_dur_us=2.)
    frame_times, offset, frame_dt = wparams_utils.compute_frame_times(wparams, n_frames=2, precision='pixel')
    assert frame_dt == pytest.approx(5 * 3 * 2e-6)
    assert offset.shape == (3, 3)
    assert_allclose(frame_times, [0., frame_dt])


def test_compute_frame_times_from_wparams_errors():
    with pytest.raises(ValueError, match='precision'):
        wparams_utils.compute_frame_times(_wparams(), n_frames=2, precision='frame')
    with pytest.raises(NotImplementedError):
        wparams_utils.compute_frame_times(_wparams(nz=3), n_frames=2)


def test_check_dims_ch_stack_wparams():
    wparams = {k.lower(): v for k, v in _wparams(nx=6, ny=3, nz=2, offs=1, retrace=2).items()}
    wparams_utils.check_dims_ch_stack_wparams(np.zeros((3, 3, 4)), wparams)
    wparams_utils.check_dims_ch_stack_wparams(np.zeros((3, 2, 4)), wparams)
    with pytest.raises(ValueError, match='Stack shape'):
        wparams_utils.check_dims_ch_stack_wparams(np.zeros((6, 3, 4)), wparams)
    with pytest.raises(ValueError, match='Stack shape'):
        wparams_utils.check_dims_ch_stack_wparams(np.zeros((3, 4, 4)), wparams)


# compute_traces

def test_compute_traces_handcrafted():
    nx, ny, nf = 4, 3, 5
    wparams = _wparams(nx=nx, ny=ny, pix_dur_us=1000.)
    roi_mask = np.ones((nx, ny), dtype=int)
    roi_mask[0:2, 0] = -1  # line 0
    roi_mask[3, 1:3] = -3  # lines 1 and 2
    stack = np.zeros((nx, ny, nf))
    stack[0, 0, :] = np.arange(nf)
    stack[1, 0, :] = 10.
    stack[3, 1:3, :] = 7.

    roi_ids, traces, traces_times, frame_dt = ttu.compute_traces(stack, roi_mask, wparams, precision='pixel')
    assert_array_equal(roi_ids, [-1, -3])
    assert frame_dt == pytest.approx(12e-3)
    assert_allclose(traces[:, 0], (np.arange(nf) + 10.) / 2)
    assert_allclose(traces[:, 1], 7.)
    frame_times = np.arange(nf) * 12e-3
    assert_allclose(traces_times[:, 0], frame_times + 0.5e-3)  # median of x=0,1 in line 0
    assert_allclose(traces_times[:, 1], frame_times + (4 + 3 + 8 + 3) / 2 * 1e-3)

    _, _, traces_times_line, _ = ttu.compute_traces(stack, roi_mask, wparams, precision='line')
    assert_allclose(traces_times_line[:, 0], frame_times)
    assert_allclose(traces_times_line[:, 1], frame_times + 6e-3)


def test_compute_traces_errors():
    wparams = _wparams(nx=4, ny=3)
    roi_mask = -np.ones((4, 3), dtype=int)
    roi_mask[0, 0] = 0
    with pytest.raises(ValueError, match='3d'):
        ttu.compute_traces(np.zeros((4, 3)), roi_mask, wparams)
    with pytest.raises(ValueError, match='must match'):
        ttu.compute_traces(np.zeros((4, 4, 2)), roi_mask, wparams)


# check_valid_triggers_rel_to_tracetime

@pytest.mark.parametrize('trace_valid, triggertimes, expected', [
    (1, [], 1),
    (0, [], 1),
    (0, [2., 3.], 0),
    (1, [2., 3.], 1),
    (1, [0., 10.], 1),  # boundaries are inclusive
    (1, [-0.1, 3.], 0),
    (1, [2., 10.1], 0),
])
def test_check_valid_triggers_rel_to_tracetime(trace_valid, triggertimes, expected):
    trace_times = np.linspace(0., 10., 11)
    assert ttu.check_valid_triggers_rel_to_tracetime(trace_valid, trace_times, np.array(triggertimes)) == expected


# roi2trace_from_stack with the synthetic recording

def test_roi2trace_from_stack_synthetic(synthetic_recording_path: Callable[[str], Path]):
    filepath = synthetic_recording_path('chirp')
    roi_mask = np.ones((8, 8), dtype=int)
    roi_mask[3:5, 1:3] = -1
    roi_mask[5:7, 5:7] = -2

    roi2trace, frame_dt = ttu.roi2trace_from_stack(
        filepath=str(filepath), roi_ids=np.array([2]), roi_mask=roi_mask, data_stack_name='wDataCh0',
        precision='line')
    assert list(roi2trace) == [2]
    assert frame_dt == pytest.approx(64 * PIXEL_DURATION_US * 1e-6)
    import h5py
    with h5py.File(filepath, 'r') as f:
        traces = f['Traces0_raw'][()]
        tracetimes = f['Tracetimes0'][()]
    assert_allclose(roi2trace[2]['trace'], traces[:, 1])
    assert_allclose(roi2trace[2]['trace_times'], tracetimes[:, 1] - STIMULATOR_DELAY_MS / 1000)

    # Negative ids are accepted as well
    roi2trace_neg, _ = ttu.roi2trace_from_stack(
        filepath=str(filepath), roi_ids=np.array([-1, -2]), roi_mask=roi_mask, data_stack_name='wDataCh0',
        precision='line')
    assert sorted(roi2trace_neg) == [1, 2]


def test_roi2trace_from_stack_missing_rois(synthetic_recording_path: Callable[[str], Path]):
    filepath = str(synthetic_recording_path('chirp'))
    roi_mask = np.ones((8, 8), dtype=int)
    roi_mask[3:5, 1:3] = -1
    with pytest.raises(ValueError, match=r'missing in the mask: .*3'):
        ttu.roi2trace_from_stack(filepath=filepath, roi_ids=np.array([1, 3]), roi_mask=roi_mask,
                                 data_stack_name='wDataCh0', precision='line')
    roi2trace, _ = ttu.roi2trace_from_stack(filepath=filepath, roi_ids=np.array([1, 3]), roi_mask=roi_mask,
                                            data_stack_name='wDataCh0', precision='line', accept_missing_rois=True)
    assert list(roi2trace) == [1]


# roi2trace_from_stack_motion with the synthetic recording

def _synthetic_mask_and_weights():
    roi_mask = np.ones((8, 8), dtype=int)
    roi_mask[3:5, 1:3] = -1
    roi_mask[5:7, 5:7] = -2
    weights = np.stack([(roi_mask == -1), (roi_mask == -2), np.zeros_like(roi_mask, dtype=bool)]).astype(float)
    return roi_mask, weights


def test_roi2trace_from_stack_motion_without_motion_equals_static(synthetic_recording_path):
    filepath = str(synthetic_recording_path('MB'))
    roi_mask, weights = _synthetic_mask_and_weights()
    static, frame_dt = ttu.roi2trace_from_stack(filepath=filepath, roi_ids=np.array([1, 2]), roi_mask=roi_mask,
                                                data_stack_name='wDataCh0', precision='pixel')
    moved, frame_dt_m = ttu.roi2trace_from_stack_motion(
        filepath=filepath, roi_ids=np.array([1, 2]), roi_weights=weights[:2], weight_roi_ids=np.array([1, 2]),
        disp_frames=np.zeros((1200, 2)), data_stack_name='wDataCh0', precision='pixel')
    assert frame_dt_m == frame_dt
    for roi_id in (1, 2):
        assert_allclose(moved[roi_id]['trace'], static[roi_id]['trace'], rtol=1e-6)
        assert_allclose(moved[roi_id]['trace_times'], static[roi_id]['trace_times'])
        assert moved[roi_id]['trace_valid'] == 1


def test_roi2trace_from_stack_motion_invalid_rois(synthetic_recording_path):
    filepath = str(synthetic_recording_path('MB'))
    _, weights = _synthetic_mask_and_weights()
    disp = np.zeros((1200, 2))
    disp[100:110, 0] = 20.  # whole field moved out of the FOV for 10 frames
    with pytest.warns(UserWarning, match='NaN'):
        roi2trace, _ = ttu.roi2trace_from_stack_motion(
            filepath=filepath, roi_ids=np.array([-1, 3]), roi_weights=weights, weight_roi_ids=np.array([-1, -2, -3]),
            disp_frames=disp, data_stack_name='wDataCh0', precision='line')
    assert sorted(roi2trace) == [1, 3]
    assert roi2trace[1]['trace_valid'] == 0
    assert np.all(np.isnan(roi2trace[1]['trace'][100:110]))
    assert roi2trace[3]['trace_valid'] == 0  # no weight at all


def test_roi2trace_from_stack_motion_errors(synthetic_recording_path):
    filepath = str(synthetic_recording_path('MB'))
    _, weights = _synthetic_mask_and_weights()
    with pytest.raises(ValueError, match='frames'):
        ttu.roi2trace_from_stack_motion(filepath=filepath, roi_ids=np.array([1]), roi_weights=weights[:1],
                                        weight_roi_ids=np.array([1]), disp_frames=np.zeros((10, 2)),
                                        data_stack_name='wDataCh0', precision='line')
    with pytest.raises(ValueError, match=r'No ROI weights'):
        ttu.roi2trace_from_stack_motion(filepath=filepath, roi_ids=np.array([1, 2]), roi_weights=weights[:1],
                                        weight_roi_ids=np.array([1]), disp_frames=np.zeros((1200, 2)),
                                        data_stack_name='wDataCh0', precision='line')
