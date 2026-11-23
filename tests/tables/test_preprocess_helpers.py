"""Tests for module-level (DB-free) helpers in djimaging.tables.core.{preprocesstraces,snippets,averages}."""
import numpy as np
import pytest

from djimaging.tables.core import preprocesstraces as pp
from djimaging.tables.core.averages import compute_upsampled_average
from djimaging.tables.core.snippets import fetch_snippets_and_times, get_aligned_snippets_times
from djimaging.utils.filter_utils import lowpass_filter_trace


# ---------------------------------------------------------------------------
# drop_left_and_right
# ---------------------------------------------------------------------------

def test_drop_left_and_right_noop_when_disabled():
    trace = np.array([-9., 1., 2., 3., 9.])
    out = pp.drop_left_and_right(trace, drop_nmin_lr=(0, 0), drop_nmax_lr=(0, 0))
    np.testing.assert_array_equal(out, trace)


def test_drop_left_and_right_only_leading_out_of_range_run_is_replaced():
    trace = np.array([9., 0.5, -9., 0.2, 0.3, 0.4, 0.1, 0.6, 0.5, 0.4])
    out = pp.drop_left_and_right(trace, drop_nmin_lr=(0, 0), drop_nmax_lr=(3, 0))
    # Only the first sample is out of range before the first in-range one
    np.testing.assert_array_equal(out[:3], [0.5, 0.5, -9.])
    np.testing.assert_array_equal(out[3:], trace[3:])


def test_drop_left_and_right_right_only():
    trace = np.array([0.5, 0.2, 0.3, 0.4, 0.1, 0.6, 0.5, 9., -9.])
    out = pp.drop_left_and_right(trace, drop_nmin_lr=(0, 0), drop_nmax_lr=(0, 2))
    np.testing.assert_array_equal(out, [0.5, 0.2, 0.3, 0.4, 0.1, 0.6, 0.5, 0.5, 0.5])


def test_drop_left_and_right_inplace():
    trace = np.array([-9., 0.2, 0.3, 0.4, 0.1, 0.6])
    out = pp.drop_left_and_right(trace, drop_nmin_lr=(0, 0), drop_nmax_lr=(1, 0), inplace=True)
    assert out is trace
    assert trace[0] == 0.2

    trace = np.array([-9., 0.2, 0.3, 0.4, 0.1, 0.6])
    pp.drop_left_and_right(trace, drop_nmin_lr=(0, 0), drop_nmax_lr=(1, 0), inplace=False)
    assert trace[0] == -9.


def test_drop_left_and_right_nmin_left():
    out = pp.drop_left_and_right(np.arange(10.), drop_nmin_lr=(2, 0), drop_nmax_lr=(0, 0))
    np.testing.assert_array_equal(out, [2., 2., 2., 3., 4., 5., 6., 7., 8., 9.])


@pytest.mark.xfail(strict=True, reason="BUG: right drop_nmin fills with trace[-n] (a dropped sample) "
                                       "instead of trace[-(n + 1)]")
def test_drop_left_and_right_nmin_right():
    out = pp.drop_left_and_right(np.arange(10.), drop_nmin_lr=(0, 2), drop_nmax_lr=(0, 0))
    np.testing.assert_array_equal(out, [0., 1., 2., 3., 4., 5., 6., 7., 7., 7.])


# ---------------------------------------------------------------------------
# detrend_trace / non_negative_trace
# ---------------------------------------------------------------------------

def test_detrend_trace_removes_polynomial_trend():
    fs = 10.
    t = np.arange(200) / fs
    trend = 0.5 * t ** 2 - 3 * t + 2
    detrended, smoothed = pp.detrend_trace(trend, window_len_seconds=2., fs=fs, poly_order=2)
    # Away from the edges (mode='nearest') a quadratic is reproduced exactly
    np.testing.assert_allclose(smoothed[20:-20], trend[20:-20], atol=1e-8)
    np.testing.assert_allclose(detrended + smoothed, trend)


def test_detrend_trace_window_is_odd_number_of_frames():
    # 1 s at 10 Hz = 10 frames -> 9 frames. A spike is spread over exactly 9 samples by a order-0 filter.
    trace = np.zeros(51)
    trace[25] = 9.
    _, smoothed = pp.detrend_trace(trace, window_len_seconds=1., fs=10., poly_order=0)
    np.testing.assert_allclose(smoothed[21:30], 1.)
    np.testing.assert_allclose(smoothed[:21], 0., atol=1e-12)
    np.testing.assert_allclose(smoothed[30:], 0., atol=1e-12)


def test_non_negative_trace():
    trace = np.arange(41.)  # 2.5 percentile = 1
    out = pp.non_negative_trace(trace)
    assert out.min() == 0.
    np.testing.assert_array_equal(out[:3], [0., 0., 1.])
    assert out[-1] == 39.
    assert trace[0] == 0. and trace[-1] == 40.  # not inplace

    out = pp.non_negative_trace(trace, inplace=True)
    assert out is trace


# ---------------------------------------------------------------------------
# extract_baseline / subtract_baseline_trace
# ---------------------------------------------------------------------------

def test_extract_baseline_stim_before_trace_raises():
    with pytest.raises(ValueError):
        pp.extract_baseline(np.arange(5.), np.arange(5.) + 10, stim_start=9.)


def test_extract_baseline_contains_all_prestimulus_samples():
    times = np.arange(10.) + 10.
    baseline = pp.extract_baseline(np.arange(10.), times, stim_start=13.5)
    np.testing.assert_array_equal(baseline, [0., 1., 2., 3.])


def test_extract_baseline_single_prestimulus_sample_at_time_zero():
    # Valid input (process_trace only checks that some sample precedes stim_start)
    baseline = pp.extract_baseline(np.arange(10.) + 5., np.arange(10.), stim_start=0.5)
    assert len(baseline) <= 1
    assert np.all(baseline == 5.)


def test_subtract_baseline_trace_subtracts_prestim_median():
    times = np.arange(10.) + 10.
    trace = np.array([2., 2., 2., 2., 5., 6., 7., 8., 9., 10.])
    out = pp.subtract_baseline_trace(trace, times, stim_start=13.5)
    np.testing.assert_array_equal(out, trace - 2.)
    assert trace[0] == 2.  # not inplace


# ---------------------------------------------------------------------------
# process_trace
# ---------------------------------------------------------------------------

_NO_DROP = dict(drop_nmin_lr=(0, 0), drop_nmax_lr=(0, 0))


def test_process_trace_noop_returns_copy():
    trace = np.random.default_rng(0).normal(size=50)
    out, smoothed, dt = pp.process_trace(trace, trace_t0=1., trace_dt=0.1, **_NO_DROP)
    np.testing.assert_array_equal(out, trace)
    assert out is not trace
    np.testing.assert_array_equal(smoothed, 0.)
    assert dt == 0.1


def test_process_trace_default_drops_ramp_edges():
    out, _, _ = pp.process_trace(np.arange(20.), trace_t0=0., trace_dt=0.1)
    np.testing.assert_array_equal(out[:4], [3., 3., 3., 3.])
    np.testing.assert_array_equal(out[-4:], [16., 16., 16., 16.])


def test_process_trace_detrend_returns_trend():
    trace = np.random.default_rng(1).normal(size=200) + np.linspace(0, 10, 200)
    out, smoothed, _ = pp.process_trace(trace, trace_t0=0., trace_dt=0.1, window_len_seconds=3., poly_order=2,
                                        **_NO_DROP)
    np.testing.assert_allclose(out + smoothed, trace)
    assert abs(np.mean(out)) < 0.2


def test_process_trace_lowpass_matches_filter():
    trace = np.random.default_rng(2).normal(size=300)
    out, _, _ = pp.process_trace(trace, trace_t0=0., trace_dt=0.01, f_cutoff=5., **_NO_DROP)
    np.testing.assert_allclose(out, lowpass_filter_trace(trace, fs=100., f_cutoff=5.))
    out0, _, _ = pp.process_trace(trace, trace_t0=0., trace_dt=0.01, f_cutoff=0., **_NO_DROP)
    np.testing.assert_array_equal(out0, trace)


def test_process_trace_subtract_baseline_and_standardize_by_trace_std():
    trace = np.r_[np.full(10, 3.), np.arange(10.)]
    out, _, _ = pp.process_trace(trace, trace_t0=0., trace_dt=1., subtract_baseline=True, stim_start=10.,
                                 standardize=2, **_NO_DROP)
    expected = (trace - 3.) / np.std(trace - 3.)
    np.testing.assert_allclose(out, expected)
    assert np.std(out) == pytest.approx(1.)


def test_process_trace_standardize_by_baseline_std():
    rng = np.random.default_rng(3)
    trace = np.r_[rng.normal(scale=2., size=30), rng.normal(scale=10., size=30)]
    trace_times = np.arange(60) * 0.5 + 2.
    out, _, _ = pp.process_trace(trace, trace_t0=2., trace_dt=0.5, standardize=1, stim_start=17., **_NO_DROP)
    np.testing.assert_allclose(out, trace / np.std(pp.extract_baseline(trace, trace_times, 17.)))


def test_process_trace_non_negative():
    trace = np.random.default_rng(4).normal(size=100)
    out, _, _ = pp.process_trace(trace, trace_t0=0., trace_dt=0.1, non_negative=True, **_NO_DROP)
    assert out.min() == 0.


def test_process_trace_integer_downsample():
    trace = np.arange(20.)
    out, _, dt = pp.process_trace(trace, trace_t0=0., trace_dt=0.1, fs_resample=5., **_NO_DROP)
    assert dt == pytest.approx(0.2)
    np.testing.assert_allclose(out, np.arange(10) * 2 + 0.5)


def test_process_trace_non_integer_resample():
    trace = np.arange(21.)  # t = 0 ... 2 s at 10 Hz
    out, _, dt = pp.process_trace(trace, trace_t0=0., trace_dt=0.1, fs_resample=4., **_NO_DROP)
    assert dt == 0.25
    np.testing.assert_allclose(out, np.arange(9) * 2.5)


def test_process_trace_upsample():
    trace = np.arange(5.)
    out, _, dt = pp.process_trace(trace, trace_t0=3., trace_dt=1., fs_resample=2., **_NO_DROP)
    assert dt == 0.5
    np.testing.assert_allclose(out, np.arange(9) * 0.5)


def test_process_trace_same_fs_resample_is_noop():
    trace = np.arange(10.)
    out, _, dt = pp.process_trace(trace, trace_t0=0., trace_dt=0.5, fs_resample=2., **_NO_DROP)
    assert dt == 0.5
    np.testing.assert_array_equal(out, trace)


def test_process_trace_invalid_inputs():
    trace = np.arange(10.)
    with pytest.raises(AssertionError):
        pp.process_trace(trace, trace_t0=0., trace_dt=1., standardize=3)
    with pytest.raises(ValueError):
        pp.process_trace(trace, trace_t0=0., trace_dt=1., subtract_baseline=True)
    with pytest.raises(ValueError):
        pp.process_trace(trace, trace_t0=0., trace_dt=1., standardize=1)
    with pytest.raises(ValueError):
        pp.process_trace(trace, trace_t0=5., trace_dt=1., stim_start=4.)


# ---------------------------------------------------------------------------
# snippets / averages helpers
# ---------------------------------------------------------------------------

class _FakeTable:
    def __init__(self, row):
        self.row = row

    def __and__(self, key):
        return self

    def fetch1(self, *attrs):
        return tuple(self.row[a] for a in attrs)


def test_fetch_snippets_and_times():
    snippets = np.arange(8.).reshape(4, 2)
    table = _FakeTable(dict(snippets_t0=np.array([1., 5.]), snippets_dt=0.5, snippets=snippets,
                            triggertimes_snippets=np.array([[1., 5.]])))
    out_snippets, out_times, out_tt = fetch_snippets_and_times(table, dict(a=1))
    np.testing.assert_array_equal(out_snippets, snippets)
    np.testing.assert_allclose(out_times, [[1., 5.], [1.5, 5.5], [2., 6.], [2.5, 6.5]])
    np.testing.assert_array_equal(out_tt, [[1., 5.]])


def test_get_aligned_snippets_times_consistent():
    times = np.array([[1., 5.], [1.5, 5.5], [2., 6.]])
    np.testing.assert_allclose(get_aligned_snippets_times(times), [0., 0.5, 1.])


def test_get_aligned_snippets_times_inconsistent():
    times = np.array([[1., 5.], [1.5, 5.6], [2., 6.2]])
    with pytest.raises(ValueError):
        get_aligned_snippets_times(times)
    with pytest.warns(UserWarning):
        aligned = get_aligned_snippets_times(times, raise_error=False)
    np.testing.assert_allclose(aligned, [0., 0.55, 1.1])


def test_compute_upsampled_average_aligns_to_first_trigger():
    dt = 0.1
    tau = np.arange(20) * dt
    rep_starts = np.array([0., 2.])
    snippets_times = tau[:, None] + rep_starts[None, :]
    snippets = np.stack([3 * tau, 3 * tau + 1.], axis=1)
    triggertimes = np.stack([rep_starts, rep_starts + 1.])  # two triggers per rep; only the first is used

    average, average_times, resampled = compute_upsampled_average(
        snippets, snippets_times, triggertimes, f_resample=20)
    assert average_times.size == 40
    np.testing.assert_allclose(average_times, np.arange(40) / 20)
    expected = 3 * np.minimum(average_times, 1.9) + 0.5  # held constant after the last sample
    np.testing.assert_allclose(average, expected)
    assert resampled.shape == (40, 2)
    np.testing.assert_allclose(resampled[:, 1] - resampled[:, 0], 1.)


def test_compute_upsampled_average_with_offset_sampling():
    tau = np.arange(20) * 0.1
    rep_starts = np.array([0., 2., 4.])
    offsets = np.array([0., 0.03, 0.07])  # frames do not coincide with triggers
    snippets_times = tau[:, None] + (rep_starts + offsets)[None, :]
    snippets = 2. * (snippets_times - rep_starts[None, :])
    average, average_times, _ = compute_upsampled_average(snippets, snippets_times, rep_starts[None, :],
                                                          f_resample=10)
    valid = (average_times >= 0.07) & (average_times <= 1.9)
    np.testing.assert_allclose(average[valid], 2. * average_times[valid])
