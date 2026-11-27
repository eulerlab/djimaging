import numpy as np
import pytest

from djimaging.utils import trace_utils


def generate_random_trace_and_time(size=100, tmin=0, tmax=12):
    trace = np.random.uniform(0, 1, size)
    tracetime = np.linspace(tmin, tmax, size, endpoint=True)
    return trace, tracetime


def generate_random_stimulus_and_time(shape=(100, 4, 3), tmin=0, tmax=12):
    stim = np.random.randint(0, 2, shape)
    stimtime = np.linspace(tmin, tmax, shape[0], endpoint=True)
    return stim, stimtime


def test_align_trace_to_stim():
    # Stimulus frames (0.2 s) longer than the trace sampling interval (~0.1 s), within the trace
    _, stimtime = generate_random_stimulus_and_time(shape=(30, 3, 4), tmin=2, tmax=7.8)
    trace, tracetime = generate_random_trace_and_time(size=101, tmin=0, tmax=10)
    aligned_trace, dt, t0, dt_rel_error = trace_utils.align_trace_to_stim(
        stimtime=stimtime, trace=trace, tracetime=tracetime)
    assert aligned_trace.shape == stimtime.shape
    assert np.all(np.isfinite(aligned_trace))
    assert dt == pytest.approx(0.2)
    assert t0 == 2.
    expected = [np.mean(trace[(tracetime >= t_a) & (tracetime < t_a + dt)]) for t_a in stimtime]
    np.testing.assert_allclose(aligned_trace, expected)


def test_align_trace_to_stim_stim_faster_than_trace_raises():
    _, stimtime = generate_random_stimulus_and_time(shape=(100, 3, 4), tmin=2, tmax=8)  # dt ~0.06 s
    trace, tracetime = generate_random_trace_and_time(size=100, tmin=0, tmax=10)  # dt ~0.1 s
    with pytest.raises(ValueError, match='no trace sample'):
        trace_utils.align_trace_to_stim(stimtime=stimtime, trace=trace, tracetime=tracetime)


def test_align_trace_to_stim_frames_outside_trace_are_nan():
    stimtime = np.arange(-1., 4.)  # frames [-1, 0) and [3, 4) are not covered by the trace
    tracetime = np.arange(0, 12) * 0.25  # 0 ... 2.75 s
    trace = np.arange(12.)
    aligned_trace, _, _, _ = trace_utils.align_trace_to_stim(stimtime, trace, tracetime)
    np.testing.assert_allclose(aligned_trace, [np.nan, 1.5, 5.5, 9.5, np.nan])


def test_find_closest():
    obs = trace_utils.find_closest(target=0.6, data=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
    assert obs == 1


def test_find_closest2():
    obs = trace_utils.find_closest(target=12, data=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
    assert obs == 10


def test_find_closest_as_index():
    data = np.array([0., 1., 2., 3., 4.])
    idx = trace_utils.find_closest(target=2.4, data=data, as_index=True)
    assert idx == 2


def test_find_closest_atol_raises():
    data = np.array([0., 1., 2., 3., 4.])
    with pytest.raises(ValueError):
        trace_utils.find_closest(target=10., data=data, atol=1.)


def test_find_closest_after_basic():
    data = np.array([0., 1., 2., 3., 4.])
    obs = trace_utils.find_closest_after(target=2.1, data=data)
    assert obs == 3.


def test_find_closest_after_exact_match():
    data = np.array([0., 1., 2., 3., 4.])
    obs = trace_utils.find_closest_after(target=2., data=data)
    assert obs == 2.


def test_find_closest_after_as_index():
    data = np.array([0., 1., 2., 3., 4.])
    idx = trace_utils.find_closest_after(target=2.1, data=data, as_index=True)
    assert idx == 3


def test_find_closest_before_basic():
    data = np.array([0., 1., 2., 3., 4.])
    obs = trace_utils.find_closest_before(target=2.9, data=data)
    assert obs == 2.


def test_find_closest_before_exact_match():
    data = np.array([0., 1., 2., 3., 4.])
    obs = trace_utils.find_closest_before(target=3., data=data)
    assert obs == 3.


def test_find_closest_before_as_index():
    data = np.array([0., 1., 2., 3., 4.])
    idx = trace_utils.find_closest_before(target=2.9, data=data, as_index=True)
    assert idx == 2


def test_get_mean_dt_regular():
    tracetime = np.linspace(0, 10, 101)
    dt, dt_rel_error = trace_utils.get_mean_dt(tracetime)
    assert np.isclose(dt, 0.1, rtol=1e-3)
    assert dt_rel_error < 0.01


def test_get_mean_dt_inconsistent_raises():
    tracetime = np.array([0., 1., 2., 10., 11.])  # big jump
    with pytest.raises(ValueError):
        trace_utils.get_mean_dt(tracetime, rtol_error=1.0)


def test_argsort_traces_basic():
    traces = np.array([[1., 2., 3.], [3., 2., 1.], [1., 2., 3.]])
    idxs = trace_utils.argsort_traces(traces)
    assert len(idxs) == 3
    assert set(idxs) == {0, 1, 2}


def test_argsort_traces_few_samples():
    """With <=2 samples argsort_traces returns identity permutation."""
    traces = np.array([[1., 2., 3.], [3., 2., 1.]])
    idxs = trace_utils.argsort_traces(traces)
    assert np.array_equal(idxs, np.arange(2))


def test_sort_traces_shape():
    traces = np.array([[1., 2., 3.], [3., 2., 1.], [0., 1., 0.]])
    sorted_traces = trace_utils.sort_traces(traces)
    assert sorted_traces.shape == traces.shape


def test_sort_traces_ignore_nan():
    traces = np.array([[1., 2., np.nan], [3., 2., np.nan], [0., 1., np.nan]])
    sorted_traces = trace_utils.sort_traces(traces, ignore_nan=True)
    assert sorted_traces.shape == traces.shape
