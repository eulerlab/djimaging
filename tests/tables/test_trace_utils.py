import numpy as np

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
    stim, stimtime = generate_random_stimulus_and_time(shape=(100, 3, 4), tmin=2, tmax=8)
    trace, tracetime = generate_random_trace_and_time(size=100, tmin=0, tmax=10)
    aligned_trace, dt, t0, dt_rel_error = trace_utils.align_trace_to_stim(
        stimtime=stimtime, trace=trace, tracetime=tracetime)
    aligned_stim = stim
    assert stim.shape == aligned_stim.shape
    assert np.all(stim == aligned_stim)


def test_find_closest():
    obs = trace_utils.find_closest(target=0.6, data=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
    assert obs == 1


def test_find_closest2():
    obs = trace_utils.find_closest(target=12, data=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
    assert obs == 10
