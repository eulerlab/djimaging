import warnings

import numpy as np
import pytest

from djimaging.utils.receptive_fields.preprocess_rf_utils import (
    finalize_stim, finalize_trace, prepare_noise_data, preprocess_stimulus, preprocess_trace, transform_trace)


# --- stimulus ---

def test_preprocess_stimulus_one_frame_per_trigger():
    stim = np.arange(5 * 2 * 3).reshape(5, 2, 3)
    triggertimes = np.array([1., 2., 3., 4., 5.])
    stimtime, stim_out = preprocess_stimulus(stim, triggertimes)
    np.testing.assert_array_equal(stimtime, triggertimes)
    np.testing.assert_array_equal(stim_out, stim)


def test_preprocess_stimulus_several_frames_per_trigger():
    stim = np.arange(6)[:, None, None] * np.ones((1, 2, 2))
    triggertimes = np.array([0., 1., 2.])
    stimtime, stim_out = preprocess_stimulus(stim, triggertimes, nframes_per_trigger=2)
    np.testing.assert_allclose(stimtime, [0., 0.5, 1., 1.5, 2., 2.5])
    np.testing.assert_array_equal(stim_out, stim)


def test_preprocess_stimulus_too_many_triggers():
    with pytest.raises(ValueError):
        preprocess_stimulus(np.zeros((3, 2, 2)), np.arange(4.))


def test_preprocess_stimulus_too_few_triggers_truncates():
    stim = np.arange(10)[:, None, None] * np.ones((1, 2, 2))
    with pytest.warns(UserWarning, match='Less triggertimes'):
        stimtime, stim_out = preprocess_stimulus(stim, np.arange(6.))
    assert stim_out.shape == (6, 2, 2)
    np.testing.assert_array_equal(stim_out, stim[:6])
    assert not np.shares_memory(stim_out, stim)


def test_preprocess_stimulus_upsampling():
    stim = np.arange(3)[:, None] * np.ones((1, 4))
    stimtime, stim_out = preprocess_stimulus(stim, np.array([0., 1., 2.]), fupsample_stim=2)
    np.testing.assert_allclose(stimtime, [0., 0.5, 1., 1.5, 2., 2.5])
    np.testing.assert_array_equal(stim_out[:, 0], [0, 0, 1, 1, 2, 2])


def test_finalize_stim_binary_to_plus_minus_one():
    stim_bool = np.array([[True, False], [False, True]])
    out = finalize_stim(stim_bool)
    np.testing.assert_array_equal(out, [[1, -1], [-1, 1]])
    assert out.dtype == np.int8

    stim_01 = np.array([[1., 0.], [0., 1.]])
    np.testing.assert_array_equal(finalize_stim(stim_01), [[1, -1], [-1, 1]])

    stim_pm = np.array([[1, -1], [-1, 1]])
    np.testing.assert_array_equal(finalize_stim(stim_pm), stim_pm)


def test_finalize_stim_continuous_is_zscored():
    rng = np.random.default_rng(0)
    stim = rng.uniform(0, 255, size=(100, 3, 4))
    out = finalize_stim(stim)
    assert out.shape == stim.shape
    assert np.mean(out) == pytest.approx(0., abs=1e-10)
    assert np.std(out) == pytest.approx(1.)


def test_finalize_stim_1d_indexes_unchanged():
    stim = np.array([0, 1, 1, 0, 1])
    np.testing.assert_array_equal(finalize_stim(stim), stim)


# --- trace ---

def test_transform_trace_kinds():
    tracetime = np.arange(10) * 0.1
    trace = np.array([0., 0., 1., 1., 0., 0., 3., 3., 3., 0.])

    t, x = transform_trace(tracetime, trace, kind='trace')
    np.testing.assert_array_equal(x, trace)

    t, x = transform_trace(tracetime, trace, kind='gradient')
    np.testing.assert_array_equal(x, [0., 0., 1., 0., 0., 0., 3., 0., 0., 0.])

    t, x = transform_trace(tracetime, trace, kind='events')
    # Only positive steps are events, with their height
    np.testing.assert_array_equal(np.nonzero(x)[0], [2, 6])
    np.testing.assert_allclose(x[[2, 6]], [1., 3.])

    with pytest.raises(NotImplementedError):
        transform_trace(tracetime, trace, kind='spikes')


@pytest.mark.parametrize('kind', ['trace', 'gradient', 'events'])
def test_transform_trace_upsampling(kind):
    tracetime = np.arange(20) * 0.1
    trace = np.sin(tracetime * 3)
    t, x = transform_trace(tracetime, trace, kind=kind, fupsample=3, dt=0.1)
    assert t.size == x.size
    assert t.size == pytest.approx(3 * tracetime.size, abs=3)
    np.testing.assert_allclose(np.diff(t), 0.1 / 3, rtol=1e-6)
    if kind != 'trace':
        assert np.all(x >= 0)


def test_transform_trace_upsampling_requires_dt():
    with pytest.raises(AssertionError):
        transform_trace(np.arange(5.), np.arange(5.), fupsample=2)


def test_preprocess_trace_regular():
    tracetime = np.arange(200) * 0.01
    trace = np.sin(2 * np.pi * tracetime)
    t, x = preprocess_trace(tracetime, trace)
    np.testing.assert_array_equal(t, tracetime)
    np.testing.assert_array_equal(x, trace)


def test_preprocess_trace_resamples_irregular_steps():
    rng = np.random.default_rng(0)
    tracetime = np.cumsum(rng.uniform(0.05, 0.15, size=100))
    trace = np.sin(tracetime)
    with pytest.warns(UserWarning, match='resample'):
        t, x = preprocess_trace(tracetime, trace)
    np.testing.assert_allclose(np.diff(t), np.mean(np.diff(tracetime)), rtol=1e-6)
    np.testing.assert_allclose(x, np.sin(t), atol=0.01)


def test_preprocess_trace_lowpass_and_blur_remove_high_frequencies():
    tracetime = np.arange(1000) * 0.01  # 100 Hz
    slow = np.sin(2 * np.pi * 0.5 * tracetime)
    trace = slow + np.sin(2 * np.pi * 30 * tracetime)
    _, x_lp = preprocess_trace(tracetime, trace, lowpass_cutoff=5.)
    _, x_blur = preprocess_trace(tracetime, trace, pre_blur_sigma_s=0.05)
    inner = slice(100, -100)
    for x in (x_lp, x_blur):
        assert np.max(np.abs(x[inner] - slow[inner])) < 0.1


def test_finalize_trace_scales_to_unit_std_and_clips():
    rng = np.random.default_rng(0)
    trace = 3. * rng.normal(size=10_000)
    out = finalize_trace(trace, dt=0.1, post_blur_sigma_s=0)
    assert np.std(out) == pytest.approx(1., rel=0.01)

    trace = np.zeros(1000)
    trace[0] = 1.
    out = finalize_trace(trace, dt=0.1, post_blur_sigma_s=0)
    assert np.max(out) == 7.


def test_finalize_trace_post_blur():
    trace = np.zeros(101)
    trace[50] = 1.
    out = finalize_trace(trace, dt=0.1, post_blur_sigma_s=0.2)
    assert np.argmax(out) == 50
    assert out[48] > 0 and out[52] > 0


# --- prepare_noise_data ---

def _noise_data(seed=0):
    rng = np.random.default_rng(seed)
    triggertimes = np.arange(100) * 0.2 + 1.
    stim = rng.choice([0, 1], size=(100, 3, 4))
    tracetime = np.arange(0.5, 22., 0.05)
    trace = rng.normal(size=tracetime.size)
    return stim, triggertimes, trace, tracetime


@pytest.mark.parametrize('ref_time', ['trace', 'stim'])
def test_prepare_noise_data_shapes(ref_time):
    stim, triggertimes, trace, tracetime = _noise_data()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        stim_out, trace_out, dt, t0, dt_rel_error = prepare_noise_data(
            stim, triggertimes, trace, tracetime, ref_time=ref_time)
    assert stim_out.shape[0] == trace_out.shape[0]
    assert stim_out.shape[1:] == (3, 4)
    assert set(np.unique(stim_out)) <= {-1, 1}
    assert np.std(trace_out) == pytest.approx(1., rel=0.05)
    if ref_time == 'stim':
        assert dt == pytest.approx(0.2)
    else:
        assert dt == pytest.approx(0.05)


def test_prepare_noise_data_unknown_ref_time():
    stim, triggertimes, trace, tracetime = _noise_data()
    with pytest.raises(NotImplementedError):
        prepare_noise_data(stim, triggertimes, trace, tracetime, ref_time='other')
