"""Additional tests for branches of djimaging.utils.trace_utils not covered in tests/tables/test_trace_utils.py."""
import numpy as np
import pytest

from djimaging.utils import trace_utils


def test_get_mean_dt_warns_between_warning_and_error_tolerance():
    tracetime = np.array([0., 1., 2., 3.5, 4.5])  # dts 1, 1, 1.5, 1 -> mean 1.125, rel. error 1/3
    with pytest.warns(UserWarning, match='Inconsistent dts'):
        dt, dt_rel_error = trace_utils.get_mean_dt(tracetime, rtol_error=1., rtol_warning=0.3)
    assert dt == pytest.approx(1.125)
    assert dt_rel_error == pytest.approx(0.375 / 1.125)


def test_find_closest_after_atol_raises():
    with pytest.raises(ValueError):
        trace_utils.find_closest_after(target=1.2, data=np.array([0., 1., 2.]), atol=0.5)


def test_find_closest_before_atol_raises():
    with pytest.raises(ValueError):
        trace_utils.find_closest_before(target=1.8, data=np.array([0., 1., 2.]), atol=0.5)


def test_find_closest_tie_returns_first():
    assert trace_utils.find_closest(target=1.5, data=np.array([0., 1., 2.]), as_index=True) == 1


def test_align_stim_to_trace_hand_computed():
    stim = np.array([1, 2, 3, 4])
    stimtime = np.array([0., 1., 2., 3.])
    tracetime = np.arange(-4, 21) * 0.25  # -1 ... 5 s
    trace = tracetime * 10.
    aligned_stim, aligned_trace, dt_trace, t0, dt_rel_error = trace_utils.align_stim_to_trace(
        stim=stim, stimtime=stimtime, trace=trace, tracetime=tracetime, stim_lag_atol=0.003)
    # Trace samples in [-0.003, 3.997) are kept: 0, 0.25, ..., 3.75
    assert dt_trace == pytest.approx(0.25)
    assert t0 == pytest.approx(0.)
    assert dt_rel_error == pytest.approx(0., abs=1e-9)
    np.testing.assert_allclose(aligned_trace, np.arange(16) * 2.5)
    np.testing.assert_array_equal(aligned_stim, np.repeat(stim, 4))


def test_align_stim_to_trace_stim_lag_assigns_early_samples_to_next_frame():
    stim = np.array([1, 2])
    stimtime = np.array([0., 1.])
    tracetime = np.arange(-2, 10) * 0.5 - 0.001  # samples 1 ms before each half second
    trace = np.arange(tracetime.size, dtype=float)

    aligned_stim, aligned_trace, _, t0, _ = trace_utils.align_stim_to_trace(
        stim=stim, stimtime=stimtime, trace=trace, tracetime=tracetime, stim_lag_atol=0.003)
    # Samples at -0.001 and 0.999 are within 3 ms before a frame onset and belong to that frame
    assert t0 == pytest.approx(-0.001)
    np.testing.assert_array_equal(aligned_stim, [1, 1, 2, 2])
    np.testing.assert_allclose(aligned_trace, [2., 3., 4., 5.])

    aligned_stim, aligned_trace, _, t0, _ = trace_utils.align_stim_to_trace(
        stim=stim, stimtime=stimtime, trace=trace, tracetime=tracetime, stim_lag_atol=0.)
    assert t0 == pytest.approx(0.499)
    np.testing.assert_array_equal(aligned_stim, [1, 1, 2, 2])
    np.testing.assert_allclose(aligned_trace, [3., 4., 5., 6.])


def test_align_stim_to_trace_nd_stim():
    stim = np.arange(3 * 2 * 2).reshape(3, 2, 2)
    stimtime = np.array([0., 1., 2.])
    tracetime = np.arange(0, 12) * 0.5
    aligned_stim, aligned_trace, _, _, _ = trace_utils.align_stim_to_trace(
        stim=stim, stimtime=stimtime, trace=np.ones_like(tracetime), tracetime=tracetime)
    assert aligned_stim.shape == (6, 2, 2)
    np.testing.assert_array_equal(aligned_stim[2], stim[1])


@pytest.mark.parametrize('stim_lag_atol', [-0.001, 1.0, 2.0])
def test_align_stim_to_trace_invalid_lag_raises(stim_lag_atol):
    with pytest.raises(ValueError):
        trace_utils.align_stim_to_trace(
            stim=np.arange(4), stimtime=np.arange(4.), trace=np.arange(20.), tracetime=np.arange(20.) * 0.25,
            stim_lag_atol=stim_lag_atol)


def test_align_trace_to_stim_bin_means():
    stimtime = np.array([0., 1., 2.])
    tracetime = np.arange(0, 12) * 0.25  # 4 samples per stimulus frame
    trace = np.arange(12.)
    aligned_trace, dt, t0, _ = trace_utils.align_trace_to_stim(stimtime, trace, tracetime)
    np.testing.assert_allclose(aligned_trace, [1.5, 5.5, 9.5])
    assert dt == pytest.approx(1.)
    assert t0 == 0.
