import matplotlib
import numpy as np
import pytest

from djimaging.utils import filter_utils


def _sine(f, fs=100., dur=10.):
    t = np.arange(int(dur * fs)) / fs
    return t, np.sin(2 * np.pi * f * t)


# ---------------------------------------------------------------------------
# LowPassFilter / lowpass_filter_trace
# ---------------------------------------------------------------------------

def test_lowpass_filter_normal_cutoff_and_order():
    lpf = filter_utils.LowPassFilter(fs=100, cutoff=10, order=4)
    assert lpf.nyq == 50.
    assert lpf.normal_cutoff == pytest.approx(0.2)
    assert len(lpf.b) == len(lpf.a) == 5


def test_lowpass_filter_unknown_direction_raises():
    with pytest.raises(NotImplementedError):
        filter_utils.LowPassFilter(fs=100, cutoff=10, direction='x')


def test_lowpass_filter_cutoff_above_nyquist_raises():
    with pytest.raises(ValueError):
        filter_utils.LowPassFilter(fs=100, cutoff=60)


def test_lowpass_filter_causal_vs_zero_phase():
    impulse = np.zeros(201)
    impulse[100] = 1.
    y_causal = filter_utils.LowPassFilter(fs=100, cutoff=10, direction='l').filter_data(impulse)
    y_ff = filter_utils.LowPassFilter(fs=100, cutoff=10, direction='ff').filter_data(impulse)
    assert y_causal.shape == y_ff.shape == impulse.shape
    # Causal filter: no response before the impulse
    np.testing.assert_array_equal(y_causal[:100], 0.)
    # Zero-phase filter: symmetric response around the impulse
    assert np.abs(y_ff[99]) > 1e-3
    np.testing.assert_allclose(y_ff[90:100], y_ff[101:111][::-1], atol=1e-6)


def test_lowpass_filter_trace_passes_low_and_attenuates_high_frequencies():
    fs = 100.
    _, low = _sine(1., fs=fs)
    _, high = _sine(30., fs=fs)
    out_low = filter_utils.lowpass_filter_trace(low, fs=fs, f_cutoff=5.)
    out_high = filter_utils.lowpass_filter_trace(high, fs=fs, f_cutoff=5.)
    assert out_low.shape == low.shape
    # Zero phase: low frequency passes without delay (ignore edges)
    np.testing.assert_allclose(out_low[100:-100], low[100:-100], atol=1e-2)
    assert np.max(np.abs(out_high[100:-100])) < 1e-3


def test_lowpass_filter_trace_preserves_dc():
    trace = np.full(500, 3.5)
    np.testing.assert_allclose(filter_utils.lowpass_filter_trace(trace, fs=31.25, f_cutoff=2.), 3.5)


def test_lowpass_filter_plot_runs(monkeypatch):
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    monkeypatch.setattr(plt, 'show', lambda: None)
    filter_utils.LowPassFilter(fs=100, cutoff=5).plot()
    plt.close('all')


# ---------------------------------------------------------------------------
# resample_trace
# ---------------------------------------------------------------------------

def test_resample_trace_grid_and_values():
    tracetime = np.array([1., 2., 3., 4.])
    trace = 10 * tracetime
    t_new, trace_new = filter_utils.resample_trace(tracetime, trace, dt=0.5)
    np.testing.assert_allclose(t_new, [1., 1.5, 2., 2.5, 3., 3.5, 4.])
    np.testing.assert_allclose(trace_new, 10 * t_new)


def test_resample_trace_end_not_on_grid():
    tracetime = np.array([0., 1., 2., 3.])
    t_new, trace_new = filter_utils.resample_trace(tracetime, tracetime ** 2, dt=0.8)
    np.testing.assert_allclose(t_new, [0., 0.8, 1.6, 2.4])
    np.testing.assert_allclose(trace_new, [0., 0.8, 1. + 0.6 * 3, 4. + 0.4 * 5])


def test_resample_trace_irregular_input_to_regular_grid():
    tracetime = np.array([0., 0.1, 0.25, 0.3, 0.4])
    t_new, trace_new = filter_utils.resample_trace(tracetime, 2 * tracetime + 1, dt=0.1)
    np.testing.assert_allclose(np.diff(t_new), 0.1)
    np.testing.assert_allclose(trace_new, 2 * t_new + 1)


def test_resample_trace_same_dt_keeps_all_samples():
    tracetime = np.arange(481) * 0.002
    trace = np.arange(481, dtype=float)
    t_new, trace_new = filter_utils.resample_trace(tracetime, trace, dt=np.mean(np.diff(tracetime)))
    assert t_new.size == tracetime.size
    np.testing.assert_allclose(trace_new, trace)


# ---------------------------------------------------------------------------
# upsample_stim / upsample_trace / downsample_trace
# ---------------------------------------------------------------------------

def test_upsample_stim_1d():
    stimtime, stim = filter_utils.upsample_stim(np.array([0., 1., 2.]), np.array([10, 20, 30]), fupsample=2)
    np.testing.assert_allclose(stimtime, [0., 0.5, 1., 1.5, 2., 2.5])
    np.testing.assert_array_equal(stim, [10, 10, 20, 20, 30, 30])


def test_upsample_stim_nd_repeats_frames_along_first_axis():
    stim = np.arange(2 * 3 * 2).reshape(2, 3, 2)
    stimtime, stim_up = filter_utils.upsample_stim(np.array([5., 5.3]), stim, fupsample=3)
    np.testing.assert_allclose(stimtime, [5., 5.1, 5.2, 5.3, 5.4, 5.5])
    assert stim_up.shape == (6, 3, 2)
    for i in range(3):
        np.testing.assert_array_equal(stim_up[i], stim[0])
        np.testing.assert_array_equal(stim_up[3 + i], stim[1])


def test_upsample_trace_linear_interpolation():
    tracetime, trace = filter_utils.upsample_trace(np.array([0., 1., 2.]), np.array([0., 10., 20.]), fupsample=4)
    np.testing.assert_allclose(tracetime, np.arange(12) * 0.25)
    # Samples after the last original time are held constant (np.interp clamps)
    np.testing.assert_allclose(trace, [0., 2.5, 5., 7.5, 10., 12.5, 15., 17.5, 20., 20., 20., 20.])


def test_upsample_trace_accepts_integer_valued_float():
    tracetime, trace = filter_utils.upsample_trace(np.arange(4.), np.arange(4.), fupsample=2.0)
    assert tracetime.size == trace.size == 8


@pytest.mark.parametrize('fupsample', [1, 0, 2.5])
def test_upsample_trace_invalid_factor_raises(fupsample):
    with pytest.raises(AssertionError):
        filter_utils.upsample_trace(np.arange(4.), np.arange(4.), fupsample=fupsample)


def test_downsample_trace_1d_block_means_and_drops_remainder():
    tracetime, trace = filter_utils.downsample_trace(np.arange(10.) * 0.1, np.arange(10.), fdownsample=3)
    np.testing.assert_allclose(tracetime, [0., 0.3, 0.6])
    np.testing.assert_allclose(trace, [1., 4., 7.])


def test_downsample_trace_2d_last_axis_is_time():
    traces = np.vstack([np.arange(8.), -np.arange(8.)])
    tracetime, trace = filter_utils.downsample_trace(np.arange(8.), traces, fdownsample=4)
    np.testing.assert_allclose(tracetime, [0., 4.])
    np.testing.assert_allclose(trace, [[1.5, 5.5], [-1.5, -5.5]])


def test_downsample_trace_does_not_modify_input():
    trace = np.arange(6.)
    tracetime = np.arange(6.)
    filter_utils.downsample_trace(tracetime, trace, fdownsample=2)
    np.testing.assert_array_equal(trace, np.arange(6.))


@pytest.mark.parametrize('fdownsample', [1, 2.0])
def test_downsample_trace_invalid_factor_raises(fdownsample):
    with pytest.raises(AssertionError):
        filter_utils.downsample_trace(np.arange(6.), np.arange(6.), fdownsample=fdownsample)
