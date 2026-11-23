"""Tests for the contrast step (CSL) helpers: Naka-Rushton / sigmoid fits, F1 amplitude and analyse_csl_response."""
import matplotlib

matplotlib.use('Agg')

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402

from djimaging.tables.response.csl.csl import analyse_csl_response, f1_amp_phase  # noqa: E402
from djimaging.tables.response.csl.naka_rushton_utils import (  # noqa: E402
    fit_naka_rushton, fit_naka_rushton_repeated, init_naka_rushton, naka_rushton)
from djimaging.tables.response.csl.sigmoid_utils import (  # noqa: E402
    fit_sigmoid, fit_sigmoid_with_retry, init_sigmoid_params, sigmoid)

CONTRASTS = np.array([0., 0.1, 0.2, 0.4, 0.6, 0.8, 1.])


def _nr_reference(S, R_base, R_max, S50, n):
    S = np.asarray(S, dtype=float)
    return R_max * S ** n / (S ** n + S50 ** n) + R_base


# ---------------------------------------------------------------- naka_rushton

def test_naka_rushton_matches_formula():
    S = np.array([0., 0.1, 0.35, 0.7, 1.])
    np.testing.assert_allclose(naka_rushton(S, 0.05, 0.8, 0.35, 2.), _nr_reference(S, 0.05, 0.8, 0.35, 2.))


def test_naka_rushton_half_max_at_s50():
    out = naka_rushton(np.array([0.4]), 0.2, 1.0, 0.4, 3.)
    np.testing.assert_allclose(out, 0.2 + 0.5)


def test_naka_rushton_s50_zero_is_constant():
    out = naka_rushton(np.array([0., 0.5, 1.]), 0.1, 1.0, 0.0, 2.)
    np.testing.assert_allclose(out, 0.6)


def test_naka_rushton_stimulus_equal_to_r_base():
    S = np.array([0., 0.1, 0.5, 1.])
    np.testing.assert_allclose(naka_rushton(S, 0.5, 1.0, 0.3, 2.), _nr_reference(S, 0.5, 1.0, 0.3, 2.))


@pytest.mark.xfail(strict=True, reason="BUG: naka_rushton allocates the output with zeros_like(S), truncating for int S")
def test_naka_rushton_integer_stimulus():
    S = np.array([0, 1, 2])
    np.testing.assert_allclose(naka_rushton(S, 0., 1.0, 1.0, 1.), [0., 0.5, 2 / 3])


# ---------------------------------------------------------------- init / fit

def test_init_naka_rushton_within_bounds():
    y = _nr_reference(CONTRASTS, 0.05, 0.9, 0.35, 2.5)
    p0, (lb, ub) = init_naka_rushton(CONTRASTS, y)
    assert len(p0) == 4
    for p, lo, hi in zip(p0, lb, ub):
        assert lo <= p <= hi
    np.random.seed(0)
    p0_noise, _ = init_naka_rushton(CONTRASTS, y, noise=True)
    for p, lo, hi in zip(p0_noise, lb, ub):
        assert lo <= p <= hi


def test_fit_naka_rushton_repeated_recovers_params():
    true = (0.05, 0.9, 0.35, 2.5)
    y = _nr_reference(CONTRASTS, *true)
    popt = fit_naka_rushton_repeated(CONTRASTS, y, n=3)
    np.testing.assert_allclose(naka_rushton(CONTRASTS, *popt), y, atol=1e-4)


def test_fit_naka_rushton_half_max_point():
    true = (0.05, 0.9, 0.35, 2.5)
    y = _nr_reference(CONTRASTS, *true)
    half_max, half_max_stim, slope = fit_naka_rushton(CONTRASTS, y)

    y_min, y_max = y.min(), y.max()
    assert half_max == pytest.approx((y_min + y_max) / 2, abs=1e-3)
    # The fitted curve passes through (half_max_stim, half_max)
    assert _nr_reference([half_max_stim], *true)[0] == pytest.approx(half_max, abs=1e-3)
    # Slope equals the numerical derivative of the true curve
    eps = 1e-5
    num_slope = (_nr_reference([half_max_stim + eps], *true) - _nr_reference([half_max_stim - eps], *true))[0] / (2 * eps)
    assert slope == pytest.approx(num_slope, rel=1e-2)


def test_fit_naka_rushton_uses_linear_fit_for_linear_data():
    y = 0.5 * CONTRASTS + 0.1
    half_max, half_max_stim, slope = fit_naka_rushton(CONTRASTS, y)
    assert np.isnan(half_max_stim)
    assert slope == pytest.approx(0.5, rel=1e-6)
    assert half_max == pytest.approx(0.35, rel=1e-6)


def test_fit_naka_rushton_plot():
    y = _nr_reference(CONTRASTS, 0.05, 0.9, 0.35, 2.5)
    fig, ax = plt.subplots()
    fit_naka_rushton(CONTRASTS, y, ax=ax)
    assert len(ax.lines) > 0
    plt.close(fig)


@pytest.mark.xfail(strict=True, reason="BUG: init_naka_rushton gives lower == upper bounds for flat data; curve_fit "
                                       "raises ValueError, which is not caught (only RuntimeError)")
def test_fit_naka_rushton_flat_response_does_not_crash():
    half_max, half_max_stim, slope = fit_naka_rushton(CONTRASTS, np.zeros_like(CONTRASTS))
    assert slope == pytest.approx(0.)


# ---------------------------------------------------------------- sigmoid

def test_sigmoid_values():
    assert sigmoid(3., 3., 2., 4.) == pytest.approx(2.)
    assert sigmoid(50., 3., 2., 4.) == pytest.approx(4.)
    assert sigmoid(-50., 3., 2., 4.) == pytest.approx(0.)


def test_init_sigmoid_params_within_bounds():
    x = np.arange(20.)
    y = sigmoid(x, 8., 0.8, 2.)
    p0, (lb, ub) = init_sigmoid_params(x, y)
    for p, lo, hi in zip(p0, lb, ub):
        assert lo <= p <= hi


def test_fit_sigmoid_with_retry_recovers_params():
    x = np.arange(20.)
    y = sigmoid(x, 8., 0.8, 2.)
    np.testing.assert_allclose(fit_sigmoid_with_retry(x, y), [8., 0.8, 2.], rtol=1e-3)


@pytest.mark.parametrize('sign', [1, -1])
def test_fit_sigmoid_half_amplitude_and_slope(sign):
    x = np.arange(20.)
    y = sign * sigmoid(x, 8., 0.8, 2.)
    half_amp, half_amp_x, slope = fit_sigmoid(y, x, sign=sign)
    assert half_amp == pytest.approx(sign * 1.0, rel=1e-3)
    assert half_amp_x == pytest.approx(8., rel=1e-3)
    # derivative of a / (1 + exp(-k (x - x0))) at x0 is a * k / 4
    assert slope == pytest.approx(sign * 2. * 0.8 / 4, rel=1e-3)


def test_fit_sigmoid_default_x_and_plot():
    y = sigmoid(np.arange(20.), 8., 0.8, 2.)
    fig, ax = plt.subplots()
    half_amp, half_amp_x, slope = fit_sigmoid(y, ax=ax)
    assert half_amp_x == pytest.approx(8., rel=1e-3)
    plt.close(fig)


# ---------------------------------------------------------------- f1_amp_phase

def test_f1_amp_phase_cosine_and_sine():
    fs, f = 99., 6.
    t = np.arange(495) / fs  # odd length, exactly 30 cycles
    amp, phase = f1_amp_phase(2. * np.cos(2 * np.pi * f * t), f, fs)
    assert amp == pytest.approx(2., rel=1e-6)
    assert phase == pytest.approx(0., abs=1e-6)

    amp, phase = f1_amp_phase(0.5 * np.sin(2 * np.pi * f * t), f, fs)
    assert amp == pytest.approx(0.5, rel=1e-6)
    assert phase == pytest.approx(-90., abs=1e-6)


def test_f1_amp_phase_ignores_offset():
    fs, f = 99., 6.
    t = np.arange(495) / fs
    amp, _ = f1_amp_phase(3. + np.cos(2 * np.pi * f * t), f, fs)
    assert amp == pytest.approx(1., rel=1e-6)


def test_f1_amp_phase_edge_cases():
    assert f1_amp_phase(np.array([1., 2.]), 6., 100.) == (0.0, 0.0)
    assert f1_amp_phase(np.ones(101), 0., 100.) == (0.0, 0.0)


@pytest.mark.xfail(strict=True, reason="BUG: f1_amp_phase forces odd N, so bin N//2 is never Nyquist, "
                                       "but it still skips the factor 2 for k == N//2")
def test_f1_amp_phase_highest_bin_of_odd_length():
    n = 11
    x = np.cos(2 * np.pi * 5 * np.arange(n) / n)  # bin 5 == N//2 of an odd-length signal
    amp, _ = f1_amp_phase(x, stim_frequency=5., trace_frequency=float(n))
    assert amp == pytest.approx(1., rel=1e-6)


# ---------------------------------------------------------------- analyse_csl_response

CSL_LEVELS = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0])


def _make_csl_trace(n_reps=3, fs=50., on_amp=1.0, off_amp=0.2, noise=0.0, seed=0):
    """Synthetic contrast step trace: 8 triggers per rep, 8 s apart; on step at trigger 0, off step at trigger 7,
    6 Hz flicker with increasing contrast from 3 s after trigger k to trigger k + 1 (k = 0..5)."""
    rng = np.random.default_rng(seed)
    step_t, rep_len = 8., 64.
    t = np.arange(int((n_reps * rep_len + 10) * fs)) / fs
    trace = np.zeros(t.size)
    triggertimes = []
    for r in range(n_reps):
        tts = 5. + r * rep_len + np.arange(8) * step_t
        triggertimes.extend(tts)
        m = t >= tts[0]
        trace[m] += on_amp * np.exp(-(t[m] - tts[0]) / 0.3)
        m = t >= tts[7]
        trace[m] += off_amp * np.exp(-(t[m] - tts[7]) / 0.3)
        for k, c in enumerate(CSL_LEVELS):
            m = (t >= tts[k] + 3) & (t < tts[k + 1])
            amp = _nr_reference([c], 0., 1., 0.4, 2.)[0]
            trace[m] += amp * 0.5 * (1 + np.sin(2 * np.pi * 6 * t[m]))
    trace += noise * rng.normal(size=t.size)
    return trace, 0., 1 / fs, np.array(triggertimes)


@pytest.mark.parametrize('fit_kind,metric_kind', [('naka_rushton', 'auc'), ('sigmoid', 'fft_f1')])
def test_analyse_csl_response_synthetic(fit_kind, metric_kind):
    trace, t0, dt, tt = _make_csl_trace()
    res = analyse_csl_response(trace, t0, dt, tt, ntrigger_rep=8, fs_resample=50.,
                               fit_kind=fit_kind, metric_kind=metric_kind, stim_frequency=6.)
    assert not res['droppedlastrep_flag']
    assert res['bc_snippets'].shape[1] == 3
    # Identical repetitions
    assert res['qidx_full'] == pytest.approx(1., abs=0.01)
    assert res['qidx_contrast'] == pytest.approx(1., abs=0.01)
    # On step response dominates the off step response
    assert res['on_off_index'] > 0
    # Zero contrast first (w_zero_fit), then increasing responses with contrast
    assert res['contrast_metrics'].size == 7
    assert np.all(np.diff(res['contrast_metrics'][1:]) > 0)
    assert np.all(np.isfinite(res['fit'][0::2]))
    if metric_kind == 'fft_f1':
        assert res['contrast_phases_deg'].shape == (7,)
    else:
        assert 'contrast_phases_deg' not in res


def test_analyse_csl_response_off_dominated():
    trace, t0, dt, tt = _make_csl_trace(on_amp=0.2, off_amp=1.0)
    res = analyse_csl_response(trace, t0, dt, tt, ntrigger_rep=8, fs_resample=50.)
    assert res['on_off_index'] < 0


def test_analyse_csl_response_noise_lowers_quality():
    trace, t0, dt, tt = _make_csl_trace(on_amp=0., off_amp=0., noise=1.)
    res = analyse_csl_response(trace, t0, dt, tt, ntrigger_rep=8, fs_resample=50.)
    assert res['qidx_full'] < 0.6


@pytest.mark.parametrize('kw', [dict(fit_kind='unknown'), dict(metric_kind='unknown')])
def test_analyse_csl_response_unknown_kind(kw):
    trace, t0, dt, tt = _make_csl_trace(n_reps=2)
    with pytest.raises(ValueError):
        analyse_csl_response(trace, t0, dt, tt, ntrigger_rep=8, fs_resample=50., **kw)


def test_analyse_csl_response_plot(monkeypatch):
    monkeypatch.setattr(plt, 'show', lambda *a, **k: None)
    trace, t0, dt, tt = _make_csl_trace(n_reps=2)
    analyse_csl_response(trace, t0, dt, tt, ntrigger_rep=8, fs_resample=50., plot=True)
    plt.close('all')


# ---------------------------------------------------------------- CslMetrics table methods without database

def _csl_table(trace, t0, dt, triggertimes, metric_kind='auc', fit_kind='naka_rushton', nypix=5):
    from djimaging.tables.response.csl.csl import CslMetricsTemplate
    from tests.tables._fake_dj import FakeRel, make_fake

    attrs = {k: v for k, v in vars(CslMetricsTemplate).items()
             if k.startswith('_') and not k.startswith('__') and isinstance(v, (int, float, tuple, str))}
    attrs.update(_metric_kind=metric_kind, _fit_kind=fit_kind, _stim_frequency=6.)
    presentation = FakeRel(dict(scan_type='xy', nypix=nypix, nzpix=0, triggertimes=triggertimes),
                           ScanInfo=FakeRel(dict(line_duration=1 / 50.)))
    return make_fake(CslMetricsTemplate, ['make', '_make_fetch_and_compute'],
                     traces_table=FakeRel(dict(trace=trace, trace_t0=t0, trace_dt=dt)),
                     presentation_table=presentation, stimulus_table=FakeRel(dict(ntrigger_rep=8)), **attrs)


@pytest.mark.parametrize('metric_kind', ['auc', 'fft_f1'])
def test_csl_metrics_make(metric_kind):
    trace, t0, dt, tt = _make_csl_trace()
    table = _csl_table(trace, t0, dt, tt, metric_kind=metric_kind)
    table.make(dict(roi_id=1))
    entry = table.inserted[0]
    assert entry['fs_metrics'] == pytest.approx(50.)
    assert entry['fs'] == pytest.approx(10.)  # downsampled by the number of lines
    assert entry['average'].size == int(np.ceil(64 * 50 / 5))
    assert entry['snippets'].shape[1] == 3
    assert entry[f'contrast_{metric_kind}s'].shape == (7,)
    assert ('contrast_fft_f1_phases' in entry) == (metric_kind == 'fft_f1')
    assert entry['qidx_full'] == pytest.approx(1., abs=0.01)


def test_csl_metrics_make_empty_trace_and_debug():
    table = _csl_table(np.array([]), 0., 0.02, np.arange(8.))
    table.make(dict(roi_id=1))
    assert table.inserted == []
    with pytest.raises(ValueError, match='empty'):
        table._make_fetch_and_compute(dict(roi_id=1))

    trace, t0, dt, tt = _make_csl_trace(n_reps=2)
    table = _csl_table(trace, t0, dt, tt)
    table.make(dict(roi_id=1), verbose=True, DEBUG=True)
    assert table.inserted == []
