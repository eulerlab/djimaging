"""Tests for the bipolar cell chirp features (Franke et al. 2017) on synthetic traces with known ground truth."""
import warnings

import matplotlib

matplotlib.use('Agg')

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402

from djimaging.tables.response.chirp.chirp_features_bc import (  # noqa: E402
    ChirpFeaturesBcTemplate, compute_contrast_response_ratio, compute_freq_response, compute_high_frequency_index,
    compute_peak_to_post_peak_ratio, compute_plateau_index, compute_polarity_index, compute_tonic_release_index,
    compute_transience_index)
from tests.tables._fake_dj import FakeRel, make_fake  # noqa: E402

FS = 500.
T = np.arange(int(32 * FS)) / FS


def _box(t_a, t_b):
    return ((T >= t_a) & (T < t_b)).astype(float)


# ---------------------------------------------------------------- polarity

def test_polarity_index_on_off_none():
    assert compute_polarity_index(_box(2, 5), FS) == pytest.approx(1.)
    assert compute_polarity_index(_box(5, 8), FS) == pytest.approx(-1.)
    assert compute_polarity_index(np.zeros_like(T), FS) == 0.
    assert compute_polarity_index(_box(2, 5) + _box(5, 8), FS) == pytest.approx(0.)


def test_polarity_index_clips_negative_responses():
    # An On response with suppression during light off is still a pure On response
    assert compute_polarity_index(_box(2, 5) - _box(5, 8), FS) == pytest.approx(1.)


# ---------------------------------------------------------------- transience / plateau

def test_transience_and_plateau_sustained():
    avg = _box(2, 5)
    assert compute_transience_index(avg, FS) == pytest.approx(0.)
    assert compute_plateau_index(avg, FS) == pytest.approx(1.)


def test_transience_and_plateau_transient():
    avg = np.where(T >= 2, np.exp(-(T - 2) / 0.05), 0.)
    assert compute_transience_index(avg, FS) == pytest.approx(1., abs=1e-2)
    assert compute_plateau_index(avg, FS) == pytest.approx(0., abs=1e-2)


def test_transience_exponential_decay_matches_delay():
    tau = 0.5
    avg = np.where(T >= 2, np.exp(-(T - 2) / tau), 0.)
    # RTi = 1 - r(peak + 0.4 s) / r(peak) with a +-0.15 s window
    window = np.exp(-np.arange(0.4 - 0.15, 0.4 + 0.15 + 1 / FS, 1 / FS) / tau)
    assert compute_transience_index(avg, FS) == pytest.approx(1 - np.mean(window), abs=1e-3)


def test_transience_is_scale_and_offset_invariant():
    avg = np.where(T >= 2, np.exp(-(T - 2) / 0.5), 0.)
    assert compute_transience_index(3 * avg + 2, FS) == pytest.approx(compute_transience_index(avg, FS))


def test_transience_nan_without_positive_peak():
    assert np.isnan(compute_transience_index(-_box(2, 5), FS))
    assert np.isnan(compute_plateau_index(np.zeros_like(T), FS))


def test_peak_to_post_peak_ratio_invalid_inputs():
    with pytest.raises(ValueError, match='alpha_dt'):
        compute_peak_to_post_peak_ratio(_box(2, 5), FS, alpha=0.1, alpha_dt=0.1)
    with pytest.raises(ValueError, match='too short'):
        compute_peak_to_post_peak_ratio(np.zeros(100), FS, alpha=0.4, alpha_dt=0.1)


# ---------------------------------------------------------------- tonic release

def test_tonic_release_index():
    flicker = np.zeros_like(T)
    m = (T >= 10) & (T < 29)
    flicker[m] = np.sin(2 * np.pi * 2 * T[m])
    assert compute_tonic_release_index(flicker, FS) == pytest.approx(0.5, abs=1e-3)
    assert compute_tonic_release_index(np.abs(flicker), FS) == pytest.approx(0.)
    assert compute_tonic_release_index(-np.abs(flicker), FS) == pytest.approx(1.)


def test_tonic_release_index_uses_pre_flicker_baseline():
    flicker = np.zeros_like(T)
    m = (T >= 10) & (T < 29)
    flicker[m] = np.abs(np.sin(2 * np.pi * 2 * T[m]))
    assert compute_tonic_release_index(flicker + 5., FS) == pytest.approx(0.)


# ---------------------------------------------------------------- frequency / contrast responses

def test_freq_response_low_pass():
    avg = np.zeros_like(T)
    m = (T >= 10) & (T < 18)
    avg[m] = np.linspace(1, 0, m.sum())
    low, high, idx = compute_freq_response(avg, FS)
    assert low > high
    assert idx == pytest.approx((high - low) / (abs(high) + abs(low)))
    assert -1 <= idx < 0


def test_contrast_response_ratio_increasing():
    avg = np.zeros_like(T)
    m = (T >= 20) & (T < 28)
    avg[m] = np.linspace(0, 1, m.sum())
    low, high, idx = compute_contrast_response_ratio(avg, FS)
    assert high > low
    assert 0 < idx <= 1


def test_freq_and_contrast_response_zero_trace():
    with pytest.warns(UserWarning, match='divide by zero'):
        assert compute_freq_response(np.zeros_like(T), FS)[2] == -1
    with pytest.warns(UserWarning, match='divide by zero'):
        assert compute_contrast_response_ratio(np.zeros_like(T), FS)[2] == -1


# ---------------------------------------------------------------- high frequency index

FS_SCAN = 7.8125
T_SCAN = np.arange(int(32 * FS_SCAN)) / FS_SCAN


def _snippets(signal, n_reps=3, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    return signal[:, None] + noise * rng.normal(size=(signal.size, n_reps))


def test_high_frequency_index_orders_low_and_high_frequency_responses():
    low = _snippets(np.sin(2 * np.pi * 0.75 * T_SCAN))
    high = _snippets(np.sin(2 * np.pi * 3. * T_SCAN))
    assert compute_high_frequency_index(high, FS_SCAN) > compute_high_frequency_index(low, FS_SCAN) + 2


def test_high_frequency_index_invariant_to_linear_transform():
    s = _snippets(np.sin(2 * np.pi * 0.75 * T_SCAN) + 0.3 * np.sin(2 * np.pi * 3. * T_SCAN))
    assert compute_high_frequency_index(3 * s + 5, FS_SCAN) == pytest.approx(compute_high_frequency_index(s, FS_SCAN))


def test_high_frequency_index_constant_scales_log():
    s = _snippets(np.sin(2 * np.pi * 0.75 * T_SCAN) + 0.3 * np.sin(2 * np.pi * 3. * T_SCAN))
    assert (compute_high_frequency_index(s, FS_SCAN, constant=1e5)
            - compute_high_frequency_index(s, FS_SCAN, constant=1e4)) == pytest.approx(1.)


def test_feature_functions_plot():
    fig, axs = plt.subplots(1, 7)
    avg = np.where(T >= 2, np.exp(-(T - 2) / 0.5), 0.)
    compute_high_frequency_index(_snippets(np.sin(2 * np.pi * 0.75 * T_SCAN)), FS_SCAN, plot=axs[0])
    compute_polarity_index(avg, FS, plot=axs[1])
    compute_transience_index(avg, FS, plot=axs[2])
    compute_plateau_index(avg, FS, plot=axs[3])
    compute_tonic_release_index(avg + 0.01 * np.sin(T), FS, plot=axs[4])
    compute_freq_response(avg + 0.01 * np.sin(T), FS, plot=axs[5])
    compute_contrast_response_ratio(avg + 0.01 * np.sin(T), FS, plot=axs[6])
    plt.close('all')


# ---------------------------------------------------------------- table compute_entry / make without database

def _chirp_snippets_row(response, n_reps=3, rep_len=33., t0=5.):
    n = T_SCAN.size
    snippets = np.stack([response(T_SCAN) for _ in range(n_reps)], axis=1)
    snippets_t0 = t0 + np.arange(n_reps) * rep_len
    triggertimes = np.stack([snippets_t0, snippets_t0 + 32.], axis=0)
    assert snippets.shape == (n, n_reps)
    return dict(snippets_t0=snippets_t0, snippets_dt=1 / FS_SCAN, snippets=snippets,
                triggertimes_snippets=triggertimes)


def _fake_bc_table(row):
    attrs = {k: v for k, v in vars(ChirpFeaturesBcTemplate).items()
             if k.startswith('_') and not k.startswith('__') and isinstance(v, (int, float))}
    return make_fake(ChirpFeaturesBcTemplate, ['compute_entry', 'make'], snippets_table=FakeRel(row), **attrs)


def _on_cell(t):
    return (np.where((t >= 2.2) & (t < 5), np.exp(-(t - 2.2) / 1.0), 0.)
            + np.where((t >= 10) & (t < 29), 0.3 * np.sin(2 * np.pi * 1.0 * t), 0.))


def test_chirp_features_bc_make_on_cell():
    table = _fake_bc_table(_chirp_snippets_row(_on_cell))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        table.make(dict(roi_id=1))
    row = table.inserted[0]
    assert row['roi_id'] == 1
    assert row['polarity_index'] > 0.9
    assert 0 < row['transience_index'] < 1
    assert 0 < row['plateau_index'] < 1
    assert 0 <= row['tonic_release_index'] <= 1
    assert np.isfinite(row['high_frequency_index'])


def test_chirp_features_bc_flat_trace_raises():
    table = _fake_bc_table(_chirp_snippets_row(lambda t: np.zeros_like(t)))
    with pytest.raises(ValueError, match='flat'):
        table.compute_entry(dict(roi_id=1))


def test_chirp_features_bc_compute_entry_plot(monkeypatch):
    monkeypatch.setattr(plt, 'show', lambda *a, **k: None)
    table = _fake_bc_table(_chirp_snippets_row(_on_cell))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        table.compute_entry(dict(roi_id=1), plot=True)
    plt.close('all')
