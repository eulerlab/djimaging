"""Tests for the RGC chirp features (on-off and transience index) and the chirp surround index (V2)."""
import matplotlib

matplotlib.use('Agg')

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402

from djimaging.tables.response.chirp.chirp_features_rgc import (  # noqa: E402
    ChirpFeaturesRgcTemplate, compute_transience_index)
from djimaging.tables.response.chirp.chirp_surround_v2 import ChirpSurroundTemplateV2  # noqa: E402
from tests.tables._fake_dj import FakeRel, KeyedRel, make_fake  # noqa: E402


def _snippets(response, fs, n_reps=2, t0=3.7, rep_len=40., trigger_offset=0.):
    n = int(32 * fs)
    times = np.stack([t0 + rep_len * i + np.arange(n) / fs for i in range(n_reps)], axis=1)
    triggers = times[0, :] + trigger_offset
    return response(times - triggers), times, triggers


def _gauss_bump(t, t_peak=3., sigma=0.5):
    return np.exp(-(t - t_peak) ** 2 / (2 * sigma ** 2))


# ---------------------------------------------------------------- RGC transience index

def test_rgc_transience_index_sustained_is_zero():
    snippets, times, triggers = _snippets(lambda t: _gauss_bump(t, t_peak=3., sigma=50.), 500.)
    assert compute_transience_index(snippets, times, triggers) == pytest.approx(0., abs=0.02)


@pytest.mark.parametrize('trigger_offset', [0., 1.])
def test_rgc_transience_index_independent_of_trigger_offset(trigger_offset):
    fs = 500.
    ref = compute_transience_index(*_snippets(_gauss_bump, fs))
    obs = compute_transience_index(*_snippets(_gauss_bump, fs, trigger_offset=trigger_offset))
    assert obs == pytest.approx(ref, abs=1e-3)


@pytest.mark.parametrize('fs', [500., 7.8125])
def test_rgc_transience_index_uses_response_0p4s_after_peak(fs):
    snippets, times, triggers = _snippets(_gauss_bump, fs)
    expected = 1 - _gauss_bump(3.4) / _gauss_bump(3.)  # 1 - r(peak + 0.4 s) / r(peak), as documented
    assert compute_transience_index(snippets, times, triggers) == pytest.approx(expected, abs=0.02)


def test_chirp_features_rgc_make():
    fs = 7.8125

    def on_cell(t):  # smooth edges: FFT resampling of a sharp step rings (Gibbs), which would dominate the index
        return 1 / (1 + np.exp(-(t - 2.3) / 0.1)) / (1 + np.exp((t - 5.3) / 0.1)) + 0.1

    snippets, times, triggers = _snippets(on_cell, fs, t0=0., rep_len=40.)
    row = dict(snippets_t0=times[0], snippets_dt=1 / fs, snippets=snippets,
               triggertimes_snippets=np.stack([triggers, triggers + 32.]))
    table = make_fake(ChirpFeaturesRgcTemplate, ['make'], snippets_table=FakeRel(row))
    table.make(dict(roi_id=3))
    entry = table.inserted[0]
    assert entry['roi_id'] == 3
    assert entry['on_off_index'] > 0.9
    assert entry['transience_index'] == pytest.approx(0., abs=0.1)


# ---------------------------------------------------------------- chirp surround V2

def _surround_table(l_resp, g_resp, polarity_index, fs=7.8125, n_l=3, n_g=3):
    def row(resp, n_reps):
        snippets, times, triggers = _snippets(resp, fs, n_reps=n_reps)
        return dict(snippets_t0=times[0], snippets_dt=1 / fs, snippets=snippets,
                    triggertimes_snippets=np.stack([triggers, triggers + 32.]))

    snippets_table = KeyedRel('stim_name', {'lChirp': FakeRel(row(l_resp, n_l)), 'gChirp': FakeRel(row(g_resp, n_g))})
    attrs = dict(_l_name='lChirp', _g_name='gChirp', _t_on_step=2, _t_off_step=5, _dt=2, _t_plot_max=8,
                 snippets_table=snippets_table, chirp_features_table=FakeRel(dict(polarity_index=polarity_index)))
    return make_fake(ChirpSurroundTemplateV2, ['compute_entry', 'make'], **attrs)


def _step(amp, t_a, t_b):
    return lambda t: np.where((t >= t_a) & (t < t_b), amp, 0.)


def test_chirp_surround_on_cell_with_suppressive_surround():
    table = _surround_table(_step(1.0, 2, 5), _step(0.3, 2, 5), polarity_index=1.)
    table.make(dict(roi_id=1))
    entry = table.inserted[0]
    np.testing.assert_allclose(entry['l_response_mus'], 1.0)
    np.testing.assert_allclose(entry['g_response_mus'], 0.3)
    assert entry['chirp_surround_index'] == pytest.approx(0.3 - 1.0)


def test_chirp_surround_off_cell_uses_off_step_window():
    table = _surround_table(_step(1.0, 4.95, 8), _step(1.5, 4.95, 8), polarity_index=-1.)
    surround_index, l_mus, g_mus = table.compute_entry(dict(roi_id=1))
    np.testing.assert_allclose(l_mus, 1.0)
    np.testing.assert_allclose(g_mus, 1.5)
    assert surround_index == pytest.approx(0.5)


def test_chirp_surround_negative_local_response_is_clipped():
    table = _surround_table(_step(-1.0, 2, 5), _step(0.5, 2, 5), polarity_index=1.)
    surround_index, _, _ = table.compute_entry(dict(roi_id=1))
    assert surround_index == pytest.approx(0.5)


def test_chirp_surround_different_number_of_repetitions():
    table = _surround_table(_step(1.0, 2, 5), _step(0.3, 2, 5), polarity_index=1., n_l=2, n_g=3)
    surround_index, l_mus, g_mus = table.compute_entry(dict(roi_id=1))
    assert l_mus.shape == (2,) and g_mus.shape == (3,)
    assert np.all(np.isfinite(l_mus)) and np.all(np.isfinite(g_mus))
    assert surround_index == pytest.approx(-0.7)


def test_chirp_surround_plot(monkeypatch):
    monkeypatch.setattr(plt, 'show', lambda *a, **k: None)
    table = _surround_table(_step(1.0, 2, 5), _step(0.3, 2, 5), polarity_index=1.)
    table.compute_entry(dict(roi_id=1), plot=True)
    plt.close('all')


def test_chirp_surround_plot_different_number_of_repetitions(monkeypatch):
    monkeypatch.setattr(plt, 'show', lambda *a, **k: None)
    table = _surround_table(_step(1.0, 2, 5), _step(0.3, 2, 5), polarity_index=1., n_l=2, n_g=3)
    try:
        surround_index, _, _ = table.compute_entry(dict(roi_id=1), plot=True)
    finally:
        plt.close('all')
    assert surround_index == pytest.approx(-0.7)


def test_chirp_surround_sampling_rate_mismatch_raises():
    table = _surround_table(_step(1.0, 2, 5), _step(0.3, 2, 5), polarity_index=1.)
    g_row = table.snippets_table.rels['gChirp'].rows[0]
    g_row['snippets_dt'] = g_row['snippets_dt'] * 2
    with pytest.raises(ValueError, match='Sampling rate'):
        table.compute_entry(dict(roi_id=1))
