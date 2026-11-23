"""Tests for the repeat quality index (RepeatQI) and the sinespot features, computed without a database."""
import numpy as np
import pytest

from djimaging.tables.response.response_quality import RepeatQITemplate
from djimaging.tables.response.sinespot.sinespot_features import (
    SineSpotFeaturesTemplate, compute_sinespot_response_matrix)
from tests.tables._fake_dj import FakeRel, make_fake


def _qi_table(snippets, stim_family='chirp', stim_name=None):
    return make_fake(RepeatQITemplate, ['make', 'get_stim_restriction'],
                     snippets_table=FakeRel(dict(snippets=snippets)),
                     _stim_family=stim_family, _stim_name=stim_name)


def _qi(snippets):
    table = _qi_table(snippets)
    table.make(dict(roi_id=1))
    return table.inserted[0]


def test_repeat_qi_identical_repetitions_is_one():
    t = np.linspace(0, 10, 300)
    snippets = np.tile(np.sin(t)[:, None], (1, 5))
    entry = _qi(snippets)
    assert entry['qidx'] == pytest.approx(1.)
    assert entry['min_qidx'] == pytest.approx(1 / 5)


def test_repeat_qi_pure_noise_is_min_qidx():
    rng = np.random.default_rng(0)
    n_reps = 4
    entry = _qi(rng.normal(size=(20_000, n_reps)))
    # Var of the mean of r independent repeats is 1/r of the single-trial variance
    assert entry['qidx'] == pytest.approx(1 / n_reps, rel=0.05)
    assert entry['min_qidx'] == pytest.approx(1 / n_reps)


def test_repeat_qi_signal_plus_noise_between_bounds():
    rng = np.random.default_rng(1)
    t = np.linspace(0, 10, 2000)
    snippets = np.sin(t)[:, None] + 0.5 * rng.normal(size=(t.size, 3))
    # Expected: var(s) / (var(s) + var(n)) for many samples
    expected = (np.var(np.sin(t)) + 0.25 / 3) / (np.var(np.sin(t)) + 0.25)
    assert _qi(snippets)['qidx'] == pytest.approx(expected, rel=0.05)


def test_repeat_qi_scale_and_offset_invariant():
    rng = np.random.default_rng(2)
    snippets = rng.normal(size=(500, 3)) + np.linspace(0, 1, 500)[:, None]
    assert _qi(3 * snippets + 7)['qidx'] == pytest.approx(_qi(snippets)['qidx'])


@pytest.mark.parametrize('snippets', [np.ones((100, 3)), np.full((100, 3), np.nan)])
def test_repeat_qi_non_finite_raises(snippets):
    with pytest.raises(ValueError, match='not finite'):
        _qi(snippets)


@pytest.mark.parametrize('family,name,expected', [
    ('chirp', 'gChirp', "stim_family='chirp' or stim_name='gChirp'"),
    ('chirp', None, "stim_family='chirp'"),
    (None, 'gChirp', "stim_name='gChirp'"),
    (None, None, dict()),
])
def test_repeat_qi_stim_restriction(family, name, expected):
    assert _qi_table(np.ones((2, 2)), stim_family=family, stim_name=name).get_stim_restriction() == expected


# ---------------------------------------------------------------- sinespot

SINESPOT_AMPS = np.array([1.0, 0.2, 0.5, 0.6, 0.7, 0.4])  # small, large, a, b, c, d


def _sinespot_trace(n_reps=3, fs=31.25, dt_cond=2.0, t_start=3.0, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    n_cond = SINESPOT_AMPS.size
    t = np.arange(int((t_start + n_reps * n_cond * dt_cond + 3.) * fs)) / fs
    trace = np.zeros(t.size)
    triggertimes = t_start + np.arange(n_reps * n_cond) * dt_cond
    for i, tt in enumerate(triggertimes):
        trace[(t >= tt + 0.05) & (t < tt + 1.2)] += SINESPOT_AMPS[i % n_cond]
    trace += noise * rng.normal(size=t.size)
    return trace, t, triggertimes


def test_sinespot_response_matrix_known_amplitudes():
    trace, t, triggertimes = _sinespot_trace()
    mat = compute_sinespot_response_matrix(trace, t, triggertimes, ntrigger_rep=6)
    assert mat.shape == (3, 6)
    np.testing.assert_allclose(mat, np.tile(SINESPOT_AMPS / np.std(trace), (3, 1)), rtol=1e-6)


def test_sinespot_response_matrix_scale_invariant():
    trace, t, triggertimes = _sinespot_trace(noise=0.05)
    mat1 = compute_sinespot_response_matrix(trace, t, triggertimes, ntrigger_rep=6)
    mat2 = compute_sinespot_response_matrix(5 * trace, t, triggertimes, ntrigger_rep=6)
    np.testing.assert_allclose(mat1, mat2)


def test_sinespot_features_make():
    trace, t, triggertimes = _sinespot_trace()
    table = make_fake(
        SineSpotFeaturesTemplate, ['make'], _delay=0.1, _rep_dt=0.8,
        preprocesstraces_table=FakeRel(dict(trace_t0=t[0], trace_dt=t[1] - t[0], trace=trace)),
        presentation_table=FakeRel(dict(triggertimes=triggertimes)),
        stimulus_table=FakeRel(dict(ntrigger_rep=6)))
    table.make(dict(roi_id=1))
    entry = table.inserted[0]
    r = SINESPOT_AMPS / np.std(trace)
    assert entry['response_spot_small'] == pytest.approx(r[0])
    assert entry['response_spot_large'] == pytest.approx(r[1])
    assert entry['surround_index'] == pytest.approx(r[1] - r[0])
    assert entry['offset_index'] == pytest.approx(np.max(r[2:]) - r[0])
    assert entry['response_rep_x_cond'].shape == (3, 6)
