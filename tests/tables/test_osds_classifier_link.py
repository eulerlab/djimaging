"""OsDsIndexes as source of the classifier's moving bar features, and the per-trial snippet baseline."""
import inspect

import numpy as np
import pytest

from djimaging.tables.classifier_v2.baden16_traces_v2 import (
    BADEN16_DT, MIN_N_SHUFFLES, check_bar_features_match_classifier, preprocess_bar_v2)
from djimaging.tables.core.snippets import SnippetsTemplate
from djimaging.tables.core.stimulus import CHIRP_SNIPPET_BASE_DT, MOVINGBAR_SNIPPET_BASE_DT, StimulusTemplate
from djimaging.tables.response.movingbar.orientation_utils import compute_null_dist, compute_os_ds_idxs

DIR_ORDER = np.array([0, 180, 45, 225, 90, 270, 135, 315])
N_T = 32
N_REPS = 3


def _ds_snippets(rng, drift=0.):
    """Moving bar snippets (time x trials) of a DS cell, trials in presentation order, with a slow drift."""
    t = np.arange(N_T) * BADEN16_DT
    kernel = np.exp(-0.5 * ((t - 2.0) / 0.4) ** 2)
    snippets = []
    for trial, d in enumerate(np.tile(DIR_ORDER, N_REPS)):
        amp = 1. + 0.8 * np.cos(np.deg2rad(d - 90))
        offset = drift * trial / (len(DIR_ORDER) * N_REPS - 1)  # monotonic, e.g. bleaching
        snippets.append(amp * kernel + offset + 0.1 * rng.standard_normal(N_T))
    return np.stack(snippets, axis=1)


def _subtract_baseline(snippets, base_dt=MOVINGBAR_SNIPPET_BASE_DT, dt=BADEN16_DT):
    n_baseline = int(np.round(base_dt / dt))
    return snippets - np.median(snippets[:n_baseline, :], axis=0)


def test_stimulus_defaults_use_baden16_snippet_baseline():
    assert np.isclose(MOVINGBAR_SNIPPET_BASE_DT, 5 * 0.128)
    assert np.isclose(CHIRP_SNIPPET_BASE_DT, 8 * 0.128)
    params_mb = inspect.signature(StimulusTemplate.add_movingbar).parameters
    params_chirp = inspect.signature(StimulusTemplate.add_chirp).parameters
    assert params_mb['snippet_base_dt'].default == MOVINGBAR_SNIPPET_BASE_DT
    assert params_chirp['snippet_base_dt'].default == CHIRP_SNIPPET_BASE_DT


def test_snippets_reject_removed_baseline_dict():
    with pytest.raises(TypeError, match='snippet_base_dt'):
        class Snippets(SnippetsTemplate):  # noqa: F841
            _dt_base_line_dict = {'movingbar': 5 * 0.128}


def test_null_dist_is_reproducible_with_seed():
    rng = np.random.default_rng(0)
    counts = rng.standard_normal((N_REPS, len(DIR_ORDER)))
    dirs = np.deg2rad(np.sort(DIR_ORDER))
    p1, q1, null1 = compute_null_dist(dirs, counts, per=1, iters=200, seed=3)
    p2, q2, null2 = compute_null_dist(dirs, counts, per=1, iters=200, seed=3)
    _, _, null3 = compute_null_dist(dirs, counts, per=1, iters=200, seed=4)
    assert p1 == p2 and q1 == q2
    assert np.array_equal(null1, null2)
    assert not np.array_equal(null1, null3)
    assert null1.shape == (200,)


def test_bar_features_invariant_to_trace_offset_and_scale_with_baseline():
    """Baseline window and standardization of the trace (they differ between PreprocessTraces and the former
    classifier preprocessing) do not change the classifier features if the per-trial baseline is subtracted."""
    snippets = _ds_snippets(np.random.default_rng(1), drift=0.5)
    ref = compute_os_ds_idxs(_subtract_baseline(snippets), DIR_ORDER, BADEN16_DT, n_shuffles=MIN_N_SHUFFLES, seed=0)
    alt = compute_os_ds_idxs(_subtract_baseline(3.7 * snippets - 1.3), DIR_ORDER, BADEN16_DT,
                             n_shuffles=MIN_N_SHUFFLES, seed=0)
    assert np.isclose(ref[0], alt[0])  # DSI
    assert ref[1] == alt[1]  # DS p-value
    assert np.allclose(preprocess_bar_v2(ref[10]), preprocess_bar_v2(alt[10]))


def test_missing_trial_baseline_lets_drift_change_osds():
    """Without per-trial baseline, slow drift (shared by consecutive opposite directions) changes DSI and OSI."""
    snippets = _ds_snippets(np.random.default_rng(2), drift=1.5)
    with_bl = compute_os_ds_idxs(_subtract_baseline(snippets), DIR_ORDER, BADEN16_DT, n_shuffles=200, seed=0)
    without_bl = compute_os_ds_idxs(snippets, DIR_ORDER, BADEN16_DT, n_shuffles=200, seed=0)
    clean = compute_os_ds_idxs(_subtract_baseline(_ds_snippets(np.random.default_rng(2), drift=0.)),
                               DIR_ORDER, BADEN16_DT, n_shuffles=200, seed=0)
    assert np.isclose(with_bl[0], clean[0], atol=0.02)  # DSI
    assert np.isclose(with_bl[4], clean[4], atol=0.02)  # OSI
    assert without_bl[4] > clean[4] + 0.2  # spurious orientation selectivity


def test_preprocess_bar_v2():
    time_component = np.concatenate([np.linspace(0, -2, 20), np.linspace(-2, 1, 20)])
    bar = preprocess_bar_v2(time_component)
    assert bar.shape == (32,)
    assert np.isclose(np.max(np.abs(bar)), 1.)
    assert np.allclose(bar, time_component[:32] / 2.)


@pytest.mark.parametrize('kwargs, match', [
    (dict(n_shuffles=100), 'shuffles'),
    (dict(snippet_base_dt=None), 'baseline'),
    (dict(snippet_base_dt=8 * 0.128), 'baseline'),
    (dict(dt=1 / 31.25), 'dt'),
])
def test_check_bar_features_match_classifier_raises(kwargs, match):
    params = dict(n_shuffles=MIN_N_SHUFFLES, snippet_base_dt=MOVINGBAR_SNIPPET_BASE_DT, dt=BADEN16_DT)
    params.update(kwargs)
    with pytest.raises(ValueError, match=match):
        check_bar_features_match_classifier(**params)


def test_check_bar_features_match_classifier_passes():
    check_bar_features_match_classifier(n_shuffles=MIN_N_SHUFFLES, snippet_base_dt=np.float32(5 * 0.128),
                                        dt=np.float32(0.128))
