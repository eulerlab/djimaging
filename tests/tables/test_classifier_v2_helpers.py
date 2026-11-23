"""Tests for the RGC classifier V2 helpers (feature extraction, probability summary, classifier file IO, chirp
preprocessing) and the table make methods with fake relations."""
import numpy as np
import pytest

from djimaging.tables.classifier_v2.baden16_traces_v2 import (
    BADEN16_DT, Baden16TracesV2Template, preprocess_chirp_v2)
from djimaging.tables.classifier_v2.celltype_assignment_v2 import (
    CelltypeAssignmentV2Template, baden16_cluster_probs_to_info, classify_cells, extract_features)
from djimaging.tables.classifier_v2.rgc_classifier_v2 import (
    check_classifier_dict, load_classifier_from_file, save_classifier_to_file)
from djimaging.tables.core.stimulus import MOVINGBAR_SNIPPET_BASE_DT, MOVINGBAR_TRIGGER_DELAY
from djimaging.utils.baden16_utils import BADEN_CLUSTER_INFO
from tests.tables._fake_dj import FakeRel, KeyedRel, make_fake


# ---------------------------------------------------------------- probabilities -> info

def _one_hot(cluster_id):
    probs = np.zeros(75)
    probs[cluster_id - 1] = 1.
    return probs


@pytest.mark.parametrize('cluster_id', [1, 5, 32, 46, 50, 75])
def test_cluster_probs_to_info_one_hot(cluster_id):
    cid, gid, supergroup, p_cluster, p_group, p_supergroup, p_class = baden16_cluster_probs_to_info(
        _one_hot(cluster_id))
    row = BADEN_CLUSTER_INFO[cluster_id - 1]
    assert cid == cluster_id
    assert gid == int(row[2])
    assert supergroup == row[3]
    assert p_cluster == p_group == p_supergroup == p_class == 1.


def test_cluster_probs_to_info_sums():
    group_ids = BADEN_CLUSTER_INFO[:, 2].astype(int)
    supergroups = BADEN_CLUSTER_INFO[:, 3]
    rng = np.random.default_rng(0)
    probs = rng.random(75)
    probs /= probs.sum()
    cid, gid, supergroup, p_cluster, p_group, p_supergroup, p_class = baden16_cluster_probs_to_info(probs)
    assert cid == np.argmax(probs) + 1
    assert p_cluster == pytest.approx(probs.max())
    assert p_group == pytest.approx(probs[group_ids == gid].sum())
    assert p_supergroup == pytest.approx(probs[supergroups == supergroup].sum())
    p_dac = probs[supergroups == 'dAC'].sum()
    assert p_class == pytest.approx(p_dac if supergroup == 'dAC' else 1 - p_dac)


def test_cluster_probs_to_info_dac_class_probability():
    dac_idxs = np.flatnonzero(BADEN_CLUSTER_INFO[:, 3] == 'dAC')
    probs = np.zeros(75)
    probs[dac_idxs[0]] = 0.6
    probs[0] = 0.4  # an RGC cluster
    cid, gid, supergroup, p_cluster, p_group, p_supergroup, p_class = baden16_cluster_probs_to_info(probs)
    assert supergroup == 'dAC'
    assert p_class == pytest.approx(0.6)  # probability of being a dAC, not of being an RGC


def test_cluster_probs_to_info_wrong_length():
    with pytest.raises(ValueError, match='75'):
        baden16_cluster_probs_to_info(np.ones(74) / 74)


# ---------------------------------------------------------------- features / classification

def test_extract_features_projection_and_names():
    rng = np.random.default_rng(0)
    chirps, bars = rng.normal(size=(4, 249)), rng.normal(size=(4, 32))
    chirp_feats, bar_feats = rng.normal(size=(249, 5)), rng.normal(size=(32, 3))
    pvals, sizes = rng.random(4), rng.random(4) * 100
    features, names = extract_features(chirps, bars, pvals, sizes, chirp_feats, bar_feats)
    assert features.shape == (4, 5 + 3 + 2)
    assert names == [f'chirp_{i}' for i in range(5)] + [f'bar_{i}' for i in range(3)] + ['bar_ds_pvalue',
                                                                                         'roi_size_um2']
    np.testing.assert_allclose(features[:, :5], chirps @ chirp_feats)
    np.testing.assert_allclose(features[:, 5:8], bars @ bar_feats)
    np.testing.assert_allclose(features[:, 8], pvals)
    np.testing.assert_allclose(features[:, 9], sizes)


class _ArgmaxClassifier:
    """Returns one-hot probabilities of cluster (first chirp feature mod 75)."""

    def predict_proba(self, X):
        probs = np.zeros((X.shape[0], 75))
        probs[np.arange(X.shape[0]), np.round(X[:, 0]).astype(int) % 75] = 1.
        return probs


def test_classify_cells_passes_features_to_classifier():
    chirps = np.zeros((2, 3))
    chirps[0, 0], chirps[1, 0] = 4, 49
    probs = classify_cells(chirps, np.zeros((2, 2)), np.zeros(2), np.zeros(2),
                           chirp_features=np.eye(3)[:, :1], bar_features=np.eye(2)[:, :1], classifier=_ArgmaxClassifier())
    assert np.argmax(probs, axis=1).tolist() == [4, 49]


def test_celltype_assignment_make():
    roi_keys = [dict(roi_id=1), dict(roi_id=2)]
    chirps = np.zeros((2, 3))
    chirps[0, 0], chirps[1, 0] = 4, 49
    table = make_fake(CelltypeAssignmentV2Template, ['make'])
    table._fetch_data = lambda key: (roi_keys, chirps, np.zeros((2, 2)), np.zeros(2), np.zeros(2))
    table.make(dict(classifier_id=1, field='f'), classifier=_ArgmaxClassifier(),
               chirp_feats=np.eye(3)[:, :1], bar_feats=np.eye(2)[:, :1])
    assert [r['roi_id'] for r in table.inserted] == [1, 2]
    assert [r['cluster_id'] for r in table.inserted] == [5, 50]
    assert table.inserted[1]['supergroup'] == 'dAC'
    assert all(r['classifier_id'] == 1 and r['field'] == 'f' for r in table.inserted)


def test_celltype_assignment_make_no_rois():
    table = make_fake(CelltypeAssignmentV2Template, ['make'])
    table._fetch_data = lambda key: ([], np.array([]), np.array([]), np.array([]), np.array([]))
    table.make(dict(classifier_id=1), classifier=None, chirp_feats=None, bar_feats=None)
    assert table.inserted == []


# ---------------------------------------------------------------- classifier dict / file IO

def _clf_dict(**overrides):
    from sklearn.ensemble import RandomForestClassifier
    rng = np.random.default_rng(0)
    train_x = rng.normal(size=(10, 3))
    train_y = np.array([1, 2] * 5)
    clf = RandomForestClassifier(n_estimators=2, random_state=0).fit(train_x, train_y)
    d = dict(classifier=clf, chirp_feats=np.ones((4, 1)), bar_feats=np.ones((2, 1)), feature_names=['a', 'b', 'c'],
             train_x=train_x, train_y=train_y, y_names={1: 'one', 2: 'two'})
    d.update(overrides)
    return d


def test_check_classifier_dict_valid():
    d = _clf_dict()
    assert check_classifier_dict(d) is d


@pytest.mark.parametrize('overrides', [
    dict(train_y=np.array([1, 2, 3])),
    dict(y_names={1: 'one'}),
    dict(classifier='not a classifier'),
    dict(train_x=[[1, 2, 3]]),
])
def test_check_classifier_dict_invalid(overrides):
    with pytest.raises(AssertionError):
        check_classifier_dict(_clf_dict(**overrides))


def test_check_classifier_dict_missing_key():
    d = _clf_dict()
    del d['bar_feats']
    with pytest.raises(AssertionError, match='bar_feats'):
        check_classifier_dict(d)


def test_save_and_load_classifier_roundtrip(tmp_path):
    d = _clf_dict()
    path = tmp_path / 'clf.pkl'
    save_classifier_to_file(classifier_file=str(path), extra='info', **d)
    loaded = load_classifier_from_file(path, store=None)
    assert loaded['extra'] == 'info'
    np.testing.assert_array_equal(loaded['train_x'], d['train_x'])
    np.testing.assert_array_equal(loaded['classifier'].predict(d['train_x']), d['classifier'].predict(d['train_x']))


# ---------------------------------------------------------------- chirp preprocessing (V2)

def _chirp_trace(n_reps=3, fs=1 / BADEN16_DT, t_start=80 * BADEN16_DT, rep_len=258 * BADEN16_DT, noise=0., seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int((t_start + n_reps * rep_len + 5) * fs)) / fs
    trace = np.zeros(t.size)
    starts = t_start + np.arange(n_reps) * rep_len
    for s in starts:
        rel = t - s
        trace += np.where((rel >= 2) & (rel < 5), 1., 0.)
        trace += np.where((rel >= 10) & (rel < 18), 0.5 * np.sin(2 * np.pi * rel), 0.)
    trace += noise * rng.normal(size=t.size)
    triggertimes = np.sort(np.concatenate([starts, starts + 32.]))
    return trace, t[0], t[1] - t[0], triggertimes


def test_preprocess_chirp_v2_shape_norm_and_quality():
    trace, t0, dt, triggertimes = _chirp_trace()
    qi, avg = preprocess_chirp_v2(trace, t0, dt, triggertimes)
    assert avg.shape == (249,)
    assert np.max(np.abs(avg)) == pytest.approx(1.)
    assert qi == pytest.approx(1., abs=0.02)
    # On step response 2-5 s after the trigger (the trace is shifted by one frame, 0.128 s)
    assert np.mean(avg[18:38]) > 0.5


def test_preprocess_chirp_v2_noise_lowers_quality():
    trace, t0, dt, triggertimes = _chirp_trace(noise=3.)
    qi, _ = preprocess_chirp_v2(trace, t0, dt, triggertimes)
    assert qi < 0.6


def test_preprocess_chirp_v2_invariant_to_scale_and_offset():
    trace, t0, dt, triggertimes = _chirp_trace(noise=0.1)
    qi1, avg1 = preprocess_chirp_v2(trace, t0, dt, triggertimes)
    qi2, avg2 = preprocess_chirp_v2(4 * trace + 10, t0, dt, triggertimes)
    assert qi1 == pytest.approx(qi2)
    np.testing.assert_allclose(avg1, avg2, atol=1e-8)


def test_baden16_traces_v2_make():
    trace, t0, dt, triggertimes = _chirp_trace()
    rng = np.random.default_rng(0)
    time_component = rng.normal(size=40)
    stimulus_table = KeyedRel('stim_name', {
        'gChirp': FakeRel(dict(stim_dict=dict(), ntrigger_rep=2)),
        'movingbar': FakeRel(dict(snippet_base_dt=MOVINGBAR_SNIPPET_BASE_DT,
                                  stim_dict=dict(trigger_delay=MOVINGBAR_TRIGGER_DELAY))),
    })
    osds_table = FakeRel(dict(time_component=time_component, time_component_dt=BADEN16_DT, ds_pvalue=0.01),
                         _n_shuffles=1000)
    table = make_fake(Baden16TracesV2Template, ['make'], _stim_name_chirp='gChirp', _stim_name_bar='movingbar',
                      stimulus_table=lambda: stimulus_table,
                      presentation_table=FakeRel(dict(triggertimes=triggertimes)),
                      traces_table=FakeRel(dict(trace=trace, trace_t0=t0, trace_dt=dt)), osds_table=osds_table)
    table.make(dict(roi_id=1))
    entry = table.inserted[0]
    assert entry['preproc_chirp'].shape == (249,)
    np.testing.assert_allclose(entry['preproc_bar'], time_component[:32] / np.max(np.abs(time_component[:32])))
    assert entry['ds_pvalue'] == 0.01


def test_baden16_traces_v2_make_rejects_too_few_shuffles():
    trace, t0, dt, triggertimes = _chirp_trace(n_reps=2)
    stimulus_table = KeyedRel('stim_name', {
        'gChirp': FakeRel(dict(stim_dict=dict(), ntrigger_rep=2)),
        'movingbar': FakeRel(dict(snippet_base_dt=MOVINGBAR_SNIPPET_BASE_DT,
                                  stim_dict=dict(trigger_delay=MOVINGBAR_TRIGGER_DELAY))),
    })
    table = make_fake(Baden16TracesV2Template, ['make'], _stim_name_chirp='gChirp', _stim_name_bar='movingbar',
                      stimulus_table=lambda: stimulus_table,
                      presentation_table=FakeRel(dict(triggertimes=triggertimes)),
                      traces_table=FakeRel(dict(trace=trace, trace_t0=t0, trace_dt=dt)),
                      osds_table=FakeRel(dict(time_component=np.ones(32), time_component_dt=BADEN16_DT, ds_pvalue=0.5),
                                         _n_shuffles=100))
    with pytest.raises(ValueError, match='shuffles'):
        table.make(dict(roi_id=1))
