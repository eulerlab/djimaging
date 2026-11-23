"""Tests for the clustering and decomposition helpers (and their table methods) without a database."""
import numpy as np
import pytest

from djimaging.tables.clustering.clustering import (
    ClusteringTemplate, cluster_features, cluster_gmm, cluster_hierarchical_cc, cluster_hierarchical_ward,
    remove_clusters, sort_clusters)
from djimaging.tables.clustering.decomposition import (
    FeaturesTemplate, compute_features, compute_features_pca, compute_features_sparse_pca,
    compute_variance_explained_sparse_pca)
from tests.tables._fake_dj import FakeRel, make_fake


def _blobs(n_per=20, centers=((0, 0), (10, 0), (0, 10)), seed=0):
    rng = np.random.default_rng(seed)
    X = np.concatenate([np.asarray(c) + 0.3 * rng.normal(size=(n_per, 2)) for c in centers])
    labels = np.repeat(np.arange(len(centers)), n_per)
    return X, labels


def _same_partition(a, b):
    """True if two label vectors describe the same partition (up to label permutation)."""
    a, b = np.asarray(a), np.asarray(b)
    return all(np.unique(b[a == la]).size == 1 for la in np.unique(a)) and np.unique(a).size == np.unique(b).size


# ---------------------------------------------------------------- remove_clusters

def test_remove_clusters_relabels_consecutively_from_one():
    out = remove_clusters(np.array([5, 5, 2, 2, 2, 9]), min_count=0)
    np.testing.assert_array_equal(out, [2, 2, 1, 1, 1, 3])


def test_remove_clusters_removes_small_clusters():
    out = remove_clusters(np.array([0, 0, 0, 1, 2, 2, 2, 2]), min_count=2, invalid_value=-7)
    np.testing.assert_array_equal(out, [1, 1, 1, -7, 2, 2, 2, 2])


def test_remove_clusters_float_labels():
    out = remove_clusters(np.array([1., 0., 1.]), min_count=0)
    np.testing.assert_array_equal(out, [2, 1, 2])


@pytest.mark.xfail(strict=True, reason="BUG: remove_clusters keeps clusters with count > min_count, but min_count is "
                                       "documented as the minimum number of samples a cluster must have to be retained")
def test_remove_clusters_keeps_clusters_with_exactly_min_count():
    out = remove_clusters(np.array([0, 0, 1, 1, 1, 2]), min_count=2)
    np.testing.assert_array_equal(out, [1, 1, 2, 2, 2, -1])


# ---------------------------------------------------------------- sort_clusters

def _traces_for_sorting():
    t = np.linspace(0, 2 * np.pi, 50)
    base = np.sin(t)
    # cluster 1: 3 samples similar to base; cluster 2: 5 samples = base (largest); cluster 3: 2 samples = -base
    traces = np.concatenate([np.tile(base + 0.5 * np.cos(t), (3, 1)), np.tile(base, (5, 1)), np.tile(-base, (2, 1))])
    cluster_idxs = np.array([1] * 3 + [2] * 5 + [3] * 2)
    return traces, cluster_idxs


def test_sort_clusters_largest_first_then_by_correlation():
    traces, cluster_idxs = _traces_for_sorting()
    out = sort_clusters(traces, cluster_idxs)
    np.testing.assert_array_equal(out, [2] * 3 + [1] * 5 + [3] * 2)


def test_sort_clusters_keeps_invalid():
    traces, cluster_idxs = _traces_for_sorting()
    cluster_idxs[0] = -1
    out = sort_clusters(traces, cluster_idxs)
    assert out[0] == -1
    np.testing.assert_array_equal(out[1:], [2] * 2 + [1] * 5 + [3] * 2)


def test_sort_clusters_single_cluster_unchanged():
    cluster_idxs = np.array([4, 4, -1])
    assert sort_clusters(np.ones((3, 5)), cluster_idxs) is cluster_idxs


@pytest.mark.xfail(strict=True, reason="BUG: sort_clusters fills the output with a hard-coded -1 instead of invalid_value")
def test_sort_clusters_respects_invalid_value():
    traces, cluster_idxs = _traces_for_sorting()
    cluster_idxs[0] = 0
    out = sort_clusters(traces, cluster_idxs, invalid_value=0)
    assert out[0] == 0


# ---------------------------------------------------------------- clustering algorithms

def test_cluster_hierarchical_ward_finds_blobs():
    X, labels = _blobs()
    _, idxs = cluster_hierarchical_ward(X, n_clusters=None, distance_threshold=20)
    assert _same_partition(idxs, labels)
    assert idxs.dtype == float


def test_cluster_hierarchical_cc_groups_by_shape():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 2 * np.pi, 40)
    shapes = [np.sin(t), np.cos(t), np.sin(2 * t)]
    X = np.concatenate([(1 + rng.random((10, 1))) * s + 0.05 * rng.normal(size=(10, t.size)) for s in shapes])
    _, idxs = cluster_hierarchical_cc(X, distance_threshold=0.5)
    assert _same_partition(idxs, np.repeat([0, 1, 2], 10))


def test_cluster_gmm_finds_blobs():
    X, labels = _blobs()
    model, idxs = cluster_gmm(X, ncomp_min=2, ncomp_max=4, cv=1, covariance_types=('spherical',),
                              plot_results=False)
    assert model.best_params_['n_components'] == 3
    assert _same_partition(idxs, labels)


@pytest.mark.parametrize('kind,params', [
    ('hierarchical_ward', dict(distance_threshold=20)),
    ('hierarchical_correlation', dict(n_clusters=2, distance_threshold=None)),
])
def test_cluster_features_dispatch(kind, params):
    X, _ = _blobs()
    model, idxs = cluster_features(X, kind=kind, params_dict=params)
    assert idxs.shape == (X.shape[0],)


def test_cluster_features_unknown_kind():
    with pytest.raises(NotImplementedError):
        cluster_features(np.zeros((3, 2)), kind='kmeans', params_dict={})
    with pytest.raises(NotImplementedError):
        cluster_gmm(np.zeros((3, 2)), cv_metric='unknown', plot_results=False)


# ---------------------------------------------------------------- decomposition

def _low_rank_traces(n=60, t=40, rank=3, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, rank)) @ rng.normal(size=(rank, t))


def test_compute_features_pca_reconstructs_low_rank_data():
    traces = [_low_rank_traces(rank=3), _low_rank_traces(rank=2, seed=1)]
    features, rec, infos = compute_features_pca(traces, ncomps=[3, 2])
    assert features[0].shape == (60, 3) and features[1].shape == (60, 2)
    for tr, re, info in zip(traces, rec, infos):
        np.testing.assert_allclose(re, tr, atol=1e-8)
        assert np.sum(info['explained_variance_ratio']) == pytest.approx(1.)


def test_compute_features_pca_standardized_reconstruction():
    traces = [_low_rank_traces(rank=3) * 5 + 2]
    _, rec, _ = compute_features_pca(traces, ncomps=[3], standardize=True)
    np.testing.assert_allclose(rec[0], traces[0], atol=1e-8)


def test_variance_explained_sparse_pca_matches_pca_for_orthonormal_components():
    from sklearn.decomposition import PCA
    X = _low_rank_traces(rank=5) + 0.1 * np.random.default_rng(3).normal(size=(60, 40))
    pca = PCA(n_components=3).fit(X)
    explained, total = compute_variance_explained_sparse_pca(X, pca.components_.T)
    assert explained / total == pytest.approx(np.sum(pca.explained_variance_ratio_), rel=1e-6)


def test_compute_features_sparse_pca():
    traces = [_low_rank_traces(n=30, t=20, rank=2)]
    features, rec, infos = compute_features_sparse_pca(traces, ncomps=[2], alpha=0.1, verboselvl=0)
    assert features[0].shape == (30, 2)
    assert infos[0]['components'].shape == (2, 20)
    assert 0.9 < infos[0]['explained_variance_ratio'] <= 1. + 1e-9
    if rec[0] is not None:
        assert np.corrcoef(rec[0].ravel(), traces[0].ravel())[0, 1] > 0.95


def test_compute_features_dispatch_and_post_standardize():
    traces = [_low_rank_traces(rank=3)]
    features, rec, infos = compute_features(traces, ncomps=[3], kind='PCA', params_dict={}, post_standardize=True)
    np.testing.assert_allclose(features[0].mean(axis=0), 0., atol=1e-10)
    np.testing.assert_allclose(features[0].std(axis=0), 1., atol=1e-10)

    features, rec, infos = compute_features(traces, ncomps=[], kind='none', params_dict={})
    np.testing.assert_array_equal(features[0], traces[0])
    assert features[0] is not traces[0]
    assert infos == [dict()]

    with pytest.raises(NotImplementedError):
        compute_features(traces, ncomps=[3], kind='ica', params_dict={})


# ---------------------------------------------------------------- table methods with fake relations

class _JoinRel(FakeRel):
    """Rows joined on `roi_id`, restriction by a "stim_name='x'" string selects rows, proj renames attributes."""

    def __and__(self, other):
        if isinstance(other, str) and other.startswith('stim_name='):
            name = other.split('=')[1].strip("'")
            return _JoinRel(rows=[r for r in self.rows if r.get('stim_name') == name])
        return self

    def proj(self, *args, **renames):
        if not renames:
            return self
        return _JoinRel(rows=[{'roi_id': r['roi_id'], **{new: r[old] for new, old in renames.items()}}
                              for r in self.rows])

    def __mul__(self, other):
        by_roi = {r['roi_id']: r for r in other.rows}
        return _JoinRel(rows=[{**r, **by_roi[r['roi_id']]} for r in self.rows if r['roi_id'] in by_roi])


def _features_table(n_rois=4, n_t=30, dt=0.1, t0=1.0):
    rng = np.random.default_rng(0)
    averages = [dict(roi_id=i, stim_name=s, average=rng.normal(size=n_t), average_norm=rng.normal(size=n_t),
                     average_dt=dt, average_t0=t0)
                for i in range(n_rois) for s in ('gChirp', 'lChirp')]
    roi_table = _JoinRel(rows=[dict(roi_id=i) for i in range(n_rois)], primary_key=['roi_id'])
    return make_fake(FeaturesTemplate, ['fetch_traces'], _restr_filter=None, roi_filter_table=None,
                     _use_channels=False, roi_table=roi_table, averages_table=_JoinRel(rows=averages),
                     params_table=FakeRel(dict(norm_trace=0, stim_names='gChirp_lChirp')))


@pytest.mark.xfail(strict=True, reason="BUG: FeaturesTemplate.fetch_traces builds times as arange(n_rois) * dt + t0 "
                                       "(a 1d float array) and passes it to truncated_vstack, which raises TypeError")
def test_features_fetch_traces_single_roi():
    table = _features_table(n_rois=1, n_t=30)
    traces, times, roi_keys, stim_names = table.fetch_traces(dict(features_id=1))
    assert stim_names == ['gChirp', 'lChirp']
    assert roi_keys == [dict(roi_id=0)]
    assert traces[0].shape == (1, 30)


@pytest.mark.xfail(strict=True, reason="BUG: FeaturesTemplate.fetch_traces calls tab.fetch1(...) on the multi-ROI join "
                                       "(and builds times from the number of ROIs instead of the trace length)")
def test_features_fetch_traces_multiple_rois():
    table = _features_table(n_rois=4, n_t=30, dt=0.1, t0=1.0)
    traces, times, roi_keys, stim_names = table.fetch_traces(dict(features_id=1))
    assert roi_keys == [dict(roi_id=i) for i in range(4)]
    assert traces[0].shape == (4, 30)
    assert times[0].shape == (4, 30)
    np.testing.assert_allclose(times[0][0], 1.0 + np.arange(30) * 0.1)


class _PartRecorder(FakeRel):
    def __init__(self):
        super().__init__()
        self.inserted = []

    def insert1(self, row, **kwargs):
        self.inserted.append(dict(row))


def _clustering_table(kind, decomp_kind, features, traces):
    roi_features = FakeRel(rows=[dict(features_id=1, roi_id=i, features_idx=i) for i in range(len(features[0]))],
                           primary_key=['features_id', 'roi_id'])
    features_table = FakeRel(dict(features=features, traces=traces),
                             params_table=FakeRel(dict(kind=decomp_kind)), RoiFeatures=roi_features)
    roi_cluster = _PartRecorder()
    table = make_fake(ClusteringTemplate, ['make'], features_table=features_table,
                      params_table=FakeRel(dict(kind=kind, params_dict=dict(distance_threshold=20), min_count=0)),
                      RoiCluster=lambda: roi_cluster)
    return table, roi_cluster


def test_clustering_make_ward():
    X, labels = _blobs()
    table, roi_cluster = _clustering_table('hierarchical_ward', 'none', [X], [X])
    table.make(dict(features_id=1, clustering_id=1))
    clusters = table.inserted[0]['clusters']
    assert _same_partition(clusters, labels)
    assert set(np.unique(clusters)) == {1, 2, 3}
    assert [r['cluster_idx'] for r in roi_cluster.inserted] == list(clusters)
    assert all(set(r) == {'features_id', 'clustering_id', 'roi_id', 'cluster_idx'} for r in roi_cluster.inserted)


@pytest.mark.xfail(strict=True, reason="BUG: ClusteringTemplate.make prints 'Skipping clustering ... Not supported.' "
                                       "but does not return, so it clusters and inserts anyway")
def test_clustering_make_skips_unsupported_combination():
    X, _ = _blobs()
    table, _ = _clustering_table('hierarchical_ward', 'pca', [X], [X])
    table.make(dict(features_id=1, clustering_id=1))
    assert table.inserted == []
