from abc import abstractmethod
import numpy as np
import datajoint as dj
from matplotlib import pyplot as plt

from djimaging.user.alpha.plot.plot_traces import plot_mean_trace_and_std
from djimaging.utils.dj_utils import get_primary_key


def cluster_features(X, kind: str, params_dict: dict) -> np.ndarray:
    assert X.ndim == 2, f"{X.shape}"

    if kind == 'hierarchical_ward':
        cluster_idxs = cluster_hierarchical_ward(X, **params_dict)
    elif kind == 'hierarchical_correlation':
        cluster_idxs = cluster_hierarchical_cc(X, **params_dict)
    elif kind == 'gmm':
        cluster_idxs = cluster_gmm(X, **params_dict)
    else:
        raise NotImplementedError

    return cluster_idxs


def cluster_hierarchical_ward(X, n_clusters=None, distance_threshold=75) -> np.ndarray:
    from sklearn.cluster import AgglomerativeClustering
    model = AgglomerativeClustering(n_clusters=n_clusters, metric="euclidean", linkage='ward',
                                    distance_threshold=distance_threshold)
    model = model.fit(X)
    cluster_idxs = np.array(model.labels_).astype(float)
    return cluster_idxs


def cluster_hierarchical_cc(X, n_clusters=None, distance_threshold=0.9, linkage='complete') -> np.ndarray:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import pairwise_distances

    cc_dists = pairwise_distances(X, metric="correlation")
    model = AgglomerativeClustering(n_clusters=n_clusters, metric="precomputed", linkage=linkage,
                                    distance_threshold=distance_threshold)
    model = model.fit(cc_dists)
    cluster_idxs = np.array(model.labels_).astype(float)
    return cluster_idxs


def cluster_gmm(X, ncomp_max=6, cv=10, cv_metric='bic'):
    from sklearn.mixture import GaussianMixture
    from sklearn.model_selection import GridSearchCV

    cv_metric = cv_metric.lower()
    assert cv_metric in ['bic', 'aic']

    def gmm_bic_score(estimator, X_):
        return -estimator.bic(X_)

    def gmm_aic_score(estimator, X_):
        return -estimator.bic(X_)

    param_grid = {
        "n_components": range(1, ncomp_max + 1),
        "covariance_type": ["spherical", "tied", "diag", "full"],
    }

    grid_search = GridSearchCV(GaussianMixture(), cv=cv, param_grid=param_grid,
                               scoring=gmm_bic_score if cv_metric == 'bic' else gmm_aic_score, verbose=1)
    grid_search.fit(X)
    cluster_idxs = grid_search.best_estimator_.fit_predict(X)

    return cluster_idxs


class ClusteringParametersTemplate(dj.Lookup):
    database = ""

    @property
    def definition(self):
        definition = """
        clustering_id: int # unique param set id
        ---
        kind: varchar(255)
        params_dict: longblob
        min_count: int
        """
        return definition

    def add(self, kind, params_dict, clustering_id=1, min_count=0, skip_duplicates=False):
        assert isinstance(params_dict, dict)
        key = dict(clustering_id=clustering_id, kind=kind, params_dict=params_dict, min_count=min_count)
        self.insert1(key, skip_duplicates=skip_duplicates)


def remove_clusters(cluster_idxs, min_count, invalid_value=-1):
    new_cluster_idxs = np.full(cluster_idxs.size, invalid_value)

    new_cluster = 1
    for cluster, count in zip(*np.unique(cluster_idxs, return_counts=True)):
        if count > min_count:
            new_cluster_idxs[cluster_idxs == cluster] = new_cluster
            new_cluster += 1

    return new_cluster_idxs


def sort_clusters(traces, cluster_idxs, invalid_value=-1):
    """Sort clusters by correlation to the largest cluster"""
    u_cidxs, counts = np.unique(cluster_idxs[cluster_idxs != invalid_value], return_counts=True)

    if u_cidxs.size <= 1:
        return u_cidxs

    X_means = [np.mean(traces[cluster_idxs == cidx, :], axis=0) for cidx in u_cidxs]
    ccs = np.corrcoef(X_means)[np.argmax(counts)]
    sorted_u_cidxs = u_cidxs[np.argsort(ccs)[::-1]]

    sorted_cluster_idxs = np.full(cluster_idxs.size, -1)
    for new_cidx, cidx in zip(sorted_u_cidxs, u_cidxs):
        sorted_cluster_idxs[cluster_idxs == new_cidx] = cidx
    return sorted_cluster_idxs


class ClusteringTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = """
        -> self.features_table
        -> self.params_table
        ---
        clusters : longblob
        """
        return definition

    @property
    @abstractmethod
    def features_table(self):
        pass

    @property
    @abstractmethod
    def params_table(self):
        pass

    def make(self, key):

        kind, params_dict, min_count = (self.params_table & key).fetch1('kind', 'params_dict', 'min_count')
        features = (self.features_table & key).fetch1('features')

        decomp_kind = (self.features_table.params_table & key).fetch1('kind')

        if kind == 'gmm' and decomp_kind == 'none':
            return
        elif kind == 'hierarchical_correlation' and not decomp_kind == 'none':
            return

        cluster_idxs = cluster_features(X=np.hstack(features), kind=kind, params_dict=params_dict)
        cluster_idxs = remove_clusters(cluster_idxs=cluster_idxs, min_count=min_count, invalid_value=-1)

        traces = self.features_table().fetch_traces(key=key)[0]
        cluster_idxs = sort_clusters(traces=np.hstack(traces), cluster_idxs=cluster_idxs, invalid_value=-1)

        main_key = key.copy()
        main_key['clusters'] = cluster_idxs
        self.insert1(main_key)

        for cluster_idx, row in zip(cluster_idxs, self.features_table.RoiFeatures & key):
            roi_key = key.copy()
            roi_key.update({k: v for k, v in row.items() if k in self.features_table.RoiFeatures.primary_key})
            roi_key['cluster_idx'] = cluster_idx
            self.RoiCluster().insert1(roi_key)

    class RoiCluster(dj.Part):
        @property
        def definition(self):
            definition = """
            -> master
            -> master.features_table.RoiFeatures
            ---
            cluster_idx : int
            """
            return definition

    def plot1(self, key=None):
        key = get_primary_key(table=self, key=key)

        clusters = (self & key).fetch1('clusters')

        fig, ax = plt.subplots(1, 1, figsize=(6, 2))
        fig.suptitle(key)
        ax.hist(clusters, bins=np.arange(clusters.min() - 0.25, clusters.max() + 0.5, 0.5))
        ax.set(title='Counts')
        plt.tight_layout()
        plt.show()

    def plot1_averages(self, key=None):
        key = get_primary_key(table=self, key=key)

        traces, times, roi_keys, stim_names = self.features_table().fetch_traces(key=key)
        clusters = (self & key).fetch1('clusters')

        unique_clusters = np.unique(clusters)

        n_rows = np.minimum(unique_clusters.size, 15)
        n_cols = len(traces)

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, 0.8 * (1 + n_rows)),
                                sharex='all', sharey='all', squeeze=False)
        fig.suptitle(key)

        for ax_col, stim_i, traces_i, times_i in zip(axs.T, stim_names, traces, times):
            ax_col[0].set(title=stim_i)
            for row_i, (ax, cluster) in enumerate(zip(ax_col, unique_clusters)):
                c = f'C{row_i}'
                plot_mean_trace_and_std(
                    ax=ax, time=np.mean(times_i, axis=0), traces=traces_i[clusters == cluster], color=c)
                ax.set(ylabel=f"{int(cluster)}\n{np.sum(clusters == cluster)}")

        plt.tight_layout()
        plt.show()
