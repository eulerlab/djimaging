from abc import abstractmethod

import datajoint as dj
import numpy as np
from matplotlib import pyplot as plt

from djimaging.utils.dj_utils import get_primary_key
from djimaging.utils.plot_utils import plot_mean_trace_and_std


def cluster_features(X, kind: str, params_dict: dict):
    assert X.ndim == 2, f"{X.shape}"

    if kind == 'hierarchical_ward':
        model, cluster_idxs = cluster_hierarchical_ward(X, **params_dict)
    elif kind == 'hierarchical_correlation':
        model, cluster_idxs = cluster_hierarchical_cc(X, **params_dict)
    elif kind == 'gmm':
        model, cluster_idxs = cluster_gmm(X, **params_dict)
    else:
        raise NotImplementedError

    return model, cluster_idxs


def cluster_hierarchical_ward(X, n_clusters=None, distance_threshold=75):
    from sklearn.cluster import AgglomerativeClustering
    model = AgglomerativeClustering(n_clusters=n_clusters, metric="euclidean", linkage='ward',
                                    distance_threshold=distance_threshold)
    model = model.fit(X)
    cluster_idxs = np.array(model.labels_).astype(float)
    return model, cluster_idxs


def cluster_hierarchical_cc(X, n_clusters=None, distance_threshold=0.9, linkage='complete'):
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import pairwise_distances

    cc_dists = pairwise_distances(X, metric="correlation")
    model = AgglomerativeClustering(n_clusters=n_clusters, metric="precomputed", linkage=linkage,
                                    distance_threshold=distance_threshold)
    model = model.fit(cc_dists)
    cluster_idxs = np.array(model.labels_).astype(float)
    return model, cluster_idxs


def plot_grid_search_results(grid_search):
    import pandas as pd
    import seaborn as sns

    rows = []
    for param, score in zip(grid_search.cv_results_['params'], grid_search.cv_results_['mean_test_score']):
        rows.append(dict(**param, score=score))
    df = pd.DataFrame(rows)

    if df.n_components.nunique() > 1:
        sns.lineplot(data=df, x='n_components', y='score', hue='covariance_type', markers=True)
    else:
        sns.scatterplot(data=df, x='n_components', y='score', hue='covariance_type')
    plt.show()


def cluster_gmm(X, ncomp_max=6, ncomp_min=1, cv=10, cv_metric='bic', seed=45312, plot_results=True):
    from sklearn.mixture import GaussianMixture
    from sklearn.model_selection import GridSearchCV

    cv_metric = cv_metric.lower()

    def gmm_bic_score(estimator, X_):
        return -estimator.bic(X_)

    def gmm_aic_score(estimator, X_):
        return -estimator.bic(X_)

    def gmm_loglikelihood_score(estimator, X_):
        return estimator.score(X_)

    if cv_metric == 'bic':
        scoring = gmm_bic_score
    elif cv_metric == 'aic':
        scoring = gmm_aic_score
    elif cv_metric in ['loglikelihood', 'likelihood']:
        scoring = gmm_loglikelihood_score
    else:
        raise NotImplementedError(cv_metric)

    param_grid = {
        "n_components": range(ncomp_min, ncomp_max + 1),
        "covariance_type": ["spherical", "tied", "diag", "full"],
    }

    if cv == 1:
        cv = [(np.arange(X.shape[0]), np.arange(X.shape[0]))]  # Use train set as test set

    np.random.seed(seed)
    model = GridSearchCV(GaussianMixture(), cv=cv, param_grid=param_grid, n_jobs=10, scoring=scoring, verbose=1)
    model.fit(X)

    if plot_results:
        plot_grid_search_results(model)

    cluster_idxs = model.best_estimator_.fit_predict(X)
    return model, cluster_idxs


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
        return cluster_idxs

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

        _, cluster_idxs = cluster_features(X=np.hstack(features), kind=kind, params_dict=params_dict)
        cluster_idxs = remove_clusters(cluster_idxs=cluster_idxs, min_count=min_count, invalid_value=-1)

        traces = (self.features_table() & key).fetch1('traces')
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
        self.plot1_traces(key=key, kind='averages')

    def plot1_heatmaps(self, key=None):
        self.plot1_traces(key=key, kind='traces')

    def plot1_traces(self, key=None, kind='traces'):
        key = get_primary_key(table=self, key=key)

        stim_names = (self.features_table.params_table & key).fetch1('stim_names').split('_')
        traces = (self.features_table & key).fetch1('traces')
        clusters = (self & key).fetch1('clusters')

        unique_clusters, cluster_counts = np.unique(clusters, return_counts=True)

        n_rows = np.minimum(unique_clusters.size, 15)
        n_cols = len(traces)

        if kind == 'averages':
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, 0.8 * (1 + n_rows)), squeeze=False,
                                    sharex='col', sharey='row')
        else:
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, 2 * (1 + n_rows)), squeeze=False,
                                    sharex='col', sharey='row', gridspec_kw=dict(height_ratios=cluster_counts))

        fig.suptitle(key, y=1, va='bottom')

        for ax_col, stim_i, traces_i in zip(axs.T, stim_names, traces):
            ax_col[0].set(title=f"stim={stim_i}")
            for row_i, (ax, cluster) in enumerate(zip(ax_col, unique_clusters)):
                c_traces_i = traces_i[clusters == cluster, :]

                if kind == 'averages':
                    self._plot_cluster_averages(ax, traces=c_traces_i, c=f'C{row_i}')
                else:
                    self._plot_cluster_trace_heatmaps(ax, traces=c_traces_i, vabsmax=np.max(np.abs(traces_i)))

                ax.set(ylabel=f"cluster={int(cluster)}\nn={c_traces_i.shape[0]}")

        plt.tight_layout()
        plt.show()

    @staticmethod
    def _plot_cluster_averages(ax, traces, c=None):
        plot_mean_trace_and_std(
            ax=ax, time=np.arange(traces.shape[1]), traces=traces, color=c)

    @staticmethod
    def _plot_cluster_trace_heatmaps(ax, traces, vabsmax=None):
        mean_trace = np.mean(traces, axis=0)
        corr_mean = np.array([np.corrcoef(trace, mean_trace)[0, 1] for trace in traces])
        sort_idx = np.argsort(corr_mean)

        im = ax.imshow(traces[sort_idx, :], aspect='auto', cmap='coolwarm', interpolation='none',
                       vmin=-vabsmax, vmax=vabsmax, origin='lower')

        plt.colorbar(im, ax=ax)
