"""
Clustering of features (e.g. PCA, parse PCA, etc.) of traces

Example usage:

from djimaging.tables import clustering

@schema
class FeaturesParams(clustering.FeaturesParamsTemplate):
    pass

@schema
class Features(clustering.FeaturesTemplate):

    # Optional quality filtering, defines table and restriction on ROI level
    _restr_filter = None
    roi_quality_table = None  # Could be any table to filter data

    params_table = FeaturesParams
    averages_table = Averages
    roi_table = Roi

    class RoiFeatures(clustering.FeaturesTemplate.RoiFeatures):
        pass

@schema
class ClusteringParameters(clustering.ClusteringParametersTemplate):
    pass

@schema
class Clustering(clustering.ClusteringTemplate):
    features_table = Features
    params_table = ClusteringParameters

    class RoiCluster(clustering.ClusteringTemplate.RoiCluster):
        pass
"""

from abc import abstractmethod

import datajoint as dj
import numpy as np
from djimaging.utils import plot_utils
from matplotlib import pyplot as plt

from djimaging.utils.dj_utils import get_primary_key


class ClusteringParametersTemplate(dj.Lookup):
    database = ""

    @property
    def definition(self):
        definition = """
        clustering_id: tinyint unsigned # unique param set id
        ---
        kind: varchar(191)
        params_dict: longblob
        min_count: int
        """
        return definition

    def add(self, kind: str, params_dict: dict, clustering_id: int = 1, min_count: int = 0,
            skip_duplicates: bool = False) -> None:
        """Insert a clustering parameter set into the table.

        Args:
            kind: Clustering algorithm identifier (e.g. 'gmm', 'hierarchical_ward').
            params_dict: Algorithm-specific keyword arguments passed to the clustering function.
            clustering_id: Unique integer identifier for this parameter set.
            min_count: Minimum number of ROIs a cluster must contain to be kept.
            skip_duplicates: If True, silently skip insertion when the key already exists.
        """
        assert isinstance(params_dict, dict)
        key = dict(clustering_id=clustering_id, kind=kind, params_dict=params_dict, min_count=min_count)
        self.insert1(key, skip_duplicates=skip_duplicates)


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
    def key_source(self):
        try:
            return self.features_table.proj() * self.params_table.proj()
        except (AttributeError, TypeError):
            pass

    @property
    @abstractmethod
    def features_table(self):
        pass

    @property
    @abstractmethod
    def params_table(self):
        pass

    def make(self, key: dict) -> None:
        """Compute clusters for all ROIs sharing the given key and populate the table.

        Args:
            key: DataJoint primary key dict identifying a unique combination of features
                and clustering parameters.
        """
        kind, params_dict, min_count = (self.params_table & key).fetch1('kind', 'params_dict', 'min_count')
        features = (self.features_table & key).fetch1('features')
        decomp_kind = (self.features_table.params_table & key).fetch1('kind')

        if (kind == 'gmm' and decomp_kind == 'none') or (kind == 'hierarchical_ward' and not decomp_kind == 'none'):
            print(f'Skipping clustering with kind={kind} and decomp_kind={decomp_kind}. Not supported.')

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

    def plot1(self, key: dict = None) -> None:
        """Plot a histogram of cluster assignments for a single entry.

        Args:
            key: DataJoint primary key dict. If None, prompts for selection.
        """
        key = get_primary_key(table=self, key=key)

        clusters = (self & key).fetch1('clusters')

        fig, ax = plt.subplots(1, 1, figsize=(6, 2))
        plot_utils.set_long_title(fig=fig, title=key)
        ax.hist(clusters, bins=np.arange(clusters.min() - 0.25, clusters.max() + 0.5, 0.5))
        ax.set(title='Counts')
        plt.tight_layout()
        plt.show()

    def plot1_averages(self, key: dict = None) -> None:
        """Plot cluster-averaged traces for a single entry.

        Args:
            key: DataJoint primary key dict. If None, prompts for selection.
        """
        self.plot1_traces(key=key, kind='averages')

    def plot1_heatmaps(self, key: dict = None) -> None:
        """Plot heatmaps of traces grouped by cluster for a single entry.

        Args:
            key: DataJoint primary key dict. If None, prompts for selection.
        """
        self.plot1_traces(key=key, kind='traces')

    def plot1_traces(self, key: dict = None, kind: str = 'traces') -> None:
        """Plot traces or averages grouped by cluster for a single entry.

        Args:
            key: DataJoint primary key dict. If None, prompts for selection.
            kind: One of ``'traces'`` (heatmap per ROI) or ``'averages'`` (cluster mean).
        """
        key = get_primary_key(table=self, key=key)

        stim_names = (self.features_table.params_table & key).fetch1('stim_names').split('_')
        traces_list = (self.features_table & key).fetch1('traces')
        clusters = (self & key).fetch1('clusters')
        plot_utils.plot_clusters(traces_list, stim_names, clusters, kind=kind, title=key)


def cluster_features(X: np.ndarray, kind: str, params_dict: dict) -> tuple:
    """Dispatch to the appropriate clustering algorithm.

    Args:
        X: 2-D feature matrix of shape ``(n_samples, n_features)``.
        kind: Algorithm name; one of ``'hierarchical_ward'``, ``'hierarchical_correlation'``,
            or ``'gmm'``.
        params_dict: Keyword arguments forwarded to the selected clustering function.

    Returns:
        A tuple ``(model, cluster_idxs)`` where *model* is the fitted sklearn object and
        *cluster_idxs* is a float array of shape ``(n_samples,)`` with cluster labels.

    Raises:
        NotImplementedError: If *kind* is not recognised.
    """
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


def cluster_hierarchical_ward(X: np.ndarray, n_clusters: int = None,
                              distance_threshold: float = 75) -> tuple:
    """Cluster samples using agglomerative (Ward) hierarchical clustering.

    Args:
        X: 2-D feature matrix of shape ``(n_samples, n_features)``.
        n_clusters: Target number of clusters. Mutually exclusive with *distance_threshold*.
        distance_threshold: Linkage distance above which clusters are not merged.

    Returns:
        A tuple ``(model, cluster_idxs)`` where *model* is the fitted
        ``AgglomerativeClustering`` object and *cluster_idxs* is a float array of shape
        ``(n_samples,)`` with cluster labels.
    """
    from sklearn.cluster import AgglomerativeClustering
    model = AgglomerativeClustering(n_clusters=n_clusters, metric="euclidean", linkage='ward',
                                    distance_threshold=distance_threshold)
    model = model.fit(X)
    cluster_idxs = np.array(model.labels_).astype(float)
    return model, cluster_idxs


def cluster_hierarchical_cc(X: np.ndarray, n_clusters: int = None,
                            distance_threshold: float = 0.9, linkage: str = 'complete') -> tuple:
    """Cluster samples using correlation-based agglomerative hierarchical clustering.

    Args:
        X: 2-D feature matrix of shape ``(n_samples, n_features)``.
        n_clusters: Target number of clusters. Mutually exclusive with *distance_threshold*.
        distance_threshold: Linkage distance above which clusters are not merged.
        linkage: Linkage criterion passed to ``AgglomerativeClustering``.

    Returns:
        A tuple ``(model, cluster_idxs)`` where *model* is the fitted
        ``AgglomerativeClustering`` object and *cluster_idxs* is a float array of shape
        ``(n_samples,)`` with cluster labels.
    """
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import pairwise_distances

    cc_dists = pairwise_distances(X, metric="correlation")
    model = AgglomerativeClustering(n_clusters=n_clusters, metric="precomputed", linkage=linkage,
                                    distance_threshold=distance_threshold)
    model = model.fit(cc_dists)
    cluster_idxs = np.array(model.labels_).astype(float)
    return model, cluster_idxs


def plot_grid_search_results(grid_search) -> None:
    """Plot cross-validation scores from a GMM grid search as a line or scatter plot.

    Args:
        grid_search: Fitted ``GridSearchCV`` object whose ``cv_results_`` contain
            ``'params'`` and ``'mean_test_score'`` entries.
    """
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


def cluster_gmm(X: np.ndarray, ncomp_max: int = 6, ncomp_min: int = 1, cv: int = 10,
                cv_metric: str = 'bic', covariance_types: tuple = ("spherical", "tied", "diag", "full"),
                seed: int = 45312, plot_results: bool = True) -> tuple:
    """Cluster samples using a Gaussian Mixture Model selected via cross-validated grid search.

    Args:
        X: 2-D feature matrix of shape ``(n_samples, n_features)``.
        ncomp_max: Maximum number of mixture components to evaluate.
        ncomp_min: Minimum number of mixture components to evaluate.
        cv: Number of cross-validation folds. If 1, the training set is used as the test set.
        cv_metric: Model-selection criterion; one of ``'bic'``, ``'aic'``, or
            ``'loglikelihood'``.
        covariance_types: GMM covariance structures to include in the grid search.
        seed: Random seed for reproducibility.
        plot_results: If True, call :func:`plot_grid_search_results` after fitting.

    Returns:
        A tuple ``(model, cluster_idxs)`` where *model* is the fitted ``GridSearchCV``
        object and *cluster_idxs* is an integer array of shape ``(n_samples,)`` with
        cluster labels from the best estimator.

    Raises:
        NotImplementedError: If *cv_metric* is not recognised.
    """
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
        "covariance_type": covariance_types,
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


def remove_clusters(cluster_idxs: np.ndarray, min_count: int,
                    invalid_value: int = -1) -> np.ndarray:
    """Remove clusters that contain fewer than *min_count* samples.

    Clusters that survive the threshold are re-labelled consecutively starting at 1.
    Samples belonging to removed clusters receive *invalid_value*.

    Args:
        cluster_idxs: 1-D integer array of cluster labels with shape ``(n_samples,)``.
        min_count: Minimum number of samples a cluster must have to be retained.
        invalid_value: Label assigned to samples whose cluster was removed.

    Returns:
        A new integer array of the same shape as *cluster_idxs* with re-labelled clusters.
    """
    new_cluster_idxs = np.full(cluster_idxs.size, invalid_value)

    new_cluster = 1
    for cluster, count in zip(*np.unique(cluster_idxs, return_counts=True)):
        if count > min_count:
            new_cluster_idxs[cluster_idxs == cluster] = new_cluster
            new_cluster += 1

    return new_cluster_idxs


def sort_clusters(traces: np.ndarray, cluster_idxs: np.ndarray,
                  invalid_value: int = -1) -> np.ndarray:
    """Sort clusters by correlation to the largest cluster.

    Args:
        traces: 2-D array of shape ``(n_samples, n_timepoints)`` used to compute
            per-cluster mean traces for correlation ranking.
        cluster_idxs: 1-D array of cluster labels of shape ``(n_samples,)``.
            Entries equal to *invalid_value* are ignored.
        invalid_value: Label used for unassigned or removed samples.

    Returns:
        A new integer array of the same shape as *cluster_idxs* with clusters
        re-ordered by descending correlation to the mean trace of the largest cluster.
    """
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
