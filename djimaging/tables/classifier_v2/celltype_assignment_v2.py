from abc import abstractmethod

import datajoint as dj
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from djimaging.tables.classifier_v2.rgc_classifier_v2 import load_classifier_from_file, check_classifier_dict
from djimaging.utils.baden16_utils import baden_cluster_id_to_group_id, baden_group_id_to_supergroup, \
    BADEN_CLUSTER_INFO, load_baden_data, baden_cluster_name_to_cluster_id
from djimaging.utils.dj_utils import merge_keys


class CelltypeAssignmentV2Template(dj.Computed):
    database = ""
    _baden_data_file = '/gpfs01/euler/data/Resources/Classifier/rgc_classifier_v2/RGCData_postprocessed.mat'
    __expected_classes = np.arange(1, 75 + 1)  # Expected classes for the classifier

    @property
    def definition(self) -> str:
        definition = """
        -> self.baden_trace_table
        -> self.classifier_table
        ---
        cluster_id :       tinyint unsigned      # cluster ID, ranging from 1 to 75
        group_id :         tinyint unsigned      # group ID, ranging from 1 to 46
        supergroup :       enum('OFF', 'ON-OFF', 'Fast ON', 'Slow ON', 'Unc. ON', 'Unc. SbC', 'dAC', 'error')
        prob_cluster :     float                 # probability of being in the given cluster
        prob_group :       float                 # aggregated probability of being in the given group
        prob_supergroup :  float                 # aggregated probability of being in the given supergroup
        prob_class :       float                 # aggregated probability of being the given cell class (RGC or dAC)
        probs_per_cluster : blob                 # probabilities for each cluster
        """
        return definition

    @property
    def key_source(self):
        """Return the key source combining classifier and field table projections."""
        try:
            return self.classifier_table.proj() * self.field_table.proj()
        except (AttributeError, TypeError):
            pass

    @property
    @abstractmethod
    def classifier_table(self) -> dj.Manual:
        pass

    @property
    @abstractmethod
    def baden_trace_table(self) -> dj.Computed:
        pass

    @property
    @abstractmethod
    def field_table(self) -> dj.Manual:
        pass

    @property
    @abstractmethod
    def roi_table(self) -> dj.Manual:
        pass

    def populate(
            self,
            *restrictions,
            keys=None,
            suppress_errors: bool = False,
            return_exception_objects: bool = False,
            reserve_jobs: bool = False,
            order: str = "original",
            limit=None,
            max_calls=None,
            display_progress: bool = False,
            processes: int = 1,
            make_kwargs=None,
    ) -> None:
        """Populate the table, loading the classifier once before iterating over keys.

        The classifier, chirp features, and bar features are loaded from the
        ``classifier_table`` entry and injected into each ``make`` call via
        ``make_kwargs``. Parallel processing is not supported.

        Args:
            *restrictions: DataJoint restrictions; must resolve to a single classifier entry.
            keys: Optional explicit list of keys to populate.
            suppress_errors: If ``True``, suppress errors during population.
            return_exception_objects: If ``True``, return exception objects instead of raising.
            reserve_jobs: If ``True``, use the job reservation mechanism.
            order: Population order, e.g. ``"original"`` or ``"random"``.
            limit: Maximum number of keys to populate.
            max_calls: Maximum number of ``make`` calls.
            display_progress: If ``True``, display a progress bar.
            processes: Number of parallel processes; must be 1.
            make_kwargs: Additional keyword arguments forwarded to ``make``. The keys
                ``'classifier'``, ``'chirp_feats'``, and ``'bar_feats'`` are reserved.

        Raises:
            NotImplementedError: If ``processes > 1``.
            ValueError: If ``make_kwargs`` contains a reserved key, if multiple classifiers
                match the restrictions, or if the classifier classes do not match the
                expected 75 Baden clusters.
        """
        if processes > 1:
            raise NotImplementedError(
                "Parallel processing is not implemented for this table."
            )

        if len(restrictions) == 0:
            restrictions = dict()

        if len(self.classifier_table & restrictions) > 1:
            raise ValueError(
                "Multiple classifiers found for the given restrictions. "
                "Please specify a single classifier.")

        if make_kwargs is None:
            make_kwargs = dict()

        for key in ['classifier', 'chirp_feats', 'bar_feats']:
            if key in make_kwargs:
                raise ValueError(
                    f"The '{key}' key is reserved and should not be provided in make_kwargs. "
                    "It will be automatically set based on the classifier_table.")

        classifier_file = (self.classifier_table & restrictions).fetch1('classifier_file')
        clf_dict = load_classifier_from_file(classifier_file)
        check_classifier_dict(clf_dict)

        make_kwargs['classifier'] = clf_dict['classifier']
        make_kwargs['chirp_feats'] = clf_dict['chirp_feats']
        make_kwargs['bar_feats'] = clf_dict['bar_feats']

        if not np.all(make_kwargs['classifier'].classes_ == self.__expected_classes):
            raise ValueError("The classifier's classes do not match the expected classes.")

        # populate
        super().populate(
            *restrictions,
            keys=keys,
            suppress_errors=suppress_errors,
            return_exception_objects=return_exception_objects,
            reserve_jobs=reserve_jobs,
            order=order,
            limit=limit,
            max_calls=max_calls,
            display_progress=display_progress,
            processes=processes,
            make_kwargs=make_kwargs
        )

    def make(self, key: dict, classifier, chirp_feats: np.ndarray, bar_feats: np.ndarray) -> None:
        """Classify all ROIs for the given key and insert the results into the table.

        Args:
            key: DataJoint key dictionary identifying the classifier and field to process.
            classifier: Trained scikit-learn classifier with ``predict_proba`` and ``classes_``.
            chirp_feats: Chirp feature basis matrix used for projection.
            bar_feats: Bar feature basis matrix used for projection.
        """
        roi_keys, chirps, bars, ds_pvalues, roi_size_um2s = self._fetch_data(key)
        if len(roi_keys) == 0:
            return
        probs = classify_cells(
            chirps, bars, ds_pvalues, roi_size_um2s,
            chirp_features=chirp_feats, bar_features=bar_feats, classifier=classifier)

        for roi_key, roi_probs in zip(roi_keys, probs):
            cluster_id, group_id, supergroup, prob_cluster, prob_group, prob_supergroup, prob_class = (
                baden16_cluster_probs_to_info(roi_probs))

            self.insert1(dict(**merge_keys(key, roi_key),
                              cluster_id=cluster_id, group_id=group_id, supergroup=supergroup,
                              prob_cluster=prob_cluster, prob_group=prob_group,
                              prob_supergroup=prob_supergroup, prob_class=prob_class,
                              probs_per_cluster=roi_probs))

    def _fetch_data(self, key: dict, restriction=None) -> tuple:
        """Fetch preprocessed traces and metadata for ROIs matching ``key``.

        Args:
            key: DataJoint key dictionary used to restrict the query.
            restriction: Additional DataJoint restriction applied on top of ``key``.

        Returns:
            A tuple ``(roi_keys, preproc_chirps, preproc_bars, bar_ds_pvalues, roi_size_um2s)``.
            The trace arrays are stacked into 2-D ``np.ndarray`` objects when ROIs are found;
            otherwise the raw (empty) arrays are returned.
        """
        if restriction is None:
            restriction = dict()

        roi_keys, preproc_chirps, preproc_bars, bar_ds_pvalues, roi_size_um2s = (
                (self.baden_trace_table & key & restriction) * self.roi_table).fetch(
            'KEY', 'preproc_chirp', 'preproc_bar', 'ds_pvalue', 'roi_size_um2')

        if len(roi_keys) > 0:
            preproc_chirps = np.vstack(preproc_chirps)
            preproc_bars = np.vstack(preproc_bars)

        return roi_keys, preproc_chirps, preproc_bars, bar_ds_pvalues, roi_size_um2s

    def plot_group_traces(self, cluster_id=None, cluster_name=None, group_id=None, min_prob: float = 0.0,
                          xlim_chirp=None, xlim_bar=None, plot_baden_data: bool = True) -> None:
        """Plot mean traces for cells assigned to a specific cluster or group.

        Exactly one of ``cluster_id``, ``cluster_name``, or ``group_id`` must be provided.
        The plot shows chirp traces, bar traces, DS p-value distributions, and ROI size
        distributions, optionally overlaid with matching Baden reference data.

        Args:
            cluster_id: Baden cluster ID (1–75) to select cells by.
            cluster_name: Baden cluster name; converted to ``cluster_id`` internally.
            group_id: Baden group ID (1–46) to select cells by.
            min_prob: Minimum probability threshold for including a cell.
            xlim_chirp: Optional x-axis limits for the chirp trace panel.
            xlim_bar: Optional x-axis limits for the bar trace panel.
            plot_baden_data: If ``True``, overlay traces from the Baden reference data.

        Raises:
            ValueError: If not exactly one of ``cluster_id``, ``cluster_name``, or
                ``group_id`` is provided.
        """
        if int(group_id is not None) + int(cluster_id is not None) + int(cluster_name is not None) != 1:
            raise ValueError("Provide exactly one of 'cluster_id', 'cluster_name', or 'group_id'.")

        df = self.fetch(format='frame')
        groups = df.groupby('classifier_id')

        for classifier_id, df_group in groups:
            classifier_key = dict(classifier_id=classifier_id)

            self._plot_group_traces_key(classifier_key=classifier_key,
                                        min_prob=min_prob,
                                        cluster_id=cluster_id, cluster_name=cluster_name, group_id=group_id,
                                        xlim_chirp=xlim_chirp, xlim_bar=xlim_bar, plot_baden_data=plot_baden_data)

    def _plot_group_traces_key(self, classifier_key: dict, min_prob: float,
                               cluster_id=None, cluster_name=None, group_id=None,
                               xlim_chirp=None, xlim_bar=None, plot_baden_data: bool = True,
                               n_traces_max: int = 20) -> None:
        """Plot traces for a single classifier entry restricted to one cluster or group.

        Args:
            classifier_key: DataJoint key dict selecting a single classifier entry.
            min_prob: Minimum probability threshold for including a cell.
            cluster_id: Baden cluster ID (1–75) to select cells by.
            cluster_name: Baden cluster name; converted to ``cluster_id`` internally.
            group_id: Baden group ID (1–46) to select cells by.
            xlim_chirp: Optional x-axis limits for the chirp trace panel.
            xlim_bar: Optional x-axis limits for the bar trace panel.
            plot_baden_data: If ``True``, overlay traces from the Baden reference data.
            n_traces_max: Maximum number of individual traces to draw (randomly sampled).

        Raises:
            ValueError: If not exactly one of ``cluster_id``, ``cluster_name``, or
                ``group_id`` is provided, or if neither ``cluster_id`` nor ``group_id``
                can be resolved.
        """
        if int(group_id is not None) + int(cluster_id is not None) + int(cluster_name is not None) != 1:
            raise ValueError("Provide exactly one of 'cluster_id', 'cluster_name', or 'group_id'.")

        if cluster_name is not None:
            cluster_id = baden_cluster_name_to_cluster_id(cluster_name)

        if cluster_id is not None:
            restriction = f"cluster_id={cluster_id} and prob_cluster >= {min_prob}"
        elif group_id is not None:
            restriction = f"group_id={group_id} and prob_group >= {min_prob}"
        else:
            raise ValueError("Either 'cluster_id' or 'group_id' must be specified.")

        roi_keys, preproc_chirps, preproc_bars, ds_pvalues, soma_um2s = self._fetch_data(
            key=classifier_key, restriction=self & classifier_key & restriction)

        fig, axs = plt.subplots(1, 4, figsize=(12, 3),
                                sharex='col', sharey='col',
                                gridspec_kw=dict(width_ratios=[1, 0.6, 0.3, 0.3]))

        if plot_baden_data:
            b_cis, b_gis, b_sgs, b_chirps, b_chirp_qi, b_bars, b_mb_qi, b_dsi, b_ds_pvalues, b_soma_um2s = \
                load_baden_data(self._baden_data_file, quality_filter=True)

            if cluster_id is not None:
                b_idxs = b_cis == cluster_id
            else:
                b_idxs = b_gis == group_id

            self._plot_group_traces_row(axs, b_chirps[b_idxs], b_bars[b_idxs], b_ds_pvalues[b_idxs],
                                        b_soma_um2s[b_idxs],
                                        is_baden_data=True, n_traces_max=n_traces_max)

            chirp_cc = np.corrcoef(np.mean(preproc_chirps, axis=0), np.mean(b_chirps[b_idxs], axis=0))[0, 1]
            axs[0].set(title=f"cc(mu_a, mu_b)={chirp_cc:.2f}", xlim=xlim_chirp)

            bar_ccs = np.corrcoef(np.mean(preproc_bars, axis=0), np.mean(b_bars[b_idxs], axis=0))[0, 1]
            axs[1].set(title=f"cc(mu_a, mu_b)={bar_ccs:.2f}", xlim=xlim_bar)

        self._plot_group_traces_row(axs, preproc_chirps, preproc_bars, ds_pvalues, soma_um2s,
                                    is_baden_data=False, n_traces_max=n_traces_max)

        axs[0].legend(loc='upper right', bbox_to_anchor=(-0.1, 1), fontsize='small')

        plt.tight_layout()

    @staticmethod
    def _plot_group_traces_row(axs, chirps: np.ndarray, bars: np.ndarray,
                                ds_pvalues: np.ndarray, roi_sizes: np.ndarray,
                                is_baden_data: bool = False, n_traces_max: int = 20) -> None:
        """Draw one row of trace and histogram panels onto ``axs``.

        Args:
            axs: Array of four Matplotlib axes (chirp, bar, DS p-value, ROI size).
            chirps: Chirp traces to plot, shape ``(n_cells, n_timepoints)``.
            bars: Bar traces to plot, shape ``(n_cells, n_timepoints)``.
            ds_pvalues: Direction-selectivity p-values, shape ``(n_cells,)``.
            roi_sizes: ROI areas in square micrometres, shape ``(n_cells,)``.
            is_baden_data: If ``True``, use grey styling and draw histograms on a twin axis.
            n_traces_max: Maximum number of individual traces to draw (randomly sampled).
        """
        if is_baden_data:
            mean_kws = dict(c='k', lw=2, alpha=0.5, zorder=1)
            trace_kes = dict(lw=0.1, c='gray', alpha=0.2, zorder=-200)
            hist_kws = dict(color='gray', alpha=0.5)
        else:
            mean_kws = dict(c='darkred', lw=2, zorder=2)
            trace_kes = dict(lw=0.1, c='r', alpha=0.2, zorder=-100)
            hist_kws = dict(color='r', alpha=0.7)

        idxs = np.random.permutation(np.arange(len(roi_sizes)))
        if len(idxs) > n_traces_max:
            idxs = idxs[:n_traces_max]

        ax = axs[0]
        ax.plot(chirps[idxs].T, **trace_kes)
        ax.plot(np.mean(chirps, axis=0), **mean_kws, label='Baden' if is_baden_data else 'New')

        ax = axs[1]
        ax.plot(bars[idxs].T, **trace_kes)
        ax.plot(np.mean(bars, axis=0), **mean_kws)

        ax = axs[2]
        if is_baden_data:
            ax = ax.twinx()
        ax.hist(ds_pvalues, bins=np.linspace(0, 1, 21), **hist_kws)
        ax.set_title('bar_ds_pvalues')
        ax.set_ylabel('Count (Baden)' if is_baden_data else 'Count (New)', fontsize='small')

        ax = axs[3]
        if is_baden_data:
            ax = ax.twinx()
        ax.hist(roi_sizes, bins=np.linspace(0, np.maximum(500, roi_sizes.max()), 21), **hist_kws)
        ax.set_title('roi_size_um2s')
        ax.set_ylabel('Count (Baden)' if is_baden_data else 'Count (New)', fontsize='small')

    def plot_type_distribution(self, min_prob: float, level: str = 'group', plot_baden_data: bool = True):
        """
        Plot the distribution of cell types (clusters, groups, or supergroups) based on the classifier results.
        Parameters
        ----------
        min_prob : float
            Minimum probability threshold for including a cell type in the plot.
        level : str
            Level of classification to plot. Can be 'cluster', 'group', or 'super'.
        plot_baden_data : bool
            Whether to plot the Baden data for comparison. Defaults to True.
        """
        if level not in ['cluster', 'group', 'super']:
            raise ValueError("Invalid level. Choose from 'cluster', 'group', or 'super'.")

        df = self.fetch(format='frame').reset_index()
        groups = df.groupby('classifier_id')

        for classifier_id, df_classifier in groups:
            self._plot_type_distribution(df=df_classifier, min_prob=min_prob, level=level,
                                         plot_baden_data=plot_baden_data)

        plt.tight_layout()
        plt.show()

    def _plot_type_distribution(self, df, min_prob: float, level: str, plot_baden_data: bool) -> None:
        """Render the cell-type count plot for a single classifier entry.

        Args:
            df: DataFrame for one classifier, as returned by ``fetch(format='frame')``.
            min_prob: Minimum probability threshold for including a cell.
            level: Label level; one of ``'cluster'``, ``'group'``, or ``'super'``.
            plot_baden_data: If ``True``, overlay counts from the Baden reference data.

        Raises:
            NotImplementedError: If ``level`` is not one of the supported values.
        """
        if level == 'cluster':
            order = np.append(-1, np.arange(1, 75 + 1))
            fig, ax = plt.subplots(figsize=(20, 4))
        elif level == 'group':
            order = np.append(-1, np.arange(1, 46 + 1))
            fig, ax = plt.subplots(figsize=(12, 4))
        elif level == 'super':
            order = ['OFF', 'ON-OFF', 'Fast ON', 'Slow ON', 'Unc. ON', 'Unc. SbC', 'dAC', 'error']
            fig, ax = plt.subplots(figsize=(8, 4))
        else:
            raise NotImplementedError(f"Level '{level}' is not implemented.")

        if plot_baden_data:
            b_cis, b_gis, b_sgs = load_baden_data(self._baden_data_file, quality_filter=True)[:3]

            if level == 'cluster':
                b_dist = b_cis
            elif level == 'group':
                b_dist = b_gis
            elif level == 'super':
                b_dist = b_sgs
            else:
                raise NotImplementedError(f"Level '{level}' is not implemented.")

            ax_baden = ax.twinx()
            sns.countplot(ax=ax_baden, x=b_dist, order=order, color='gray', alpha=0.5, label='Baden')
            ax_baden.set(ylabel='Count (Baden)')
            ax.legend(loc='upper right')

        if level == 'cluster':
            df_filt = df[df['prob_cluster'] >= min_prob]
            df_col = 'cluster_id'
        elif level == 'group':
            df_filt = df[df['prob_group'] >= min_prob]
            df_col = 'group_id'
        elif level == 'super':
            df_filt = df[df['prob_supergroup'] >= min_prob]
            df_col = 'supergroup'
        else:
            raise NotImplementedError(f"Level '{level}' is not implemented.")

        sns.countplot(ax=ax, data=df_filt, x=df_col, order=order, color='red', alpha=0.5, label='New')
        ax.legend(loc='upper left')
        ax.set(xlabel=f'{level.capitalize()} ID', ylabel='Count (New)')


def classify_cells(
        preproc_chirps: np.ndarray,
        preproc_bars: np.ndarray,
        bar_ds_pvalues: np.ndarray,
        roi_size_um2s: np.ndarray,
        chirp_features: np.ndarray,
        bar_features: np.ndarray,
        classifier,
) -> np.ndarray:
    """Classify multiple cells and return per-cluster probability arrays.

    Args:
        preproc_chirps: Preprocessed chirp traces, shape ``(n_cells, n_chirp_timepoints)``.
        preproc_bars: Preprocessed bar traces, shape ``(n_cells, n_bar_timepoints)``.
        bar_ds_pvalues: Direction-selectivity p-values for each cell, shape ``(n_cells,)``.
        roi_size_um2s: ROI areas in square micrometres for each cell, shape ``(n_cells,)``.
        chirp_features: Chirp feature basis matrix.
        bar_features: Bar feature basis matrix.
        classifier: Trained scikit-learn classifier with ``predict_proba``.

    Returns:
        Probability matrix of shape ``(n_cells, n_classes)``.
    """
    features, feature_names = extract_features(
        preproc_chirps, preproc_bars, bar_ds_pvalues, roi_size_um2s, chirp_features, bar_features)
    probs = classifier.predict_proba(features)
    return probs


def baden16_cluster_probs_to_info(probs: np.ndarray) -> tuple:
    """Convert a 75-element per-cluster probability vector to higher-level label info.

    Args:
        probs: Per-cluster probability array of length 75, corresponding to Baden et al.
            (2016) clusters 1–75.

    Returns:
        A tuple ``(cluster_id, group_id, supergroup, prob_cluster, prob_group,
        prob_supergroup, prob_class)`` where:

        - ``cluster_id`` (int): 1-indexed cluster with the highest probability.
        - ``group_id`` (int): Group ID derived from ``cluster_id``.
        - ``supergroup`` (str): Supergroup name derived from ``group_id``.
        - ``prob_cluster`` (float): Probability of the assigned cluster.
        - ``prob_group`` (float): Summed probability of all clusters in the group.
        - ``prob_supergroup`` (float): Summed probability of all clusters in the supergroup.
        - ``prob_class`` (float): Probability of the cell belonging to its inferred class
          (RGC or dAC).

    Raises:
        ValueError: If ``probs`` does not have exactly 75 elements.
    """
    if len(probs) != 75:
        raise ValueError(f"Expected 75 probabilities corresponding to 75 Baden clusters, got {len(probs)}.")

    cluster_id = np.argmax(probs) + 1  # Cluster IDs are 1-indexed
    group_id = baden_cluster_id_to_group_id(cluster_id)
    supergroup = baden_group_id_to_supergroup(group_id)
    prob_cluster = probs[cluster_id - 1]

    group_ids = BADEN_CLUSTER_INFO[:, 2].astype(int)
    supergroups = BADEN_CLUSTER_INFO[:, 3].astype(str)

    prob_group = np.sum(probs[group_ids == group_id])
    prob_supergroup = np.sum(probs[supergroups == supergroup])
    prob_rgc = np.sum(probs[supergroups != 'dAC'])
    prob_class = (1. - prob_rgc) if supergroup == 'dAC' else prob_rgc

    return cluster_id, group_id, supergroup, prob_cluster, prob_group, prob_supergroup, prob_class


def extract_features(
        preproc_chirps: np.ndarray,
        preproc_bars: np.ndarray,
        bar_ds_pvalues: np.ndarray,
        roi_size_um2s: np.ndarray,
        chirp_features: np.ndarray,
        bar_features: np.ndarray,
) -> tuple:
    """Build the feature matrix and corresponding feature names for a set of cells.

    Projects chirp and bar traces onto their respective feature bases and appends
    the DS p-value and ROI size as additional scalar features.

    Args:
        preproc_chirps: Preprocessed chirp traces, shape ``(n_cells, n_chirp_timepoints)``.
        preproc_bars: Preprocessed bar traces, shape ``(n_cells, n_bar_timepoints)``.
        bar_ds_pvalues: Direction-selectivity p-values for each cell, shape ``(n_cells,)``.
        roi_size_um2s: ROI areas in square micrometres for each cell, shape ``(n_cells,)``.
        chirp_features: Chirp feature basis matrix of shape
            ``(n_chirp_timepoints, n_chirp_features)``.
        bar_features: Bar feature basis matrix of shape
            ``(n_bar_timepoints, n_bar_features)``.

    Returns:
        A tuple ``(features, feature_names)`` where ``features`` is an ``np.ndarray`` of
        shape ``(n_cells, n_features)`` and ``feature_names`` is a list of strings.
    """
    features = np.concatenate([
        np.dot(preproc_chirps, chirp_features),
        np.dot(preproc_bars, bar_features),
        bar_ds_pvalues[:, np.newaxis],
        roi_size_um2s[:, np.newaxis]
    ], axis=-1)

    feature_names = [f'chirp_{i}' for i in range(chirp_features.shape[1])] + \
                    [f'bar_{i}' for i in range(bar_features.shape[1])] + ['bar_ds_pvalue', 'roi_size_um2']

    return features, feature_names
