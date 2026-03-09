import pickle as pkl
import warnings
from abc import abstractmethod
from typing import Mapping

import datajoint as dj
import numpy as np
from cached_property import cached_property
from matplotlib import pyplot as plt

from djimaging.utils.baden16_utils import load_baden_data
from djimaging.utils.dj_utils import merge_keys


def extract_feature(
        preproc_chirp: np.ndarray,
        preproc_bar: np.ndarray,
        bar_ds_pvalue: float,
        roi_size_um2: float,
        chirp_features: np.ndarray,
        bar_features: np.ndarray,
) -> np.ndarray:
    """Extract a feature vector for a single cell by projecting traces onto feature bases.

    Args:
        preproc_chirp: Preprocessed chirp trace for one cell.
        preproc_bar: Preprocessed bar trace for one cell.
        bar_ds_pvalue: Direction-selectivity p-value from the bar stimulus.
        roi_size_um2: ROI area in square micrometres.
        chirp_features: Feature basis matrix for the chirp stimulus.
        bar_features: Feature basis matrix for the bar stimulus.

    Returns:
        Concatenated feature vector combining chirp projections, bar projections,
        the DS p-value and ROI size.
    """
    feature_activation_chirp = np.dot(preproc_chirp, chirp_features)
    feature_activation_bar = np.dot(preproc_bar, bar_features)

    feature = np.concatenate(
        [feature_activation_chirp, feature_activation_bar, np.array([bar_ds_pvalue]), np.array([roi_size_um2])])

    return feature


def extract_features(
        preproc_chirps: np.ndarray,
        preproc_bars: np.ndarray,
        bar_ds_pvalues: np.ndarray,
        roi_size_um2s: np.ndarray,
        chirp_features: np.ndarray,
        bar_features: np.ndarray,
) -> np.ndarray:
    """Extract feature vectors for multiple cells by projecting traces onto feature bases.

    Args:
        preproc_chirps: Preprocessed chirp traces, shape ``(n_cells, n_chirp_timepoints)``.
        preproc_bars: Preprocessed bar traces, shape ``(n_cells, n_bar_timepoints)``.
        bar_ds_pvalues: Direction-selectivity p-values for each cell, shape ``(n_cells,)``.
        roi_size_um2s: ROI areas in square micrometres for each cell, shape ``(n_cells,)``.
        chirp_features: Feature basis matrix for the chirp stimulus.
        bar_features: Feature basis matrix for the bar stimulus.

    Returns:
        Feature matrix of shape ``(n_cells, n_features)``.
    """
    features = np.concatenate([
        np.dot(preproc_chirps, chirp_features),
        np.dot(preproc_bars, bar_features),
        bar_ds_pvalues[:, np.newaxis],
        roi_size_um2s[:, np.newaxis]
    ], axis=-1)

    return features


def classify_cell(
        preproc_chirp: np.ndarray,
        preproc_bar: np.ndarray,
        bar_ds_pvalue: float,
        roi_size_um2: float,
        chirp_features: np.ndarray,
        bar_features: np.ndarray,
        classifier,
) -> tuple:
    """Classify a single cell and return its predicted label and confidence scores.

    Args:
        preproc_chirp: Preprocessed chirp trace for one cell.
        preproc_bar: Preprocessed bar trace for one cell.
        bar_ds_pvalue: Direction-selectivity p-value from the bar stimulus.
        roi_size_um2: ROI area in square micrometres.
        chirp_features: Feature basis matrix for the chirp stimulus.
        bar_features: Feature basis matrix for the bar stimulus.
        classifier: Trained scikit-learn classifier with ``predict_proba`` and ``classes_``.

    Returns:
        A tuple ``(cell_label, confidence_scores)`` where ``cell_label`` is the predicted
        class and ``confidence_scores`` is the array of per-class probabilities.
    """
    feature = extract_feature(
        preproc_chirp, preproc_bar, bar_ds_pvalue, roi_size_um2, chirp_features, bar_features)
    confidence_scores = classifier.predict_proba(feature[np.newaxis, :]).flatten()
    cell_label = confidence_scores.argmax(axis=1)
    cell_label = classifier.classes_[cell_label][0]
    return cell_label, confidence_scores


def classify_cells(
        preproc_chirps: np.ndarray,
        preproc_bars: np.ndarray,
        bar_ds_pvalues: np.ndarray,
        roi_size_um2s: np.ndarray,
        chirp_features: np.ndarray,
        bar_features: np.ndarray,
        classifier,
) -> tuple:
    """Classify multiple cells and return their predicted labels and confidence scores.

    Args:
        preproc_chirps: Preprocessed chirp traces, shape ``(n_cells, n_chirp_timepoints)``.
        preproc_bars: Preprocessed bar traces, shape ``(n_cells, n_bar_timepoints)``.
        bar_ds_pvalues: Direction-selectivity p-values for each cell, shape ``(n_cells,)``.
        roi_size_um2s: ROI areas in square micrometres for each cell, shape ``(n_cells,)``.
        chirp_features: Feature basis matrix for the chirp stimulus.
        bar_features: Feature basis matrix for the bar stimulus.
        classifier: Trained scikit-learn classifier with ``predict_proba`` and ``classes_``.

    Returns:
        A tuple ``(cell_labels, confidence_scores)`` where ``cell_labels`` is an array of
        predicted classes and ``confidence_scores`` is an array of per-class probabilities,
        shape ``(n_cells, n_classes)``.
    """
    features = extract_features(
        preproc_chirps, preproc_bars, bar_ds_pvalues, roi_size_um2s, chirp_features, bar_features)
    confidence_scores = classifier.predict_proba(features)
    cell_labels = confidence_scores.argmax(axis=1)
    cell_labels = classifier.classes_[cell_labels]
    return cell_labels, confidence_scores


class CelltypeAssignmentTemplate(dj.Computed):
    database = ""

    @property
    def definition(self) -> str:
        definition = """
        # Feature-based cluster assignment of cells
        -> self.baden_trace_table
        -> self.classifier_table
        ---
        cell_label:      int         # predicted label with highest probability. Meaning of label depends on classifier
        max_confidence:  float       # confidence score for assigned cell_label, can be celltype, supergroup etc.
        confidence:      blob        # confidence scores (probabilities) for all celltypes
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
    def classifier_training_data_table(self) -> dj.Manual:
        pass

    @property
    @abstractmethod
    def classifier_table(self) -> dj.Computed:
        pass

    @property
    @abstractmethod
    def os_ds_table(self) -> dj.Computed:
        pass

    @property
    @abstractmethod
    def roi_table(self) -> dj.Manual:
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
    def current_model_key(self):
        return self._current_model_key

    @current_model_key.setter
    def current_model_key(self, value: Mapping):
        if hasattr(self, "_current_model_key"):
            if value == self._current_model_key:
                pass
            else:
                print("Updating current key and invalidating cache")
                key = "classifier"
                if key in self.__dict__.keys():
                    del self.__dict__[key]
                if not (self._current_model_key["training_data_hash"] == value["training_data_hash"]):
                    del self.__dict__["bar_features"]
                    del self.__dict__["chirp_features"]
                    del self.__dict__["baden_data"]
                self._current_model_key = value
        else:
            self._current_model_key = value

    @cached_property
    def classifier(self):
        """Load and cache the trained classifier from its stored file path."""
        model_path = (self.classifier_table() & self.current_model_key).fetch1('classifier_file')
        with open(model_path, "rb") as f:
            model = pkl.load(f)
        return model

    @cached_property
    def bar_features(self) -> np.ndarray:
        """Load and cache the bar feature basis matrix from its stored file path."""
        features_bar_file = (self.classifier_training_data_table() & self.current_model_key).fetch1(
            "bar_feats_file")
        features_bar = np.load(features_bar_file)
        return features_bar

    @cached_property
    def chirp_features(self) -> np.ndarray:
        """Load and cache the chirp feature basis matrix from its stored file path."""
        features_chirp_file = (self.classifier_training_data_table() & self.current_model_key).fetch1(
            "chirp_feats_file")
        features_chirp = np.load(features_chirp_file)
        return features_chirp

    def populate(
            self, *restrictions, suppress_errors: bool = False,
            return_exception_objects: bool = False, reserve_jobs: bool = False,
            order: str = "original", limit=None, max_calls=None,
            display_progress: bool = False, processes: int = 1, make_kwargs=None,
    ) -> None:
        """Populate the table, enforcing single-process execution.

        Args:
            *restrictions: DataJoint restrictions passed to the parent ``populate`` call.
            suppress_errors: If True, suppress errors during population.
            return_exception_objects: If True, return exception objects instead of raising.
            reserve_jobs: If True, use the job reservation mechanism.
            order: Population order, e.g. ``"original"`` or ``"random"``.
            limit: Maximum number of keys to populate.
            max_calls: Maximum number of ``make`` calls.
            display_progress: If True, display a progress bar.
            processes: Number of parallel processes. Values greater than 1 are not
                supported and will emit a warning.
            make_kwargs: Additional keyword arguments forwarded to ``make``.
        """
        if processes > 1:
            warnings.warn('Parallel processing not implemented!')
        super().populate(
            *restrictions,
            suppress_errors=suppress_errors, return_exception_objects=return_exception_objects,
            reserve_jobs=reserve_jobs, order=order, limit=limit, max_calls=max_calls,
            display_progress=display_progress, processes=1, make_kwargs=make_kwargs)

    def make(self, key: dict) -> None:
        """Classify all ROIs for the given key and insert results into the table.

        Args:
            key: DataJoint key dictionary identifying the classifier and field to process.
        """
        roi_keys, preproc_chirps, preproc_bars, bar_ds_pvalues, roi_size_um2s = self._fetch_data(key)

        if roi_keys is None:
            return

        cell_labels, confidence_scores = self._classify_cells(
            preproc_chirps, preproc_bars, bar_ds_pvalues, roi_size_um2s)

        for roi_key, cell_label, confidence_scores_i in zip(roi_keys, cell_labels, confidence_scores):
            self.insert1(dict(**merge_keys(key, roi_key), cell_label=cell_label,
                              confidence=confidence_scores_i, max_confidence=np.max(confidence_scores_i)))

    def _fetch_data(self, key: dict, restriction=None) -> tuple:
        """Fetch preprocessed traces and metadata for the ROIs matching ``key``.

        Args:
            key: DataJoint key dictionary used to restrict the query.
            restriction: Additional DataJoint restriction applied on top of ``key``.

        Returns:
            A tuple ``(roi_keys, preproc_chirps, preproc_bars, bar_ds_pvalues, roi_size_um2s)``.
            Returns ``(None, None, None, None, None)`` when no matching ROIs are found.
        """
        if restriction is None:
            restriction = dict()

        self.current_model_key = dict(
            classifier_params_hash=key["classifier_params_hash"],
            training_data_hash=key["training_data_hash"])

        roi_keys = (self.baden_trace_table & key & restriction).fetch('KEY')
        if len(roi_keys) == 0:
            return None, None, None, None, None

        if self.os_ds_table is not None:
            data_tab = (self.baden_trace_table & key & restriction) * self.os_ds_table * self.roi_table
        else:
            data_tab = (self.baden_trace_table & key & restriction) * self.roi_table

        preproc_chirps, preproc_bars, bar_ds_pvalues, roi_size_um2s = data_tab.fetch(
            'preproc_chirp', 'preproc_bar', 'ds_pvalue', 'roi_size_um2')

        preproc_chirps = np.vstack(preproc_chirps)
        preproc_bars = np.vstack(preproc_bars)

        return roi_keys, preproc_chirps, preproc_bars, bar_ds_pvalues, roi_size_um2s

    def _classify_cells(
            self,
            preproc_chirps: np.ndarray,
            preproc_bars: np.ndarray,
            bar_ds_pvalues: np.ndarray,
            roi_size_um2s: np.ndarray,
    ) -> tuple:
        """Run the cached classifier on pre-fetched feature arrays.

        Args:
            preproc_chirps: Preprocessed chirp traces, shape ``(n_cells, n_timepoints)``.
            preproc_bars: Preprocessed bar traces, shape ``(n_cells, n_timepoints)``.
            bar_ds_pvalues: Direction-selectivity p-values, shape ``(n_cells,)``.
            roi_size_um2s: ROI areas in square micrometres, shape ``(n_cells,)``.

        Returns:
            A tuple ``(cell_labels, confidence_scores)``.
        """
        cell_labels, confidence_scores = classify_cells(
            preproc_chirps, preproc_bars, bar_ds_pvalues, roi_size_um2s,
            self.chirp_features, self.bar_features, self.classifier)
        return cell_labels, confidence_scores

    def plot(self, threshold_confidence: float, classifier_level: str) -> None:
        """Plot per-group histograms of predicted cell labels.

        Args:
            threshold_confidence: Minimum confidence required for a cell to be assigned a label.
                Cells below this threshold are shown as label ``-1``.
            classifier_level: Label level to display; one of ``'cluster'``, ``'group'``,
                or ``'super'``.
        """
        df = self.fetch(format='frame')
        groups = df.groupby(['training_data_hash', 'classifier_params_hash', 'preprocess_id'])

        fig, axs = plt.subplots(len(groups), 1, figsize=(12, 3 * len(groups)), squeeze=False)
        axs = axs.flatten()

        for ax, ((tdh, cph, pid), df_group) in zip(axs, groups):
            self._plot_key(ax=ax, df=df_group,
                           classifier_params_hash=cph, training_data_hash=tdh, preprocess_id=pid,
                           threshold_confidence=threshold_confidence,
                           classifier_level=classifier_level)

        plt.tight_layout()
        plt.show()

    def _plot_key(self, ax, df, classifier_params_hash: str, training_data_hash: str,
                  preprocess_id, threshold_confidence: float, classifier_level: str) -> None:
        """Render the cell-label histogram for a single (hash, preprocess_id) group.

        Args:
            ax: Matplotlib axes on which to draw.
            df: DataFrame subset for this group, as returned by ``fetch(format='frame')``.
            classifier_params_hash: Hash identifying the classifier parameter set.
            training_data_hash: Hash identifying the training data set.
            preprocess_id: Preprocessing identifier for this group.
            threshold_confidence: Cells with confidence below this value are labelled ``-1``.
            classifier_level: Label level; one of ``'cluster'``, ``'group'``, or ``'super'``.
        """
        celltypes = df.apply(
            lambda row: row['cell_label'] if row["max_confidence"] > threshold_confidence else -1, axis=1)

        self.current_model_key = dict(classifier_params_hash=classifier_params_hash,
                                      training_data_hash=training_data_hash)
        b_features, b_c_labels, b_g_labels, b_s_labels = self.classifier_training_data_table().get_training_data(
            self.current_model_key)
        if classifier_level == 'cluster':
            b_celltypes = b_c_labels
        elif classifier_level == 'group':
            b_celltypes = b_g_labels
        elif classifier_level == 'super':
            b_celltypes = b_s_labels
        else:
            raise NotImplementedError(classifier_level)

        ax.hist(celltypes[celltypes > 0], bins=np.arange(b_celltypes.min(), b_celltypes.max() + 1, 0.5),
                align='left')
        ax.set(ylabel='Count', xlabel='cell_label', xticks=np.arange(b_celltypes.min(), b_celltypes.max() + 1))
        ax.set_title(
            f"training_data_hash={training_data_hash}\n"
            f"classifier_params_hash={classifier_params_hash}\n"
            f"preprocess_id={preprocess_id}\n"
            f"{np.sum(celltypes == -1)}={np.mean(celltypes == -1):.0%} were not classified", loc='left')

    def plot_group_traces(self, threshold_confidence: float, classifier_level: str,
                          celltypes=None, n_celltypes_max: int = 32,
                          xlim_chirp=None, xlim_bar=None, plot_baden_data: bool = True) -> None:
        """Plot mean traces per cell type grouped by classifier and preprocessing settings.

        Args:
            threshold_confidence: Minimum confidence required to include a cell.
            classifier_level: Label level; one of ``'cluster'``, ``'group'``, or ``'super'``.
            celltypes: Explicit list of cell-type labels to plot. If ``None``, all
                cell types present in the data are used.
            n_celltypes_max: Maximum number of cell types to display. Excess types are
                chosen randomly.
            xlim_chirp: Optional x-axis limits for the chirp trace panel.
            xlim_bar: Optional x-axis limits for the bar trace panel.
            plot_baden_data: If ``True``, overlay traces from the Baden training data.
        """
        df = self.fetch(format='frame')
        groups = df.groupby(['training_data_hash', 'classifier_params_hash', 'preprocess_id'])

        for (tdh, cph, pid), df_group in groups:
            self._plot_group_traces_key(classifier_params_hash=cph, training_data_hash=tdh, preprocess_id=pid,
                                        threshold_confidence=threshold_confidence,
                                        classifier_level=classifier_level,
                                        celltypes=celltypes, n_celltypes_max=n_celltypes_max,
                                        xlim_chirp=xlim_chirp, xlim_bar=xlim_bar, plot_baden_data=plot_baden_data)

    def _plot_group_traces_key(self, threshold_confidence: float, classifier_level: str,
                               classifier_params_hash: str, training_data_hash: str, preprocess_id,
                               celltypes=None, n_celltypes_max: int = 32, xlim_chirp=None, xlim_bar=None,
                               plot_baden_data: bool = True) -> None:
        """Plot mean traces per cell type for a single (hash, preprocess_id) group.

        Args:
            threshold_confidence: Minimum confidence required to include a cell.
            classifier_level: Label level; one of ``'cluster'``, ``'group'``, or ``'super'``.
            classifier_params_hash: Hash identifying the classifier parameter set.
            training_data_hash: Hash identifying the training data set.
            preprocess_id: Preprocessing identifier for this group.
            celltypes: Explicit list of cell-type labels to plot. If ``None``, all present
                cell types are used.
            n_celltypes_max: Maximum number of cell types to display.
            xlim_chirp: Optional x-axis limits for the chirp trace panel.
            xlim_bar: Optional x-axis limits for the bar trace panel.
            plot_baden_data: If ``True``, overlay traces from the Baden training data.
        """
        key = dict(classifier_params_hash=classifier_params_hash,
                   training_data_hash=training_data_hash,
                   preprocess_id=preprocess_id)

        # Get new data
        roi_keys, preproc_chirps, preproc_bars, bar_ds_pvalues, roi_size_um2s = self._fetch_data(
            key=key, restriction=(self & f'max_confidence>={threshold_confidence}'))
        data_celltypes = (self & roi_keys).fetch('cell_label')

        # Get training data
        self.current_model_key = dict(classifier_params_hash=classifier_params_hash,
                                      training_data_hash=training_data_hash)
        if celltypes is not None:
            celltypes = sorted(set(celltypes).intersection(set(data_celltypes)))
        else:
            celltypes = sorted(set(data_celltypes))

        if n_celltypes_max is not None and len(celltypes) > n_celltypes_max:
            celltypes = sorted(np.random.choice(celltypes, n_celltypes_max, replace=False))

        fig, axs = plt.subplots(len(celltypes), 4,
                                figsize=(12, np.minimum(2 * len(celltypes), 20)),
                                sharex='col', sharey='col', squeeze=False,
                                gridspec_kw=dict(width_ratios=[1, 0.6, 0.3, 0.3]))

        bar_ds_pvalues_bins = np.linspace(np.min(bar_ds_pvalues), np.max(bar_ds_pvalues), 51)
        roi_size_um2s_bins = np.linspace(np.min(roi_size_um2s), np.max(roi_size_um2s), 51)

        if plot_baden_data:
            baden_data_file = (self.classifier_training_data_table() & key).fetch1('baden_data_file')
            b_c_labels, b_g_labels, b_s_labels, b_c_traces, b_c_qi, b_mb_traces, b_mb_qi, b_mb_dsi, b_mb_dp, b_soma_um2 = \
                load_baden_data(baden_data_file)

            if classifier_level == 'cluster':
                b_celltypes = b_c_labels
            elif classifier_level == 'group':
                b_celltypes = b_g_labels
            elif classifier_level == 'super':
                b_celltypes = b_s_labels
            else:
                raise NotImplementedError(classifier_level)

            for ax_row, celltype in zip(axs, celltypes):

                if not np.any(data_celltypes == celltype):
                    continue

                ax = ax_row[0]
                b_chirps = b_c_traces[b_celltypes == celltype]
                b_mean_chirp = np.mean(b_chirps, axis=0)
                ax.plot(b_chirps.T, lw=0.5, c='gray', alpha=0.5, zorder=-200)
                ax.plot(b_mean_chirp, c='k', lw=2, alpha=0.5, zorder=1)

                mean_chirp = np.mean(preproc_chirps[data_celltypes == celltype], axis=0)
                ax.set(title=f"cc={np.corrcoef(mean_chirp, b_mean_chirp)[0, 1]:.2f}", xlim=xlim_chirp)

                ax = ax_row[1]
                b_bars = b_mb_traces[b_celltypes == celltype]
                b_mean_bar = np.mean(b_bars, axis=0)
                ax.plot(b_bars.T, lw=0.5, c='gray', alpha=0.5, zorder=-200)
                ax.plot(b_mean_bar, c='k', lw=2, alpha=0.5, zorder=1)

                mean_bar = np.mean(preproc_bars[data_celltypes == celltype], axis=0)
                ax.set(title=f"cc={np.corrcoef(mean_bar, b_mean_bar)[0, 1]:.2f}", xlim=xlim_bar)

                ax = ax_row[2].twinx()
                ax.hist(b_mb_dp[b_celltypes == celltype], bar_ds_pvalues_bins, color='gray', alpha=0.5)

                ax = ax_row[3].twinx()
                ax.hist(b_soma_um2[b_celltypes == celltype], bins=roi_size_um2s_bins, color='gray', alpha=0.5)

        for ax_row, celltype in zip(axs, celltypes):
            ax_row[0].set_ylabel(celltype)

            if not np.any(data_celltypes == celltype):
                continue

            ax = ax_row[0]
            ct_chirps = preproc_chirps[data_celltypes == celltype]
            ax.plot(ct_chirps.T, lw=0.5, c='r', alpha=0.5, zorder=-100)
            ax.plot(np.mean(ct_chirps, axis=0), c='darkred', lw=2, zorder=2)

            ax = ax_row[1]
            ct_bars = preproc_bars[data_celltypes == celltype]
            ax.plot(ct_bars.T, lw=0.5, c='r', alpha=0.5, zorder=-100)
            ax.plot(np.mean(ct_bars, axis=0), c='darkred', lw=2, zorder=2)

            ax = ax_row[2]
            ax.hist(bar_ds_pvalues[data_celltypes == celltype], bins=bar_ds_pvalues_bins, color='r', alpha=0.7)
            if celltype == celltypes[0]:
                ax.set_title('bar_ds_pvalues')

            ax = ax_row[3]
            ax.hist(roi_size_um2s[data_celltypes == celltype], bins=roi_size_um2s_bins, color='r', alpha=0.7)
            if celltype == celltypes[0]:
                ax.set_title('roi_size_um2s')

        plt.tight_layout()
