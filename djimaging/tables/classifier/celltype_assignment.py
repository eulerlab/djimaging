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


def extract_feature(preproc_chirp, preproc_bar, bar_ds_pvalue, roi_size_um2, chirp_features, bar_features):
    feature_activation_chirp = np.dot(preproc_chirp, chirp_features)
    feature_activation_bar = np.dot(preproc_bar, bar_features)

    feature = np.concatenate(
        [feature_activation_chirp, feature_activation_bar, np.array([bar_ds_pvalue]), np.array([roi_size_um2])])

    return feature


def extract_features(preproc_chirps, preproc_bars, bar_ds_pvalues, roi_size_um2s, chirp_features, bar_features):
    features = np.concatenate([
        np.dot(preproc_chirps, chirp_features),
        np.dot(preproc_bars, bar_features),
        bar_ds_pvalues[:, np.newaxis],
        roi_size_um2s[:, np.newaxis]
    ], axis=-1)

    return features


def classify_cell(preproc_chirp, preproc_bar, bar_ds_pvalue, roi_size_um2,
                  chirp_features, bar_features, classifier):
    feature = extract_feature(
        preproc_chirp, preproc_bar, bar_ds_pvalue, roi_size_um2, chirp_features, bar_features)
    confidence_scores = classifier.predict_proba(feature[np.newaxis, :]).flatten()
    cell_label = confidence_scores.argmax(axis=1)
    cell_label = classifier.classes_[cell_label][0]
    return cell_label, confidence_scores


def classify_cells(preproc_chirps, preproc_bars, bar_ds_pvalues, roi_size_um2s,
                   chirp_features, bar_features, classifier):
    features = extract_features(
        preproc_chirps, preproc_bars, bar_ds_pvalues, roi_size_um2s, chirp_features, bar_features)
    confidence_scores = classifier.predict_proba(features)
    cell_labels = confidence_scores.argmax(axis=1)
    cell_labels = classifier.classes_[cell_labels]
    return cell_labels, confidence_scores


class CelltypeAssignmentTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
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
        try:
            return self.classifier_table.proj() * self.field_table.proj()
        except (AttributeError, TypeError):
            pass

    @property
    @abstractmethod
    def classifier_training_data_table(self):
        pass

    @property
    @abstractmethod
    def classifier_table(self):
        pass

    @property
    @abstractmethod
    def os_ds_table(self):
        pass

    @property
    @abstractmethod
    def roi_table(self):
        pass

    @property
    @abstractmethod
    def baden_trace_table(self):
        pass

    @property
    @abstractmethod
    def field_table(self):
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
        model_path = (self.classifier_table() & self.current_model_key).fetch1('classifier_file')
        with open(model_path, "rb") as f:
            model = pkl.load(f)
        return model

    @cached_property
    def bar_features(self):
        features_bar_file = (self.classifier_training_data_table() & self.current_model_key).fetch1(
            "bar_feats_file")
        features_bar = np.load(features_bar_file)
        return features_bar

    @cached_property
    def chirp_features(self):
        features_chirp_file = (self.classifier_training_data_table() & self.current_model_key).fetch1(
            "chirp_feats_file")
        features_chirp = np.load(features_chirp_file)
        return features_chirp

    def populate(
            self, *restrictions, suppress_errors=False,
            return_exception_objects=False, reserve_jobs=False, order="original", limit=None, max_calls=None,
            display_progress=False, processes=1, make_kwargs=None,
    ):
        if processes > 1:
            warnings.warn('Parallel processing not implemented!')
        super().populate(
            *restrictions,
            suppress_errors=suppress_errors, return_exception_objects=return_exception_objects,
            reserve_jobs=reserve_jobs, order=order, limit=limit, max_calls=max_calls,
            display_progress=display_progress, processes=1, make_kwargs=make_kwargs)

    def make(self, key):
        roi_keys, preproc_chirps, preproc_bars, bar_ds_pvalues, roi_size_um2s = self._fetch_data(key)

        if roi_keys is None:
            return

        cell_labels, confidence_scores = self._classify_cells(
            preproc_chirps, preproc_bars, bar_ds_pvalues, roi_size_um2s)

        for roi_key, cell_label, confidence_scores_i in zip(roi_keys, cell_labels, confidence_scores):
            self.insert1(dict(**merge_keys(key, roi_key), cell_label=cell_label,
                              confidence=confidence_scores_i, max_confidence=np.max(confidence_scores_i)))

    def _fetch_data(self, key, restriction=None):
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

    def _classify_cells(self, preproc_chirps, preproc_bars, bar_ds_pvalues, roi_size_um2s):
        cell_labels, confidence_scores = classify_cells(
            preproc_chirps, preproc_bars, bar_ds_pvalues, roi_size_um2s,
            self.chirp_features, self.bar_features, self.classifier)
        return cell_labels, confidence_scores

    def plot(self, threshold_confidence: float, classifier_level: str):
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

    def _plot_key(self, ax, df, classifier_params_hash, training_data_hash, preprocess_id,
                  threshold_confidence, classifier_level):
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
                          celltypes=None, n_celltypes_max=32,
                          xlim_chirp=None, xlim_bar=None, plot_baden_data=True):
        df = self.fetch(format='frame')
        groups = df.groupby(['training_data_hash', 'classifier_params_hash', 'preprocess_id'])

        for (tdh, cph, pid), df_group in groups:
            self._plot_group_traces_key(classifier_params_hash=cph, training_data_hash=tdh, preprocess_id=pid,
                                        threshold_confidence=threshold_confidence,
                                        classifier_level=classifier_level,
                                        celltypes=celltypes, n_celltypes_max=n_celltypes_max,
                                        xlim_chirp=xlim_chirp, xlim_bar=xlim_bar, plot_baden_data=plot_baden_data)

    def _plot_group_traces_key(self, threshold_confidence: float, classifier_level: str,
                               classifier_params_hash, training_data_hash, preprocess_id,
                               celltypes=None, n_celltypes_max=32, xlim_chirp=None, xlim_bar=None,
                               plot_baden_data=True):
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
