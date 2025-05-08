"""
Classifier for RGCs used Baden et al. 2016 dataset.

Example usage:

from djimaging.tables import classifier

@schema
class Baden16Traces(classifier.Baden16TracesTemplate):
    _shift_chirp = 1
    _shift_bar = -4

    _stim_name_chirp = 'gChirp'
    _stim_name_bar = 'movingbar'

    averages_table = Averages
    os_ds_table = OsDsIndexes

@schema
class ClassifierTrainingData(classifier.ClassifierTrainingDataTemplate):
    pass


@schema
class ClassifierMethod(classifier.ClassifierMethodTemplate):
    classifier_training_data_table = ClassifierTrainingData


@schema
class Classifier(classifier.ClassifierTemplate):
    classifier_training_data_table = ClassifierTrainingData
    classifier_method_table = ClassifierMethod


@schema
class CelltypeAssignment(classifier.CelltypeAssignmentTemplate):
    classifier_training_data_table = ClassifierTrainingData
    classifier_table = Classifier
    baden_trace_table = Baden16Traces
    field_table = Field
    roi_table = Roi
    os_ds_table = OsDsIndexes
"""

import os
import pickle as pkl
import warnings
from abc import abstractmethod
from functools import lru_cache
from typing import Mapping, Dict, Any

import datajoint as dj
import numpy as np
from cached_property import cached_property
from matplotlib import pyplot as plt

from djimaging.utils.dj_utils import make_hash, merge_keys, get_primary_key
from djimaging.utils.import_helpers import extract_class_info, load_class

Key = Dict[str, Any]


def prepare_dj_config_rgc_classifier(output_folder,
                                     input_folder="/gpfs01/euler/data/Resources/Classifier/rgc_classifier_v1"):
    stores_dict = {
        "classifier_input": {"protocol": "file", "location": input_folder, "stage": input_folder},
        "classifier_output": {"protocol": "file", "location": output_folder, "stage": output_folder},
    }

    # Make sure folders exits
    for store, store_dict in stores_dict.items():
        for name in store_dict.keys():
            if name in ["location", "stage"]:
                assert os.path.isdir(store_dict[name]), f'This must be a folder you have access to: {store_dict[name]}'

    os.environ["DJ_SUPPORT_FILEPATH_MANAGEMENT"] = "TRUE"
    dj.config['enable_python_native_blobs'] = True

    dj_config_stores = dj.config.get('stores', None) or dict()
    dj_config_stores.update(stores_dict)
    dj.config['stores'] = dj_config_stores


class ClassifierTrainingDataTemplate(dj.Manual):
    database = ""
    _store = "classifier_input"

    @property
    def definition(self):
        definition = """
        # holds feature basis and training data for classifier
        training_data_hash     :   varchar(63)     # hash of the classifier training data files
        ---
        output_path            :   varchar(191)
        baden_data_file        :   filepath@{store}
        chirp_feats_file       :   filepath@{store}
        bar_feats_file         :   filepath@{store}
        """.format(store=self._store)
        return definition

    def add_default(self, skip_duplicates=False):
        ipath = dj.config['stores']["classifier_input"]["location"] + '/'
        opath = dj.config['stores']["classifier_output"]["location"] + '/'

        self.add_trainingdata(
            output_path=opath,
            baden_data_file=ipath + 'RGCData_postprocessed.mat',
            chirp_feats_file=ipath + 'chirp_feats.npz',
            bar_feats_file=ipath + 'bar_feats.npz',
            skip_duplicates=skip_duplicates,
        )

    def add_trainingdata(self, output_path: str, chirp_feats_file: str, bar_feats_file: str,
                         baden_data_file: str, skip_duplicates: bool = False) -> None:
        key = dict(
            output_path=output_path,
            chirp_feats_file=chirp_feats_file,
            bar_feats_file=bar_feats_file,
            baden_data_file=baden_data_file
        )
        key["training_data_hash"] = make_hash(key)
        self.insert1(key, skip_duplicates=skip_duplicates)

    def get_training_data(self, key: Key):
        baden_data_file, chirp_feats_file, bar_feats_file = (self & key).fetch1(
            'baden_data_file', 'chirp_feats_file', 'bar_feats_file')

        b_c_labels, b_g_labels, b_s_labels, b_c_traces, b_c_qi, b_mb_traces, b_mb_qi, b_mb_dsi, b_mb_dp, b_soma_um2 = \
            load_baden_data(baden_data_file)
        chirp_features = np.load(chirp_feats_file)
        bar_features = np.load(bar_feats_file)

        features = extract_features(
            preproc_chirps=b_c_traces,
            preproc_bars=b_mb_traces,
            bar_ds_pvalues=b_mb_dp,
            roi_size_um2s=b_soma_um2,
            chirp_features=chirp_features,
            bar_features=bar_features,
        )

        return features, b_c_labels, b_g_labels, b_s_labels


class ClassifierMethodTemplate(dj.Lookup):
    database = ""

    @property
    def definition(self):
        definition = """
        classifier_params_hash  : varchar(63)     # hash of the classifier params config
        ---
        label_kind              : enum('cluster', 'group', 'super')  # Predict group or cluster labels from Baden et al. 16
        classifier_fn           : varchar(191)    # path to classifier method fn
        classifier_config       : longblob        # method configuration object
        classifier_seed         : int
        comment                 : varchar(191)    # comment
        """
        return definition

    import_func = staticmethod(load_class)

    @property
    @abstractmethod
    def classifier_training_data_table(self):
        pass

    def add_default(self, label_kind='group', skip_duplicates=False):
        classifier_fn = "sklearn.ensemble.RandomForestClassifier"
        classifier_config = {
            'class_weight': 'balanced',
            'random_state': 2001,
            'oob_score': True,
            'ccp_alpha': 0.00021870687842726034,
            'max_depth': 50,
            'max_leaf_nodes': None,
            'min_impurity_decrease': 0,
            'n_estimators': 1000,
            'n_jobs': 20,
        }

        self.add_classifier(label_kind, classifier_fn, classifier_config, comment="Default classifier",
                            skip_duplicates=skip_duplicates)

    def add_classifier(self, label_kind: str, classifier_fn: str, classifier_config: Mapping,
                       comment: str = "", skip_duplicates: bool = False, classifier_seed: int = 42) -> None:
        if classifier_config is None:
            classifier_config = {
                'class_weight': 'balanced',
                'random_state': 2001,
                'oob_score': True,
                'ccp_alpha': 0.00021870687842726034,
                'max_depth': 50,
                'max_leaf_nodes': None,
                'min_impurity_decrease': 0,
                'n_estimators': 1000,
                'n_jobs': 20,
            }

        all_config = classifier_config.copy()
        all_config['label_kind'] = label_kind
        all_config['classifier_fn'] = classifier_fn
        classifier_params_hash = make_hash(all_config)

        self.insert1(
            dict(
                label_kind=label_kind,
                classifier_fn=classifier_fn,
                classifier_params_hash=classifier_params_hash,
                classifier_config=classifier_config,
                classifier_seed=classifier_seed,
                comment=comment),
            skip_duplicates=skip_duplicates)

    def train_classifier(self, key: Key):
        from sklearn.model_selection import train_test_split

        label_kind, classifier_fn, classifier_config, classifier_seed = \
            (self & key).fetch1("label_kind", "classifier_fn", "classifier_config", "classifier_seed")
        classifier_config["random_state"] = classifier_seed

        print(f'Train classifier: {classifier_fn} {key["classifier_params_hash"]}')

        classifier_fn = self.import_func(*extract_class_info(classifier_fn))
        features, b_c_labels, b_g_labels, b_s_labels = self.classifier_training_data_table().get_training_data(key)

        if label_kind == 'cluster':
            labels = b_c_labels
        elif label_kind == 'group':
            labels = b_g_labels
        elif label_kind == 'super':
            labels = b_s_labels
        else:
            raise NotImplementedError(label_kind)

        print(f'Use `{label_kind}`-labels with n={np.unique(labels).size} labels')

        print(f'Splitting data n={labels.size} ...')
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=2001)

        print('Fitting classifier on training data ...')
        classifier = classifier_fn(**classifier_config)
        classifier.fit(X=X_train, y=y_train)

        print('Evaluate classifier on train and test data ...')
        score_train = classifier.score(X=X_train, y=y_train)
        score_test = classifier.score(X=X_test, y=y_test)
        print('Train score: {:.3f}'.format(score_train))
        print('Test score: {:.3f}'.format(score_test))

        print('Refitting classifier on all data ...')
        classifier = classifier_fn(**classifier_config)
        classifier.fit(X=features, y=labels)
        print('Evaluate classifier on all data ...')
        score_final = classifier.score(X=features, y=labels)
        print('Final score: {:.3f}\n'.format(score_final))
        return classifier, score_train, score_test, score_final


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


class ClassifierTemplate(dj.Computed):
    database = ""
    store = "classifier_output"

    @property
    def definition(self):
        definition = """
        -> self.classifier_training_data_table 
        -> self.classifier_method_table
        ---
        classifier_file         :   attach@{store}
        score_train : float  # Train split data score
        score_test : float  # Test split score
        score_final : float  # Score after retraining on all data (the final classifier)
        """.format(store=self.store)
        return definition

    @property
    def key_source(self):
        try:
            return self.classifier_method_table.proj() * self.classifier_training_data_table().proj()
        except (AttributeError, TypeError):
            pass

    @property
    @abstractmethod
    def classifier_training_data_table(self):
        pass

    @property
    @abstractmethod
    def classifier_method_table(self):
        pass

    def make(self, key):
        output_path = (self.classifier_training_data_table() & key).fetch1("output_path")
        classifier_file = os.path.join(output_path, f'rgc_classifier_{key["classifier_params_hash"]}.pkl')
        classifier, score_train, score_test, score_final = self.classifier_method_table().train_classifier(key)

        print(f'Saving classifier to {classifier_file}\n')
        with open(classifier_file, "wb") as f:
            pkl.dump(classifier, f)

        self.insert1(dict(key, classifier_file=classifier_file,
                          score_train=score_train, score_test=score_test, score_final=score_final))


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

        preproc_chirps, preproc_bars, bar_ds_pvalues, roi_size_um2s = (
                self.baden_trace_table * self.os_ds_table * self.roi_table & key & restriction).fetch(
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
                b_chirps = b_c_traces[b_g_labels == celltype]
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


@lru_cache(maxsize=None)
def load_baden_data(baden_data_file, quality_filter=True):
    from scipy.io import loadmat
    baden_data = loadmat(baden_data_file, struct_as_record=True, matlab_compatible=False,
                         squeeze_me=True, simplify_cells=True)['data']

    roi_size_um2 = baden_data['info']['area2d']

    chirp_traces = baden_data['chirp']['traces'].T
    chirp_qi = baden_data['chirp']['qi']

    bar_traces = baden_data['ds']['tc'].T
    bar_qi = baden_data['ds']['qi']
    bar_dsi = baden_data['ds']['dsi']
    bar_dp = baden_data['ds']['dP']

    cluster_labels = np.asarray(baden_data['info']['final_idx']).flatten()
    group_labels = np.array([baden_cluster_to_group(c_label) for c_label in cluster_labels])
    super_labels = np.array([baden_group_to_supergroup(g_label) for g_label in group_labels])

    if quality_filter:
        qidx = (cluster_labels > 0) & ((bar_qi > 0.6) | (chirp_qi > 0.45))
    else:
        qidx = np.ones(cluster_labels.size, dtype=bool)

    return (
        cluster_labels[qidx], group_labels[qidx], super_labels[qidx],
        chirp_traces[qidx], chirp_qi[qidx], bar_traces[qidx], bar_qi[qidx], bar_dsi[qidx], bar_dp[qidx],
        roi_size_um2[qidx],
    )


def baden_cluster_to_group(cluster_label):
    _baden_cluster_to_group = {
        1: 1,
        2: 2,
        3: 3,
        4: 4, 5: 4,
        6: 5, 7: 5, 8: 5,
        9: 6,
        10: 7,
        11: 8, 12: 8,
        13: 9,
        14: 10,
        15: 11, 16: 11,
        17: 12, 18: 12,
        19: 13,
        20: 14,
        21: 15,
        22: 16,
        23: 17, 24: 17,
        25: 17,
        26: 18, 27: 18,
        28: 19,
        29: 20,
        30: 21,
        31: 22, 32: 22,
        33: 23,
        34: 24,
        35: 25,
        36: 26,
        37: 27,
        38: 28, 39: 28,
        40: 29,
        41: 30,
        42: 31, 43: 31, 44: 31, 45: 31, 46: 31,
        47: 32, 48: 32, 49: 32,
        50: 33,
        51: 34, 52: 34,
        53: 35, 54: 35,
        55: 36,
        56: 37, 57: 37,
        58: 38, 59: 38, 60: 38,
        61: 39,
        62: 40, 63: 40,
        64: 41,
        65: 42, 66: 42, 67: 42, 68: 42, 69: 42, 70: 42,
        71: 43,
        72: 44,
        73: 45,
        74: 46, 75: 46,
    }

    return _baden_cluster_to_group.get(cluster_label, -1)


def baden_group_to_supergroup(group_label):
    if group_label <= 0:
        return -1
    elif group_label <= 9:
        return 1  # Off
    elif group_label <= 14:
        return 2  # On Off
    elif group_label <= 20:
        return 3  # Fast On
    elif group_label <= 28:
        return 4  # Slow On
    elif group_label <= 32:
        return 5  # Uncertain
    elif group_label <= 46:
        return 6  # dAC
    else:
        return -1


def supergroup2str(supergroup_label):
    if supergroup_label <= 0:
        return 'none'
    elif supergroup_label == 1:
        return 'Off'
    elif supergroup_label == 2:
        return 'On Off'
    elif supergroup_label == 3:
        return 'Fast On'
    elif supergroup_label == 4:
        return 'Slow On'
    elif supergroup_label == 5:
        return 'Uncertain'
    elif supergroup_label == 6:
        return 'dAC'
    else:
        return 'Unknown'
