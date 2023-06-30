import os
import pickle as pkl
import warnings
from abc import abstractmethod
from typing import Mapping, Dict, Any

import datajoint as dj
import numpy as np
from cached_property import cached_property
from matplotlib import pyplot as plt

from djimaging.utils.dj_utils import make_hash, merge_keys, get_primary_key
from djimaging.utils.import_helpers import dynamic_import, split_module_name

Key = Dict[str, Any]


def prepare_dj_config_rgc_classifier(output_folder, input_folder="/gpfs01/euler/data/Resources/Classifier"):
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


class ClassifierMethodTemplate(dj.Lookup):
    database = ""

    @property
    def definition(self):
        definition = """
        classifier_params_hash  : varchar(32)     # hash of the classifier params config
        ---
        classifier_fn           : varchar(64)     # path to classifier method fn
        classifier_config       : longblob        # method configuration object
        classifier_seed         : int
        comment                 : varchar(300)    # comment
        """
        return definition

    import_func = staticmethod(dynamic_import)

    @property
    @abstractmethod
    def classifier_training_data_table(self):
        pass

    def add_classifier(self, classifier_fn: str, classifier_config: Mapping,
                       comment: str = "", skip_duplicates: bool = False, classifier_seed: int = 42) -> None:
        self.insert1(
            dict(
                classifier_fn=classifier_fn,
                classifier_params_hash=make_hash(classifier_config),
                classifier_config=classifier_config,
                classifier_seed=classifier_seed,
                comment=comment),
            skip_duplicates=skip_duplicates)

    def train_classifier(self, key: Key):
        classifier_fn, classifier_config, classifier_seed = \
            (self & key).fetch1("classifier_fn", "classifier_config", "classifier_seed")
        classifier_config["random_state"] = classifier_seed

        classifier_fn = self.import_func(*split_module_name(classifier_fn))
        training_data = self.classifier_training_data_table().get_training_data(key)
        classifier = classifier_fn(**classifier_config)
        classifier.fit(X=training_data["X"], y=training_data["y"])
        score = classifier.score(X=training_data["X"], y=training_data["y"])
        return classifier, score


class ClassifierTrainingDataTemplate(dj.Manual):
    database = ""
    store = "classifier_input"

    @property
    def definition(self):
        definition = """
        # holds feature basis and training data for classifier
        training_data_hash     :   varchar(32)     # hash of the classifier training data files
        ---
        project                :   enum("True", "False")     # flag whether to project data onto features anew or not
        output_path            :   varchar(255)
        chirp_feats_file       :   filepath@{store}
        bar_feats_file         :   filepath@{store}
        baden_data_file        :   filepath@{store}
        training_data_file     :   filepath@{store}
        """.format(store=self.store)
        return definition

    def add_trainingdata(self, project: str, output_path: str, chirp_feats_file: str, bar_feats_file: str,
                         baden_data_file: str, training_data_file: str, skip_duplicates: bool = False) -> None:

        key = dict(
            project=project,
            output_path=output_path,
            chirp_feats_file=chirp_feats_file,
            bar_feats_file=bar_feats_file,
            baden_data_file=baden_data_file,
            training_data_file=training_data_file)
        key["training_data_hash"] = make_hash(key)
        self.insert1(key, skip_duplicates=skip_duplicates)

    def get_training_data(self, key: Key):
        if (self & key).fetch1("project") == "True":
            raise NotImplementedError
        else:
            training_data_file = (self & key).fetch1("training_data_file")
            with open(training_data_file, "rb") as f:
                training_data = pkl.load(f)
            return training_data


def classify_cell(preproc_chirp, preproc_bar, bar_ds_pvalue, roi_size_um2,
                  chirp_features, bar_features, classifier):
    # feature activation matrix
    feature_activation_chirp = np.dot(preproc_chirp, chirp_features)
    feature_activation_bar = np.dot(preproc_bar, bar_features)

    features = np.concatenate(
        [feature_activation_chirp, feature_activation_bar, np.array([bar_ds_pvalue]), np.array([roi_size_um2])])

    confidence_scores = classifier.predict_proba(features[np.newaxis, :]).flatten()
    celltype = np.argmax(confidence_scores) + 1

    return celltype, confidence_scores


def extract_features(preproc_chirps, preproc_bars, bar_ds_pvalues, roi_size_um2s, chirp_features, bar_features):
    features = np.concatenate([
        np.dot(preproc_chirps, chirp_features),
        np.dot(preproc_bars, bar_features),
        bar_ds_pvalues[:, np.newaxis],
        roi_size_um2s[:, np.newaxis]
    ], axis=-1)

    return features


def classify_cells(preproc_chirps, preproc_bars, bar_ds_pvalues, roi_size_um2s,
                   chirp_features, bar_features, classifier):
    features = extract_features(
        preproc_chirps, preproc_bars, bar_ds_pvalues, roi_size_um2s, chirp_features, bar_features)
    confidence_scores = classifier.predict_proba(features)
    celltypes = np.argmax(confidence_scores, axis=1) + 1

    return celltypes, confidence_scores


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
        score_train : float
        """.format(store=self.store)
        return definition

    @property
    def key_source(self):
        try:
            return self.classifier_training_data_table().proj() * self.classifier_method_table.proj()
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
        classifier_file = os.path.join(output_path, 'rgc_classifier.pkl')
        classifier, score_train = self.classifier_method_table().train_classifier(key)

        print(f'Saving classifier to {classifier_file}')
        with open(classifier_file, "wb") as f:
            pkl.dump(classifier, f)

        self.insert1(dict(key, classifier_file=classifier_file, score_train=score_train))


class CelltypeAssignmentTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = """
        # Feature-based cluster assignment of cells
        -> self.baden_trace_table
        -> self.classifier_table
        ---
        celltype:        int         # predicted group (1-46), without quality or confidence threshold
        max_confidence:  float       # confidence score for assigned celltype for easy restriction
        confidence:      blob        # confidence score (probability) for all 46 celltypes
        """
        return definition

    @property
    def key_source(self):
        try:
            return self.classifier_table.proj() * self.baden_trace_table.roi_table.field_or_pres_table.proj()
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
    def baden_trace_table(self):
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
        self.current_model_key = dict(
            classifier_params_hash=key["classifier_params_hash"],
            training_data_hash=key["training_data_hash"])

        roi_tab = self.baden_trace_table * self.baden_trace_table().bar_tab() * self.baden_trace_table.roi_table
        roi_keys = (self.baden_trace_table & key).proj()

        if len(roi_keys) == 0:
            return

        preproc_chirps, preproc_bars, bar_ds_pvalues, roi_size_um2s = (roi_tab & roi_keys).fetch(
            'preproc_chirp', 'preproc_bar', 'bar_ds_pvalue', 'roi_size_um2')

        celltype, confidence_scores = classify_cells(
            np.vstack(preproc_chirps), np.vstack(preproc_bars), bar_ds_pvalues, roi_size_um2s,
            self.chirp_features, self.bar_features, self.classifier)

        for roi_key, celltype, confidence_scores_i in zip(roi_keys, celltype, confidence_scores):
            self.insert1(dict(**merge_keys(key, roi_key), celltype=celltype,
                              confidence=confidence_scores_i, max_confidence=np.max(confidence_scores_i)))

    def plot(self, threshold_confidence: float):
        df = self.fetch(format='frame')
        groups = df.groupby(['training_data_hash', 'classifier_params_hash', 'preprocess_id'])

        fig, axs = plt.subplots(len(groups), 1, figsize=(12, 3 * len(groups)), squeeze=False)
        axs = axs.flatten()

        for ax, ((tdh, cph, pid), df) in zip(axs, groups):
            celltypes = df.apply(
                lambda row: row["celltype"] if row["max_confidence"] > threshold_confidence else -1, axis=1)

            ax.hist(celltypes[celltypes > 0], bins=np.arange(1, 47, 0.5), align='left')
            ax.hist(celltypes[celltypes < 0], bins=(-1, -0.5, 0), align='left', color='red', alpha=0.5)
            ax.set_xticks(np.append(-1, np.arange(1, 47)))
            ax.set_xticklabels(np.append("NA", np.arange(1, 47)), rotation=90)
            ax.set(ylabel='Count', xlabel='celltype')
            ax.set_title(
                f"training_data_hash={tdh}\nclassifier_params_hash={cph}\npreprocess_id={pid}", loc='left')

        plt.tight_layout()
        plt.show()

    def plot_features(self, threshold_confidence: float, key=None, celltypes=None, n_celltypes_max=3,
                      features=None, n_features_max=5):
        key = get_primary_key(self.classifier_table, key)

        # Get new data
        self.current_model_key = dict(
            classifier_params_hash=key["classifier_params_hash"],
            training_data_hash=key["training_data_hash"])

        roi_tab = self * self.baden_trace_table * self.baden_trace_table().bar_tab() * self.baden_trace_table.roi_table
        roi_keys = (self.baden_trace_table & key).proj()
        preproc_chirps, preproc_bars, bar_ds_pvalues, roi_size_um2s, data_celltypes = \
            (roi_tab & roi_keys & f'max_confidence>="{threshold_confidence}"').fetch(
                'preproc_chirp', 'preproc_bar', 'bar_ds_pvalue', 'roi_size_um2', 'celltype')

        data_features = extract_features(
            np.vstack(preproc_chirps), np.vstack(preproc_bars), bar_ds_pvalues, roi_size_um2s,
            self.chirp_features, self.bar_features)

        # Get training data
        training_data = self.classifier_training_data_table().get_training_data(key)

        if celltypes is None:
            celltypes = np.arange(0, 46) + 1
        if features is None:
            features = np.arange(0, training_data['X'].shape[1])

        if n_celltypes_max is not None and len(celltypes) > n_celltypes_max:
            celltypes = np.sort(np.random.choice(celltypes, n_celltypes_max, replace=False))
        if n_features_max is not None and len(features) > n_features_max:
            features = np.sort(np.random.choice(features, n_features_max, replace=False))

        n_celltypes = len(celltypes)
        n_features = len(features)

        # Plot
        fig, axs = plt.subplots(n_celltypes, n_features,
                                figsize=(np.minimum(3 * n_features, 20), np.minimum(2 * n_celltypes, 20)),
                                sharex='col', sharey='col', squeeze=False)

        for ax_row, celltype in zip(axs, celltypes):
            ax_row[0].set_ylabel(f"celltype={celltype}")
            for ax, feature_i in zip(ax_row, features):
                ax.set_title(f"feature={feature_i}")
                bins = np.linspace(training_data['X'][:, feature_i].min(), training_data['X'][:, feature_i].max(), 51)
                ax.hist(training_data['X'][training_data['y'] == celltype, feature_i],
                        bins=bins, color='gray', alpha=1, label='train')

                if np.any(data_celltypes == celltype):
                    ax = ax.twinx()
                    ax.hist(data_features[data_celltypes == celltype, feature_i],
                            bins=bins, color='r', alpha=0.5, label='new')

        axs[0, 0].legend()

        plt.tight_layout()

    def plot_group_traces(self, threshold_confidence: float, key=None, celltypes=None, n_celltypes_max=32):
        key = get_primary_key(self.classifier_table, key)

        # Get new data
        self.current_model_key = dict(
            classifier_params_hash=key["classifier_params_hash"],
            training_data_hash=key["training_data_hash"])

        roi_tab = self * self.baden_trace_table * self.baden_trace_table().bar_tab() * self.baden_trace_table.roi_table
        roi_keys = (self.baden_trace_table & key).proj()
        preproc_chirps, preproc_bars, bar_ds_pvalues, roi_size_um2s, data_celltypes = \
            (roi_tab & roi_keys & f'max_confidence>="{threshold_confidence}"').fetch(
                'preproc_chirp', 'preproc_bar', 'bar_ds_pvalue', 'roi_size_um2', 'celltype')

        preproc_chirps = np.vstack(preproc_chirps)
        preproc_bars = np.vstack(preproc_bars)

        if celltypes is None:
            celltypes = np.arange(0, 32) + 1

        if n_celltypes_max is not None and len(celltypes) > n_celltypes_max:
            celltypes = np.sort(np.random.choice(celltypes, n_celltypes_max, replace=False))

        fig, axs = plt.subplots(len(celltypes), 4,
                                figsize=(12, np.minimum(2 * len(celltypes), 20)),
                                sharex='col', sharey='col', squeeze=False,
                                gridspec_kw=dict(width_ratios=[1, 0.6, 0.3, 0.3]))

        import seaborn as sns
        sns.despine(fig=fig)

        for ax_row, celltype in zip(axs, celltypes):

            ax_row[0].set_ylabel(f"{celltype}")

            if not np.any(data_celltypes == celltype):
                continue

            ax = ax_row[0]
            ax.plot(preproc_chirps[data_celltypes == celltype].T, lw=0.5)
            ax.plot(np.mean(preproc_chirps[data_celltypes == celltype], axis=0), c='k', lw=2)

            ax = ax_row[1]
            ax.plot(preproc_bars[data_celltypes == celltype].T, lw=0.5)
            ax.plot(np.mean(preproc_bars[data_celltypes == celltype], axis=0), c='k', lw=2)

            ax = ax_row[2]
            ax.hist(bar_ds_pvalues[data_celltypes == celltype],
                    bins=np.linspace(np.min(bar_ds_pvalues), np.max(bar_ds_pvalues), 51))
            if celltype == celltypes[0]:
                ax.set_title('bar_ds_pvalues')

            ax = ax_row[3]
            ax.hist(roi_size_um2s[data_celltypes == celltype],
                    bins=np.linspace(np.min(roi_size_um2s), np.max(roi_size_um2s), 51))
            if celltype == celltypes[0]:
                ax.set_title('roi_size_um2s')

        plt.tight_layout()
