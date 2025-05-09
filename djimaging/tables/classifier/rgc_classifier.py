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
        project                :   enum("True", "False")     # flag whether to project data onto features anew or not
        output_path            :   varchar(191)
        chirp_feats_file       :   filepath@{store}
        bar_feats_file         :   filepath@{store}
        baden_data_file        :   filepath@{store}
        training_data_file     :   filepath@{store}
        """.format(store=self._store)
        return definition

    def add_default(self, training_data_file='training_all.pkl', skip_duplicates=False):
        ipath = dj.config['stores']["classifier_input"]["location"] + '/'
        opath = dj.config['stores']["classifier_output"]["location"] + '/'

        self.add_trainingdata(
            project="False",
            output_path=opath,
            chirp_feats_file=ipath + 'chirp_feats.npz',
            bar_feats_file=ipath + 'bar_feats.npz',
            baden_data_file=ipath + 'RGCData_postprocessed.mat',
            training_data_file=ipath + training_data_file,
            skip_duplicates=skip_duplicates,
        )

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


class ClassifierMethodTemplate(dj.Lookup):
    database = ""

    @property
    def definition(self):
        definition = """
        classifier_params_hash  : varchar(63)     # hash of the classifier params config
        ---
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

    def add_default(self, skip_duplicates=False):
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

        self.add_classifier(classifier_fn, classifier_config, comment="Default classifier",
                            skip_duplicates=skip_duplicates)

    def add_classifier(self, classifier_fn: str, classifier_config: Mapping,
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

        classifier_fn = self.import_func(*extract_class_info(classifier_fn))
        training_data = self.classifier_training_data_table().get_training_data(key)
        classifier = classifier_fn(**classifier_config)
        classifier.fit(X=training_data["X"], y=training_data["y"])
        score = classifier.score(X=training_data["X"], y=training_data["y"])
        return classifier, score


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
        celltype:        int         # predicted group, without quality or confidence threshold
        max_confidence:  float       # confidence score for assigned celltype for easy restriction
        confidence:      blob        # confidence score (probability) for all celltypes
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

    @cached_property
    def baden_data(self):
        baden_data_file = (self.classifier_training_data_table() & self.current_model_key).fetch1("baden_data_file")
        return load_baden_data(baden_data_file)

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

        celltypes, confidence_scores = self._classify_cells(
            preproc_chirps, preproc_bars, bar_ds_pvalues, roi_size_um2s)

        for roi_key, celltype, confidence_scores_i in zip(roi_keys, celltypes, confidence_scores):
            self.insert1(dict(**merge_keys(key, roi_key), celltype=celltype,
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
        celltypes, confidence_scores = classify_cells(
            preproc_chirps, preproc_bars, bar_ds_pvalues, roi_size_um2s,
            self.chirp_features, self.bar_features, self.classifier)
        return celltypes, confidence_scores

    def plot(self, threshold_confidence: float):
        df = self.fetch(format='frame')
        groups = df.groupby(['training_data_hash', 'classifier_params_hash', 'preprocess_id'])

        fig, axs = plt.subplots(len(groups), 1, figsize=(12, 3 * len(groups)), squeeze=False)
        axs = axs.flatten()

        for ax, ((tdh, cph, pid), df) in zip(axs, groups):
            celltypes = df.apply(
                lambda row: row["celltype"] if row["max_confidence"] > threshold_confidence else -1, axis=1)

            self.current_model_key = dict(classifier_params_hash=cph, training_data_hash=tdh)
            training_data = self.classifier_training_data_table().get_training_data(self.current_model_key)
            group_ticks = np.unique(training_data['y'])
            group_names = training_data.get("group_name", group_ticks)

            ax.hist(celltypes[celltypes > 0], bins=np.arange(group_ticks.min(), group_ticks.max() + 1, 0.5),
                    align='left')
            ax.set_xticks(group_ticks)
            ax.set_xticklabels([f"{gn} [{gt}]" for gt, gn in zip(group_ticks, group_names)], rotation=90)
            ax.set(ylabel='Count', xlabel='Group-Name [Group-Num]')
            ax.set_title(
                f"training_data_hash={tdh}\n"
                f"classifier_params_hash={cph}\n"
                f"preprocess_id={pid}\n"
                f"{np.sum(celltypes == -1)}={np.mean(celltypes == -1):.0%} were not classified", loc='left')

        plt.tight_layout()
        plt.show()

    def plot_features(self, threshold_confidence: float,
                      key=None, celltypes=None, n_celltypes_max=3, features=None, n_features_max=5):
        key = get_primary_key(self.classifier_table, key)

        # Get training data
        self.current_model_key = dict(classifier_params_hash=key["classifier_params_hash"],
                                      training_data_hash=key["training_data_hash"])
        training_data = self.classifier_training_data_table().get_training_data(self.current_model_key)
        group_ticks = np.unique(training_data['y'])
        group_names = training_data.get("group_name", group_ticks)

        roi_keys, preproc_chirps, preproc_bars, bar_ds_pvalues, roi_size_um2s = self._fetch_data(
            key, restriction=(self & f'max_confidence>={threshold_confidence}'))
        data_celltypes = (self & roi_keys).fetch('celltype')
        data_celltypes = [group_names[np.argmax(celltype == group_ticks)] for celltype in data_celltypes]

        data_features = extract_features(
            preproc_chirps, preproc_bars, bar_ds_pvalues, roi_size_um2s, self.chirp_features, self.bar_features)

        if celltypes is None:
            celltypes = group_ticks
        if features is None:
            features = np.arange(0, training_data['X'].shape[1])

        celltypes = sorted(set(celltypes).intersection(set(data_celltypes)))

        if n_celltypes_max is not None and len(celltypes) > n_celltypes_max:
            celltypes = sorted(np.random.choice(celltypes, n_celltypes_max, replace=False))
        if n_features_max is not None and len(features) > n_features_max:
            features = np.sort(np.random.choice(features, n_features_max, replace=False))

        n_celltypes = len(celltypes)
        n_features = len(features)

        # Plot
        fig, axs = plt.subplots(n_celltypes, n_features,
                                figsize=(np.minimum(3 * n_features, 20), np.minimum(2 * n_celltypes, 20)),
                                squeeze=False)

        for ax_row, celltype in zip(axs, celltypes):
            ax_row[0].set_ylabel(f"celltype={group_names[np.argmax(celltype == group_ticks)]}")
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

    def plot_group_traces(self, threshold_confidence: float, key=None, celltypes=None, n_celltypes_max=32,
                          plot_baden_data=True, xlim_chirp=None, xlim_bar=None,
                          debug_shift_chirp=0, debug_shift_bar=0):
        key = get_primary_key(self.classifier_table, key)

        # Get new data
        roi_keys, preproc_chirps, preproc_bars, bar_ds_pvalues, roi_size_um2s = self._fetch_data(
            key, restriction=(self & f'max_confidence>={threshold_confidence}'))
        data_celltypes = (self & roi_keys).fetch('celltype')

        # Get training data
        self.current_model_key = dict(classifier_params_hash=key["classifier_params_hash"],
                                      training_data_hash=key["training_data_hash"])
        training_data = self.classifier_training_data_table().get_training_data(self.current_model_key)
        group_ticks = np.unique(training_data['y'])
        group_names = training_data.get("group_name", group_ticks)

        if celltypes is None:
            celltypes = group_ticks

        celltypes = sorted(set(celltypes).intersection(set(data_celltypes)))

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
            b_celltypes, b_chirp_traces, b_chirp_qi, b_bar_traces, b_bar_qi, b_bar_dsi, b_bar_dp, b_soma_size_um2 = \
                load_baden_data(baden_data_file)

            for ax_row, celltype in zip(axs, celltypes):

                if not np.any(data_celltypes == celltype):
                    continue

                ax = ax_row[0]
                b_chirps = b_chirp_traces[b_celltypes == celltype]
                b_mean_chirp = np.mean(b_chirps, axis=0)
                ax.plot(b_chirps.T, lw=0.5, c='gray', alpha=0.5, zorder=-200)
                ax.plot(b_mean_chirp, c='k', lw=2, alpha=0.5, zorder=1)

                mean_chirp = np.roll(np.mean(preproc_chirps[data_celltypes == celltype], axis=0), debug_shift_chirp)
                ax.set(title=f"cc={np.corrcoef(mean_chirp, b_mean_chirp)[0, 1]:.2f}", xlim=xlim_chirp)

                ax = ax_row[1]
                b_bars = b_bar_traces[b_celltypes == celltype]
                b_mean_bar = np.mean(b_bars, axis=0)
                ax.plot(b_bars.T, lw=0.5, c='gray', alpha=0.5, zorder=-200)
                ax.plot(b_mean_bar, c='k', lw=2, alpha=0.5, zorder=1)

                mean_bar = np.roll(np.mean(preproc_bars[data_celltypes == celltype], axis=0), debug_shift_bar)
                ax.set(title=f"cc={np.corrcoef(mean_bar, b_mean_bar)[0, 1]:.2f}", xlim=xlim_bar)

                ax = ax_row[2].twinx()
                ax.hist(b_bar_dp[b_celltypes == celltype], bar_ds_pvalues_bins, color='gray', alpha=0.5)

                ax = ax_row[3].twinx()
                ax.hist(b_soma_size_um2[b_celltypes == celltype], bins=roi_size_um2s_bins, color='gray', alpha=0.5)

        for ax_row, celltype in zip(axs, celltypes):
            ax_row[0].set_ylabel(f"celltype={group_names[np.argmax(celltype == group_ticks)]}")

            if not np.any(data_celltypes == celltype):
                continue

            ax = ax_row[0]
            ct_chirps = np.roll(preproc_chirps[data_celltypes == celltype], debug_shift_chirp, axis=1)
            ax.plot(ct_chirps.T, lw=0.5, c='r', alpha=0.5, zorder=-100)
            ax.plot(np.mean(ct_chirps, axis=0), c='darkred', lw=2, zorder=2)

            ax = ax_row[1]
            ct_bars = np.roll(preproc_bars[data_celltypes == celltype], debug_shift_bar, axis=1)
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


def load_baden_data(baden_data_file, merged_celltypes=True):
    from scipy.io import loadmat
    baden_data = loadmat(baden_data_file, struct_as_record=True, matlab_compatible=False,
                         squeeze_me=True, simplify_cells=True)['data']
    celltypes = baden_data['info']['final_idx']
    roi_size_um2 = baden_data['info']['area2d']

    chirp_traces = baden_data['chirp']['traces'].T
    chirp_qi = baden_data['chirp']['qi']

    bar_traces = baden_data['ds']['tc'].T
    bar_qi = baden_data['ds']['qi']
    bar_dsi = baden_data['ds']['dsi']
    bar_dp = baden_data['ds']['dP']

    if merged_celltypes:
        # TODO: Merging much come from training data; otherwise not flexible enough
        celltypes = np.array([celltype2merged_celltype(celltype) for celltype in celltypes])

    return celltypes, chirp_traces, chirp_qi, bar_traces, bar_qi, bar_dsi, bar_dp, roi_size_um2


def celltype2merged_celltype(celltype):
    _celltype2merged_celltype = {
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

    return _celltype2merged_celltype.get(celltype, -1)
