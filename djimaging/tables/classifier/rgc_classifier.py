import os
import pickle as pkl
from abc import abstractmethod
from typing import Mapping, Dict, Any

import datajoint as dj
import numpy as np

from djimaging.tables.classifier.celltype_assignment import extract_features
from djimaging.utils.baden16_utils import load_baden_data
from djimaging.utils.dj_utils import make_hash
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
