"""
Improved classifier for RGCs used Baden et al. 2016 dataset.

Example usage:

from djimaging.tables import classifier_v2

@schema
class Baden16TracesV2(classifier_v2.Baden16TracesV2Template):
    _stim_name_chirp = 'gChirp'
    _stim_name_bar = 'movingbar'

    traces_table = Traces
    presentation_table = Presentation
    stimulus_table = Stimulus


@schema
class ClassifierV2(classifier_v2.ClassifierV2Template):
    classifier_training_data_table = ClassifierTrainingData
    classifier_method_table = ClassifierMethod


@schema
class CelltypeAssignmentV2(classifier_v2.CelltypeAssignmentV2Template):
    classifier_table = Classifier
    baden_trace_table = Baden16Traces
    field_table = Field
"""
import pickle
from abc import abstractmethod

import datajoint as dj
import numpy as np


class ClassifierV2Template(dj.Manual):
    database = ""

    @property
    def definition(self):
        definition = """
        classifier_id : int unsigned  # Unique identifier for the classifier
        ---
        classifier_file : varchar(255)
        """
        return definition

    def add(self, classifier_file, classifier_id=1):
        # Test if classifier is valid
        clf_dict = load_classifier_from_file(classifier_file)
        check_classifier_dict(clf_dict)

        self.insert1(dict(
            classifier_id=classifier_id,
            classifier_file=classifier_file))


class CelltypeAssignmentV2Template(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = """
        -> self.baden_trace_table
        -> self.classifier_table
        ---
        cluster_id :       tinyint unsigned      # cluster ID, ranging from 1 to 75
        group_id :         tinyint unsigned      # group ID, ranging from 1 to 46
        supergroup_id :    enum("OFF", "ON-OFF", "Fast ON", "Slow ON", "Uncertain RGCs", "ACs")
        prob_cluster :     float                 # probability of being in the given cluster
        prob_group :       float                 # aggregated probability of being in the given group
        prob_supergroup :  float                 # aggregated probability of being in the given supergroup
        prob_rgc           float                 # aggregated probability of being an RGC
        probs_per_cluster : blob                 # probabilities for each cluster, e.g. [0.1, 0.9, 0.05, ...]
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
    def classifier_table(self):
        pass

    @property
    @abstractmethod
    def baden_trace_table(self):
        pass

    @property
    @abstractmethod
    def field_table(self):
        pass


def load_classifier_from_file(classifier_file):
    with open(classifier_file, 'rb') as f:
        clf_dict = pickle.load(f)
    return clf_dict


def check_classifier_dict(clf_dict):
    assert type(clf_dict) == dict, "Classifier file must contain a dictionary with classifier data."

    # Check keys
    assert 'classifier' in clf_dict, "Classifier dictionary must contain a 'classifier' key."
    assert 'chirp_feats' in clf_dict, "Classifier dictionary must contain a 'chirp_feats' key."
    assert 'bar_feats' in clf_dict, "Classifier dictionary must contain a 'bar_feats' key."
    assert 'feature_names' in clf_dict, "Classifier dictionary must contain a 'feature_names' key."
    assert 'train_x' in clf_dict, "Classifier dictionary must contain a 'train_x' key."
    assert 'train_y' in clf_dict, "Classifier dictionary must contain a 'train_y' key."
    assert 'y_names' in clf_dict, "Classifier dictionary must contain a 'y_names' key."

    # Chek value
    assert isinstance(clf_dict['train_x'], np.ndarray), "The 'train_x' key must contain a numpy array."
    assert isinstance(clf_dict['train_y'], np.ndarray), "The 'train_y' key must contain a numpy array."
    assert clf_dict['train_x'].shape[0] == clf_dict[
        'train_y'].size, "The number of samples in 'train_x' and 'train_y' must match."

    for val in np.unique(clf_dict['train_y']):
        assert val in clf_dict['y_names'].keys(), f"Value {val} in 'train_y' not found in 'y_names'."

    # Check if classifier is a valid scikit-learn classifier
    from sklearn.base import is_classifier
    assert is_classifier(clf_dict['classifier']), "The 'classifier' key must contain a valid scikit-learn classifier."

    return clf_dict


def save_classifier_to_file(classifier, chirp_feats, bar_feats, feature_names, train_x, train_y, y_names,
                            classifier_file, **kwargs):
    """
    Saves the classifier and its metadata to a file.
    """
    clf_dict = {
        'classifier': classifier,
        'chirp_feats': chirp_feats,
        'bar_feats': bar_feats,
        'feature_names': feature_names,
        'train_x': train_x,
        'train_y': train_y,
        'y_names': y_names,
        **kwargs
    }

    check_classifier_dict(clf_dict)

    with open(classifier_file, 'wb') as f:
        pickle.dump(clf_dict, f)


def extract_features(preproc_chirps, preproc_bars, bar_ds_pvalues, roi_size_um2s, chirp_features, bar_features):
    features = np.concatenate([
        np.dot(preproc_chirps, chirp_features),
        np.dot(preproc_bars, bar_features),
        bar_ds_pvalues[:, np.newaxis],
        roi_size_um2s[:, np.newaxis]
    ], axis=-1)

    feature_names = [f'chirp_{i}' for i in range(chirp_features.shape[1])] + \
                    [f'bar_{i}' for i in range(bar_features.shape[1])] + ['bar_ds_pvalue', 'roi_size_um2']

    return features, feature_names
