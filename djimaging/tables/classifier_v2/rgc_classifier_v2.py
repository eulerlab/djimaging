import pickle

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

    def add(self, classifier_file='/gpfs01/euler/data/Resources/Classifier_v2/rgc_classifier_v2.pkl', classifier_id=1):
        # Test if classifier is valid
        clf_dict = load_classifier_from_file(classifier_file)
        check_classifier_dict(clf_dict)

        self.insert1(dict(
            classifier_id=classifier_id,
            classifier_file=classifier_file))


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
