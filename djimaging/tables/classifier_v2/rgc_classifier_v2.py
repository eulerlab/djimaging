import pickle

import datajoint as dj
import numpy as np


class ClassifierV2Template(dj.Manual):
    database = ""

    @property
    def definition(self) -> str:
        definition = """
        classifier_id : int unsigned  # Unique identifier for the classifier
        ---
        classifier_file : varchar(255)
        """
        return definition

    def add(self, classifier_file: str = '/gpfs01/euler/data/Resources/Classifier_v2/rgc_classifier_v2.pkl',
            classifier_id: int = 1) -> None:
        """Validate a classifier file and insert a new entry into the table.

        Args:
            classifier_file: Path to the pickled classifier dictionary file.
            classifier_id: Unique integer identifier for the classifier entry.

        Raises:
            AssertionError: If the classifier file fails validation by
                :func:`check_classifier_dict`.
        """
        # Test if classifier is valid
        clf_dict = load_classifier_from_file(classifier_file)
        check_classifier_dict(clf_dict)

        self.insert1(dict(
            classifier_id=classifier_id,
            classifier_file=classifier_file))


def load_classifier_from_file(classifier_file: str) -> dict:
    """Load and return the classifier dictionary from a pickle file.

    Args:
        classifier_file: Path to the pickled classifier dictionary file.

    Returns:
        Dictionary containing the classifier and its associated metadata.
    """
    with open(classifier_file, 'rb') as f:
        clf_dict = pickle.load(f)
    return clf_dict


def check_classifier_dict(clf_dict: dict) -> dict:
    """Validate the structure and contents of a classifier dictionary.

    Checks that all required keys are present, that the training arrays have consistent
    shapes, that every unique training label appears in ``y_names``, and that the
    ``'classifier'`` value is a valid scikit-learn classifier.

    Args:
        clf_dict: Dictionary expected to contain at least the keys ``'classifier'``,
            ``'chirp_feats'``, ``'bar_feats'``, ``'feature_names'``, ``'train_x'``,
            ``'train_y'``, and ``'y_names'``.

    Returns:
        The validated ``clf_dict`` unchanged.

    Raises:
        AssertionError: If any validation check fails.
    """
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


def save_classifier_to_file(
        classifier,
        chirp_feats: np.ndarray,
        bar_feats: np.ndarray,
        feature_names: list,
        train_x: np.ndarray,
        train_y: np.ndarray,
        y_names: dict,
        classifier_file: str,
        **kwargs,
) -> None:
    """Save the classifier and its metadata to a pickle file.

    Assembles a classifier dictionary from the provided arguments, validates it with
    :func:`check_classifier_dict`, and writes it to ``classifier_file``.

    Args:
        classifier: Trained scikit-learn classifier.
        chirp_feats: Chirp feature basis matrix used during training.
        bar_feats: Bar feature basis matrix used during training.
        feature_names: Ordered list of feature names corresponding to the columns of
            the training feature matrix.
        train_x: Training feature matrix of shape ``(n_samples, n_features)``.
        train_y: Training label array of shape ``(n_samples,)``.
        y_names: Mapping from integer label values to human-readable label names.
        classifier_file: Destination path for the pickled classifier dictionary.
        **kwargs: Additional items to include in the saved dictionary.

    Raises:
        AssertionError: If the assembled dictionary fails validation by
            :func:`check_classifier_dict`.
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
