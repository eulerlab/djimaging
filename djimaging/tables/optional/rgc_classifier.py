import os
import pickle as pkl
import warnings
from abc import abstractmethod
from typing import Mapping, Dict, Any

import datajoint as dj
import numpy as np
from cached_property import cached_property
from scipy import interpolate

from djimaging.utils.dj_utils import make_hash
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


class CellFilterParamsTemplate(dj.Lookup):
    database = ""

    @property
    def definition(self):
        definition = """
        cell_filter_params_hash         :  varchar(32)         # hash of the classifier params config
        ---
        qi_thres_chirp                  :  float               # QI threshold for full-field chirp response
        qi_thres_bar                    :  float               # QI threshold for moving bar response
        cell_selection_constraint       :  enum("and", "or")   # constraint flag (and, or) for QI
        condition = 'control'           :  varchar(255)        # Condition to classify. Empty strings = all conditions. 
        """
        return definition

    def add_parameters(self, condition: str, qi_thres_chirp: float, qi_thres_bar: float,
                       cell_selection_constraint: str, skip_duplicates: bool = False) -> None:
        insert_dict = dict(qi_thres_chirp=qi_thres_chirp, qi_thres_bar=qi_thres_bar, condition=condition,
                           cell_selection_constraint=cell_selection_constraint)
        cell_filter_params_hash = make_hash(insert_dict)
        insert_dict.update(dict(cell_filter_params_hash=cell_filter_params_hash))
        self.insert1(insert_dict, skip_duplicates=skip_duplicates)


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
    @abstractmethod
    def classifier_training_data_table(self): pass

    @property
    @abstractmethod
    def classifier_method_table(self): pass

    def make(self, key):
        output_path = (self.classifier_training_data_table() & key).fetch1("output_path")
        classifier_file = os.path.join(output_path, 'rgc_classifier.pkl')
        classifier, score_train = self.classifier_method_table().train_classifier(key)

        print(f'Saving classifier to {classifier_file}')
        with open(classifier_file, "wb") as f:
            pkl.dump(classifier, f)

        self.insert1(dict(key, classifier_file=classifier_file, score_train=score_train))


def preprocess_chirp(snippets, dt):
    """
    Preprocesses chirp traces by resampling if necessary, averaging across repetitions, cutting off
    last 7 frames and downsampling to 249 frames to match Baden traces;
    subtracting mean of first 8 frames (baseline subtraction); and normalizing
    to be in the range [-1, 1]
    :param snippets: list of arrays; shape rois x frames x repetitions
    :param dt: float, time between two frames
    :return: array; shape rois x frames
    """
    assert len(snippets) > 0

    chirp_avgs = np.mean(np.stack(snippets), axis=-1)  # average across reps

    time_avg = np.arange(chirp_avgs.shape[1]) * dt  # Allow different frequencies
    time_baden = np.linspace(0, 32, 249)

    # Resample to Baden frequency which was (in stimulus space) slightly different
    chirp_avgs = interpolate.interp1d(time_avg, chirp_avgs, axis=1, assume_sorted=True)(time_baden)

    # Normalize
    chirp_avgs = (chirp_avgs.T - np.mean(chirp_avgs[:, :8], axis=1)).T
    chirp_avgs = (chirp_avgs.T / np.max(np.abs(chirp_avgs), axis=1)).T

    return chirp_avgs


def preprocess_bar(raw_bar_traces, dt):
    """
    Preprocesses bar time component by resampling if necessary and by rolling to match Baden traces;
    :param raw_bar_traces: list of arrays; shape rois x frames
    :param dt: float, time between two frames
    :return: array; shape rois x frames
    """
    raw_bar_traces = np.stack(raw_bar_traces)
    time = np.arange(raw_bar_traces.shape[1]) * dt
    time_baden = np.arange(32) * 0.128
    bar_traces = np.stack([interpolate.interp1d(
        time, trace, assume_sorted=True, bounds_error=False, fill_value='extrapolate')(time_baden)
                           for trace in raw_bar_traces])
    bar_traces = np.roll(bar_traces, -5)
    return bar_traces


def split_ChirpInterleaved(chirp_trace):
    chirp_trace_ave = np.mean(chirp_trace, axis=1)
    chirp_trace_global = chirp_trace_ave[int(chirp_trace_ave.shape[0] / 2):]
    return chirp_trace_global


class CelltypeAssignmentTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = """
        # Feature-based cluster assignment of cells
        -> self.roi_table
        -> self.detrend_params_table
        -> self.classifier_table 
        -> self.cell_filter_parameter_table
        ---
        celltype:        int               # predicted group (-1, 1-46)
        max_confidence:  float             # confidence score for assigned cluster for easy restriction
        confidence:      mediumblob        # confidence score (probability) for all groups
        preproc_chirp:   mediumblob        # preprocessed chirp trace (averaged, downsampled and normalized)
        preproc_bar:     mediumblob        # preprocessed bar (shifted by -5 frames)
        """
        return definition

    _stim_name_chirp = 'gChirp'
    _stim_name_bar = 'movingbar'
    _chirp_qi_key = 'qidx'
    _bar_qi_key = 'd_qi'

    @property
    @abstractmethod
    def cell_filter_parameter_table(self):
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
    def roi_table(self):
        pass

    @property
    @abstractmethod
    def field_table(self):
        pass

    @property
    @abstractmethod
    def chirp_qi_table(self):
        pass

    @property
    @abstractmethod
    def snippets_table(self):
        pass

    @property
    @abstractmethod
    def or_dir_index_table(self):
        pass

    @property
    @abstractmethod
    def detrend_params_table(self):
        pass

    @property
    def key_source(self):
        try:
            return (self.field_table() * self.detrend_params_table() * self.classifier_training_data_table() *
                    self.classifier_table() * self.cell_filter_parameter_table()).proj() & \
                ((self.snippets_table() & f"stim_name = '{self._stim_name_chirp}'").proj(chirp='stim_name') *
                 (self.snippets_table() & f"stim_name = '{self._stim_name_bar}'").proj(bar='stim_name'))
        except (AttributeError, TypeError):
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

    def make(self, key):
        model_key = dict(
            classifier_params_hash=key["classifier_params_hash"],
            training_data_hash=key["training_data_hash"])

        self.current_model_key = model_key

        if "roi_id" in key:
            self.run_cell_type_assignment(key, mode="roi")
        else:
            self.run_cell_type_assignment(key, mode="field")

    def run_classifier(self, X):
        confidence = self.classifier.predict_proba(X)
        type_predictions = self.classifier.predict(X)
        return type_predictions, confidence

    def run_cell_type_assignment(self, key, mode):
        if mode == "roi":
            raise NotImplementedError('mode == "roi"')

        # Fetch selection parameters
        qi_thres_chirp, qi_thres_bar, cell_selection_constraint, condition = (
                self.cell_filter_parameter_table() & key).fetch1(
            'qi_thres_chirp', 'qi_thres_bar', 'cell_selection_constraint', 'condition')

        # Get data
        chirp_tab = (self.snippets_table * self.chirp_qi_table & f'stim_name="{self._stim_name_chirp}"' & key).proj(
            chirp='stim_name', chirp_snippets="snippets", chirp_times="snippets_times", chirp_qi=self._chirp_qi_key)

        bar_tab = (self.snippets_table * self.or_dir_index_table & f'stim_name="{self._stim_name_bar}"' & key).proj(
            movingbar='stim_name', bar_snippets="snippets", bar_times="snippets_times",
            bar_time_component='time_component', bar_ds_pvalue='ds_pvalue', bar_qi=self._bar_qi_key)

        key_tab = (chirp_tab * bar_tab) & dict(condition=condition)

        if len(key_tab) == 0:
            warnings.warn(f'No ROIs found for condition=`{condition}` and key={key}')
            return

        # Fetch chirp data
        chirp_snippets, chirp_snippets_times, chirp_qis = key_tab.fetch(
            'chirp_snippets', 'chirp_times', 'chirp_qi')

        chirp_dt = np.mean([np.diff((t - t[0, :]), axis=0).mean() for t in chirp_snippets_times])
        chirp_traces = preprocess_chirp(chirp_snippets, chirp_dt)

        # Fetch bar data
        raw_bar_traces, ds_pvalues, bar_qis = key_tab.fetch(
            'bar_time_component', 'bar_ds_pvalue', 'bar_qi')

        bar_snippets_times = key_tab.fetch('bar_times')
        bar_dt = np.mean([np.diff((t - t[0, :]), axis=0).mean() for t in bar_snippets_times])
        bar_traces = preprocess_bar(raw_bar_traces, dt=bar_dt)

        # Get ROI size
        roi_ids, roi_sizes_um = (self.roi_table & key_tab).fetch('roi_id', 'roi_size_um2')

        # feature activation matrix
        feature_activation_bar = np.dot(bar_traces, self.bar_features)
        feature_activation_chirp = np.dot(chirp_traces, self.chirp_features)

        feature_activation_matrix = np.concatenate(
            [feature_activation_chirp, feature_activation_bar, ds_pvalues[:, np.newaxis],
             roi_sizes_um[:, np.newaxis]], axis=-1)

        if cell_selection_constraint == 'and':
            quality_mask = (bar_qis > qi_thres_bar) & (chirp_qis > qi_thres_chirp)
        elif cell_selection_constraint == 'or':
            quality_mask = (bar_qis > qi_thres_bar) | (chirp_qis > qi_thres_chirp)
        else:
            raise NotImplementedError(f"cell_selection_constraint={cell_selection_constraint}")

        if np.any(quality_mask):
            n_rois = np.sum(quality_mask)
            type_predictions, confidence = self.run_classifier(feature_activation_matrix[quality_mask, :])

            assert confidence.shape == (n_rois, 46), f"{confidence.shape} != {(n_rois, 46)} (expected)"
            assert type_predictions.size == n_rois, f"{type_predictions.size} != {n_rois} (expected)"

            for i, roi_id in enumerate(roi_ids[quality_mask]):
                key["roi_id"] = roi_id

                confidence_i = confidence[i, :]
                assert confidence_i.size == 46, \
                    f"Expected 46 confidence scores, got {confidence_i.size}"
                assert np.isclose(np.sum(confidence_i), 1., rtol=0.01, atol=0.01), \
                    f"Confidences should sum up to 1, but are {np.sum(confidence_i)}."

                self.insert1(dict(key,
                                  celltype=type_predictions[i],
                                  confidence=confidence_i,
                                  max_confidence=np.max(confidence_i),
                                  preproc_chirp=chirp_traces[quality_mask, :][i],
                                  preproc_bar=bar_traces[quality_mask][i],
                                  ))
        if np.any(~quality_mask):
            dummy_confidence = np.full((1, 46), -1)

            for i, roi_id in enumerate(roi_ids[~quality_mask]):
                key["roi_id"] = roi_id
                self.insert1(dict(key,
                                  celltype=-1,
                                  confidence=dummy_confidence,
                                  max_confidence=-1,
                                  preproc_chirp=chirp_traces[~quality_mask, :][i],
                                  preproc_bar=bar_traces[~quality_mask][i],
                                  ))

    def plot(self):
        from matplotlib import pyplot as plt
        import pandas as pd

        groups = pd.DataFrame(self.fetch()).groupby([
            'training_data_hash', 'classifier_params_hash', 'cell_filter_params_hash', 'preprocess_id'])

        fig, axs = plt.subplots(len(groups), 1, figsize=(12, 3 * len(groups)), squeeze=False)
        axs = axs.flatten()

        for ax, ((tdh, cph, cfph, pid), df) in zip(axs, groups):
            celltypes = df["celltype"]
            ax.hist(celltypes[celltypes > 0], bins=np.arange(1, 47, 0.5), align='left')
            ax.hist(celltypes[celltypes < 0], bins=[-1, -0.5, 0], align='left', color='red', alpha=0.5)
            ax.set_xticks(np.append(-1, np.arange(1, 47)))
            ax.set_xticklabels(np.append("NA", np.arange(1, 47)), rotation=90)
            ax.set(ylabel='Count', xlabel='celltype')
            ax.set_title(
                f"training_data_hash={tdh}\nclassifier_params_hash={cph}\ncell_filter_params_hash={cfph}\npreprocess_id={pid}")

        plt.tight_layout()
        plt.show()
