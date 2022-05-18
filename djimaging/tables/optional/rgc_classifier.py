import os

import datajoint as dj
import numpy as np
from typing import Mapping, Dict, Any
import pickle as pkl
from scipy import interpolate
from copy import deepcopy
from cached_property import cached_property

from djimaging.utils.import_helpers import dynamic_import, split_module_name
from djimaging.utils.dj_utils import make_hash, PlaceholderTable


Key = Dict[str, Any]


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
        """
        return definition

    def add_parameters(self, qi_thres_chirp: float, qi_thres_bar: float, cell_selection_constraint: str,
                       skip_duplicates: bool = False) -> None:
        insert_dict = dict(qi_thres_chirp=qi_thres_chirp,
                           qi_thres_bar=qi_thres_bar,
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
    classifier_training_data_table = PlaceholderTable

    def add_classifer(self, classifier_fn: str, classifier_config: Mapping,
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
        return classifier


class ClassifierTrainingDataTemplate(dj.Manual):
    database = ""
    store = None

    @property
    def definition(self):
        definition = """
        # holds feature basis and training data for classifier
        training_data_hash              :   varchar(32)     # hash of the classifier training data files
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
    store = None

    @property
    def definition(self):
        definition = """
        -> self.classifier_training_data_table 
        -> self.classifier_method_table
        ---
        classifier_file         :   attach@{store}
        """.format(store=self.store)
        return definition

    classifier_training_data_table = PlaceholderTable
    classifier_method_table = PlaceholderTable

    def make(self, key):
        output_path = (self.classifier_training_data_table() & key).fetch1("output_path")
        classifier_file = os.path.join(output_path, 'rgc_classifier.pkl')
        classifier = self.classifier_method_table().train_classifier(key)

        print(f'Saving classifier to {classifier_file}')
        with open(classifier_file, "wb") as f:
            pkl.dump(classifier, f)

        self.insert1(dict(key, classifier_file=classifier_file))


def preprocess_chirp(chirp_traces, samples=249):
    """
    Preprocesses chirp traces by averaging across repetitions, cutting off
    last 7 frames and downsampling to 249 frames to match Baden traces;
    subtracting mean of first 8 frames (baseline subtraction); and normalizing
    to be in the range [-1, 1]
    :param chirp_traces: array; shape rois x chirp frames x repetitions
    :param samples: int; target number of samples
    :return: array; shape rois x chirp frames
    """
    chirp_traces = chirp_traces.mean(axis=-1)  # average across reps
    chirp_traces_global = chirp_traces[:, :-7]

    # downsample to 249 samples
    x_chirp = np.linspace(0, 32, chirp_traces_global.shape[-1])
    f_chirp = interpolate.interp1d(x_chirp, chirp_traces_global)
    x_chirp = np.linspace(0, 32, samples)
    chirp_traces = f_chirp(x_chirp)
    to_subtract = chirp_traces[:, :8].mean(axis=-1)
    to_subtract = np.tile(to_subtract, [chirp_traces.shape[-1], 1]).transpose()
    chirp_traces -= to_subtract
    to_divide = np.max(abs(chirp_traces), axis=-1)
    to_divide = np.tile(to_divide, [chirp_traces.shape[-1], 1]).transpose()
    chirp_traces /= to_divide
    return chirp_traces


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
        confidence:      mediumblob        # confidence score (probability) per group
        preproc_chirp:   mediumblob        # preprocessed chirp trace (averaged, downsampled and normalized)
        preproc_bar:     mediumblob        # preprocessed bar (shifted by -5 frames)
        """
        return definition

    cell_filter_parameter_table = PlaceholderTable
    classifier_training_data_table = PlaceholderTable
    classifier_table = PlaceholderTable
    roi_table = PlaceholderTable
    presentation_table = PlaceholderTable
    chirp_qi_table = PlaceholderTable
    snippets_table = PlaceholderTable
    or_dir_index_table = PlaceholderTable
    user_info_table = PlaceholderTable
    detrend_params_table = PlaceholderTable
    field_table = PlaceholderTable
    exp_info_table = PlaceholderTable

    _stim_name_chirp = 'gChirp'
    _stim_name_bar = 'movingbar'
    _chirp_qi_key = 'chirp_qi'

    @property
    def key_source(self):
        return (self.classifier_training_data_table() * self.classifier_table() *
                self.field_table() * self.detrend_params_table() * self.cell_filter_parameter_table()) & \
               (self.snippets_table() & f"stim_name = '{self._stim_name_chirp}'") & \
               (self.snippets_table() & f"stim_name = '{self._stim_name_bar}'")

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

        if key.get("roi_id", False):
            self.run_cell_type_assignment(key, mode="roi")
        else:
            self.run_cell_type_assignment(key, mode="field")

    def run_classifier(self, X):
        confidence = self.classifier.predict_proba(X).reshape(-1, order="C")
        type_predictions = self.classifier.predict(X)

        return type_predictions, confidence

    def run_cell_type_assignment(self, key, mode):
        if mode == "roi":
            raise NotImplementedError

        chirp_key = deepcopy(key)
        chirp_key['stim_name'] = self._stim_name_chirp

        chirp_traces, roi_ids = (self.snippets_table() & chirp_key).fetch('snippets', "roi_id")
        chirp_traces = np.asarray([chirp_trace for chirp_trace in chirp_traces])
        chirp_traces = preprocess_chirp(chirp_traces)

        # Bar Response
        bar_key = deepcopy(key)
        bar_key['stim_name'] = self._stim_name_bar
        bar_traces = (self.or_dir_index_table() & bar_key).fetch('time_component')

        # shift by 5 frames backwards
        bar_traces = np.asarray([bar for bar in bar_traces])
        bar_traces = np.roll(bar_traces, -5)

        # fetch more parameters
        ds_pvalues, bar_qis = (self.or_dir_index_table() & bar_key).fetch('ds_pvalue', 'd_qi')
        chirp_qis = (self.chirp_qi_table() & chirp_key).fetch(self._chirp_qi_key)

        # Get ROI size
        roi_sizes_um = (self.roi_table() & key).fetch('roi_size_um2')

        # feature activation matrix
        feature_activation_bar = np.dot(bar_traces, self.bar_features)
        feature_activation_chirp = np.dot(chirp_traces, self.chirp_features)

        feature_activation_matrix = np.concatenate(
            [feature_activation_chirp, feature_activation_bar, ds_pvalues[:, np.newaxis],
             roi_sizes_um[:, np.newaxis]], axis=-1)

        # choose the model that applies to the attribute of the cell
        qi_thres_chirp, qi_thres_bar, cell_selection_constraint = (self.cell_filter_parameter_table() & key).fetch1(
            'qi_thres_chirp', 'qi_thres_bar', 'cell_selection_constraint')

        if cell_selection_constraint == 'and':
            quality_mask = (bar_qis > qi_thres_bar) & (chirp_qis > qi_thres_chirp)
        elif cell_selection_constraint == 'or':
            quality_mask = (bar_qis > qi_thres_bar) | (chirp_qis > qi_thres_chirp)
        else:
            raise NotImplementedError(f"cell_selection_constraint={cell_selection_constraint}")

        if np.any(quality_mask):
            type_predictions, confidence = self.run_classifier(feature_activation_matrix[quality_mask])

            for i, roi_id in enumerate(roi_ids[quality_mask]):
                key["roi_id"] = roi_id
                self.insert1(dict(key,
                                  celltype=type_predictions[i],
                                  confidence=confidence[i],
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
                                  preproc_chirp=chirp_traces[~quality_mask, :][i],
                                  preproc_bar=bar_traces[~quality_mask][i],
                                  ))
            
    def plot(self):
        from matplotlib import pyplot as plt
        import pandas as pd

        groups = pd.DataFrame(self.fetch()).groupby([
            'training_data_hash', 'classifier_params_hash', 'cell_filter_params_hash'])

        fig, axs = plt.subplots(len(groups), 1, figsize=(12, 3*len(groups)), squeeze=False)
        axs = axs.flatten()

        for ax, ((tdh, cph, cfph), df) in zip(axs, groups):
            celltypes = df["celltype"]
            ax.hist(celltypes[celltypes > 0], bins=np.arange(1, 47, 0.5), align='left')
            ax.hist(celltypes[celltypes < 0], bins=[-1, -0.5, 0], align='left', color='red', alpha=0.5)
            ax.set_xticks(np.append(-1, np.arange(1, 47)))
            ax.set_xticklabels(np.append("NA", np.arange(1, 47)), rotation=90)
            ax.set(ylabel='Count', xlabel='celltype')
            ax.set_title(f"training_data_hash={tdh}\nclassifier_params_hash={cph}\ncell_filter_params_hash={cfph}")

        plt.tight_layout()
        plt.show()

