from abc import abstractmethod

import datajoint as dj
import numpy as np

from djimaging.tables.classifier_v2.rgc_classifier_v2 import load_classifier_from_file, check_classifier_dict
from djimaging.utils.baden16_utils import baden_cluster_to_group, baden_group_to_supergroup
from djimaging.utils.dj_utils import merge_keys


class CelltypeAssignmentV2Template(dj.Computed):
    database = ""
    __expected_classes = np.arange(1, 75 + 1)  # Expected classes for the classifier

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

    @property
    @abstractmethod
    def roi_table(self):
        pass

    def populate(
            self,
            *restrictions,
            keys=None,
            suppress_errors=False,
            return_exception_objects=False,
            reserve_jobs=False,
            order="original",
            limit=None,
            max_calls=None,
            display_progress=False,
            processes=1,
            make_kwargs=None,
    ):
        if processes > 1:
            raise NotImplementedError(
                "Parallel processing is not implemented for this table."
            )

        if len(restrictions) == 0:
            restrictions = dict()

        if len(self.classifier_table & restrictions) > 1:
            raise ValueError(
                "Multiple classifiers found for the given restrictions. "
                "Please specify a single classifier.")

        if make_kwargs is None:
            make_kwargs = dict()

        for key in ['classifier', 'chirp_feats', 'bar_feats']:
            if key in make_kwargs:
                raise ValueError(
                    f"The '{key}' key is reserved and should not be provided in make_kwargs. "
                    "It will be automatically set based on the classifier_table.")

        classifier_file = (self.classifier_table & restrictions).fetch1('classifier_file')
        clf_dict = load_classifier_from_file(classifier_file)
        check_classifier_dict(clf_dict)

        make_kwargs['classifier'] = clf_dict['classifier']
        make_kwargs['chirp_feats'] = clf_dict['chirp_feats']
        make_kwargs['bar_feats'] = clf_dict['bar_feats']

        if not np.all(make_kwargs['classifier'].classes_, self.__expected_classes):
            raise ValueError("The classifier's classes do not match the expected classes.")

        self.populate(
            *restrictions,
            keys=keys,
            suppress_errors=suppress_errors,
            return_exception_objects=return_exception_objects,
            reserve_jobs=reserve_jobs,
            order=order,
            limit=limit,
            max_calls=max_calls,
            display_progress=display_progress,
            processes=processes,
            make_kwargs=make_kwargs,
        )

    def make(self, key, classifier, chirp_feats, bar_feats):
        roi_keys, chirps, bars, ds_pvalues, roi_size_um2s = self._fetch_data(key)
        if len(roi_keys) == 0:
            return
        probs = classify_cells(
            chirps, bars, ds_pvalues, roi_size_um2s,
            chirp_features=chirp_feats, bar_features=bar_feats, classifier=classifier)

        for roi_key, prob in zip(roi_keys, probs):
            cluster_id, group_id, supergroup_id, prob_cluster, prob_group, prob_supergroup, prob_rgc = probs_to_info(
                probs)

            self.insert1(dict(**merge_keys(key, roi_key),
                              cluster_id=cluster_id, group_id=group_id, supergroup_id=supergroup_id,
                              prob_cluster=prob_cluster, prob_group=prob_group,
                              prob_supergroup=prob_supergroup, prob_rgc=prob_rgc,
                              probs_per_cluster=probs.tolist()))

    def _fetch_data(self, key, restriction=None):
        if restriction is None:
            restriction = dict()

        roi_keys, preproc_chirps, preproc_bars, bar_ds_pvalues, roi_size_um2s = (
                (self.baden_trace_table & key & restriction) * self.roi_table).fetch(
            'KEY', 'preproc_chirp', 'preproc_bar', 'ds_pvalue', 'roi_size_um2')

        preproc_chirps = np.vstack(preproc_chirps)
        preproc_bars = np.vstack(preproc_bars)

        return roi_keys, preproc_chirps, preproc_bars, bar_ds_pvalues, roi_size_um2s


def classify_cells(preproc_chirps, preproc_bars, bar_ds_pvalues, roi_size_um2s,
                   chirp_features, bar_features, classifier):
    features = extract_features(
        preproc_chirps, preproc_bars, bar_ds_pvalues, roi_size_um2s, chirp_features, bar_features)
    probs = classifier.predict_proba(features)
    return probs


def probs_to_info(probs):
    cluster_id = np.argmax(probs) + 1  # Cluster IDs are 1-indexed
    group_id = baden_cluster_to_group(cluster_id)
    supergroup_id = baden_group_to_supergroup(group_id)
    prob_cluster = probs[cluster_id - 1]
    prob_group = 0.0  # TODO: replace with actual group probability calculation
    prob_supergroup = 0.0  # TODO: replace with actual supergroup probability calculation
    prob_rgc = 0.0  # TODO: replace with actual RGC probability calculation

    return cluster_id, group_id, supergroup_id, prob_cluster, prob_group, prob_supergroup, prob_rgc


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
