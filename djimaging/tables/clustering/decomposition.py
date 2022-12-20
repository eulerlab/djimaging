from abc import abstractmethod
from copy import deepcopy

import datajoint as dj
import numpy as np
from matplotlib import pyplot as plt

from djimaging.utils.dj_utils import get_primary_key
from djimaging.utils.math_utils import truncated_vstack


class FeaturesParamsTemplate(dj.Lookup):
    database = ""

    @property
    def definition(self):
        definition = """
        features_id: int # unique param set id
        ---
        kind: varchar(255)
        params_dict: longblob
        norm_trace: tinyint unsigned  # Used normalized averages or averages?
        stim_names: varchar(255)  # Stimuli to consider, separated by '_'
        ncomps: varchar(255)  # Number of components separated by '_'
        """
        return definition

    def add(self, features_id, kind, params_dict, skip_duplicates=False, norm_trace: bool = False,
            stim_names='gChirp_lChirp', ncomps='20_20'):
        assert isinstance(params_dict, dict)
        key = dict(features_id=features_id, kind=kind, params_dict=params_dict, norm_trace=int(norm_trace),
                   stim_names=stim_names, ncomps=ncomps)
        self.insert1(key, skip_duplicates=skip_duplicates)


def compute_features(traces: list, ncomps: list, kind: str, params_dict: dict) -> (list, list, list):
    """Reduce dimension of traces to features and stack them to feature matrix"""
    kind = kind.lower()

    if kind == 'sparse_pca':
        features, traces_reconstructed, infos = compute_features_sparse_pca(traces=traces, ncomps=ncomps, **params_dict)
    elif kind == 'pca':
        assert len(params_dict) == 0
        features, traces_reconstructed, infos = compute_features_pca(traces=traces, ncomps=ncomps)
    elif kind == 'none':
        features, traces_reconstructed, infos = deepcopy(traces), deepcopy(traces), [dict()] * len(traces)
    else:
        raise NotImplementedError(kind)

    return features, traces_reconstructed, infos


def compute_variance_explained_sparse_pca(X: np.ndarray, P: np.ndarray) -> (float, float):
    """Camacho et al. (2019): explained here https://github.com/scikit-learn/scikit-learn/issues/11512"""
    Xc = X - X.mean(axis=0)  # center data
    T = Xc @ P @ np.linalg.pinv(P.T @ P)

    explained_variance = np.trace(P @ T.T @ T @ P.T)
    total_variance = np.trace(Xc.T @ Xc)

    return explained_variance, total_variance


def compute_features_sparse_pca(traces: list, ncomps: list, alpha=1) -> (list, list, list):
    from sklearn.decomposition import SparsePCA
    from sklearn.preprocessing import StandardScaler

    features, traces_reconstructed, infos = [], [], []
    for traces_i, ncomps_i in zip(traces, ncomps):
        sc = StandardScaler()
        norm_traces_i = sc.fit_transform(X=traces_i)

        decomp = SparsePCA(n_components=ncomps_i, random_state=0, alpha=alpha, verbose=1)
        features_i = decomp.fit_transform(X=norm_traces_i)

        try:
            traces_reconstructed_i = sc.inverse_transform(X=decomp.inverse_transform(X=features_i))
        except AttributeError:  # SparsePCA.inverse_transform was only added in scikit-learn 1.2
            traces_reconstructed_i = None

        explained_variance, total_variance = compute_variance_explained_sparse_pca(
            X=norm_traces_i, P=decomp.components_.T)

        info_i = dict()
        info_i["components"] = decomp.components_
        info_i["explained_variance"] = explained_variance
        info_i["explained_variance_ratio"] = explained_variance / total_variance

        features.append(features_i)
        traces_reconstructed.append(traces_reconstructed_i)
        infos.append(info_i)

    return features, traces_reconstructed, infos


def compute_features_pca(traces: list, ncomps: list) -> (list, list, list):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    features, traces_reconstructed, infos = [], [], []
    for traces_i, ncomps_i in zip(traces, ncomps):
        sc = StandardScaler()
        norm_traces_i = sc.fit_transform(X=traces_i)

        decomp = PCA(n_components=ncomps_i, random_state=0)
        features_i = decomp.fit_transform(X=norm_traces_i)

        traces_reconstructed_i = sc.inverse_transform(X=decomp.inverse_transform(X=features_i))

        info_i = dict()
        info_i["components"] = decomp.components_
        info_i["explained_variance"] = decomp.explained_variance_
        info_i["explained_variance_ratio"] = decomp.explained_variance_ratio_

        features.append(features_i)
        traces_reconstructed.append(traces_reconstructed_i)
        infos.append(info_i)

    return features, traces_reconstructed, infos


class FeaturesTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = """
        -> self.params_table
        ---
        features: longblob  # Feature matrix.
        traces: longblob  # Input traces.
        traces_reconstructed: longblob  # Reconstructed traces.
        decomp_infos: longblob  # information about decomposition of different stimuli
        """
        return definition

    @property
    @abstractmethod
    def averages_table(self):
        pass

    @property
    @abstractmethod
    def params_table(self):
        pass

    @property
    @abstractmethod
    def roi_table(self):
        pass

    @property
    @abstractmethod
    def roi_quality_table(self):
        pass

    def fetch_traces(self, key):
        norm_trace, stim_names = (self.params_table & key).fetch1('norm_trace', 'stim_names')
        average_key = 'average_norm' if norm_trace else 'average'
        stim_names = stim_names.split('_')

        tab = (self.roi_quality_table & "q_tot = 1.")

        for stim_i in stim_names:
            tab = tab * (self.averages_table & f"stim_name='{stim_i}'").proj(
                **{f'{stim_i}_avgs': average_key, f'{stim_i}_time': 'average_times', f'{stim_i}_name': 'stim_name'})

        times = [truncated_vstack(tab.fetch(f'{stim_i}_time'), rtol=0.15) for stim_i in stim_names]
        traces = [truncated_vstack(tab.fetch(f'{stim_i}_avgs'), rtol=0.15) for stim_i in stim_names]
        roi_keys = tab.fetch(*self.roi_table.primary_key, as_dict=True)
        return traces, times, roi_keys, stim_names

    def make(self, key):
        kind, params_dict, ncomps = (self.params_table & key).fetch1('kind', 'params_dict', 'ncomps')

        ncomps = [int(ncomps_i) for ncomps_i in ncomps.split('_')] if len(ncomps) > 0 else []
        traces, times, roi_keys, stim_names = self.fetch_traces(key=key)

        features, traces_reconstructed, infos = compute_features(
            traces=traces, ncomps=ncomps, kind=kind, params_dict=params_dict)

        main_key = key.copy()
        main_key['features'] = features
        main_key['traces'] = traces
        main_key['traces_reconstructed'] = traces_reconstructed
        main_key['decomp_infos'] = infos
        self.insert1(main_key)

        for features_idx, roi_key in enumerate(roi_keys):
            self.RoiFeatures().insert1(dict(**roi_key, **key, features_idx=features_idx))

    class RoiFeatures(dj.Part):
        @property
        def definition(self):
            definition = """
            -> master
            -> master.roi_table
            ---
            features_idx : int
            """
            return definition

    def plot1_components(self, key=None):
        key = get_primary_key(table=self, key=key)
        decomp_infos = (self & key).fetch1('decomp_infos')

        hsize = np.max([decomp_info['components'].shape[0] for decomp_info in decomp_infos])

        fig, axs = plt.subplots(1, len(decomp_infos), sharex='all', sharey='all',
                                figsize=(4 * len(decomp_infos), hsize * 0.2))

        axs[0].set_ylabel('Components')

        fig.suptitle(f"{key!r}", y=1, va='bottom')
        for ax, decomp_info in zip(axs, decomp_infos):
            components = decomp_info['components']
            explained_variance_ratio = decomp_info['explained_variance_ratio']
            vabsmax = np.max(np.abs(components))
            ax.set_title(np.around(np.asarray(explained_variance_ratio), 2))
            ax.imshow(components, aspect='auto', cmap='coolwarm', interpolation='none', vmin=-vabsmax, vmax=vabsmax)
        plt.show()

    def plot1_traces(self, key=None):
        """Plot traces, reconstructed traces, and reconstruction errors as heatmaps"""
        key = get_primary_key(table=self, key=key)

        stim_names = (self.params_table & key).fetch1('stim_names')
        traces, traces_reconstructed = (self & key).fetch1('traces', 'traces_reconstructed')

        fig, axs = plt.subplots(len(traces), 3, sharex='all', sharey='all', figsize=(12, 6))

        for ax_row, traces_i, reconstructed_i, stim_i in zip(axs, traces, traces_reconstructed, stim_names.split('_')):
            vabsmax = np.maximum(np.max(np.abs(traces_i)), np.max(np.abs(reconstructed_i)))
            ax_row[0].set_ylabel(stim_i)
            for ax, data_i, title_i in zip(ax_row,
                                           [traces_i, reconstructed_i, reconstructed_i - traces_i],
                                           ["Traces", "Reconstruction", "Error(R-T)"]):
                ax.set_title(title_i)
                im = ax.imshow(data_i, aspect='auto', cmap='coolwarm', interpolation='none', vmin=-vabsmax,
                               vmax=vabsmax)
                plt.colorbar(im, ax=ax)

        plt.show()
