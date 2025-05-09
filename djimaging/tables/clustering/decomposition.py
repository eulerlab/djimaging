"""
Features extraction (e.g. PCA, sparse PCA, etc.) of traces.
Mostly used for clustering.
"""

from abc import abstractmethod
from copy import deepcopy

import datajoint as dj
import numpy as np
from matplotlib import pyplot as plt

from djimaging.utils.dj_utils import get_primary_key
from djimaging.utils.math_utils import truncated_vstack
from djimaging.utils.trace_utils import argsort_traces


class FeaturesParamsTemplate(dj.Lookup):
    database = ""

    @property
    def definition(self):
        definition = """
        features_id: tinyint unsigned # unique param set id
        ---
        kind: varchar(191)
        params_dict: longblob
        norm_trace: tinyint unsigned  # Used normalized averages or averages?
        stim_names: varchar(191)  # Stimuli to consider, separated by '_'
        ncomps: varchar(191)  # Number of components separated by '_'
        pre_standardize: tinyint unsigned  # Standardize features before decomposition?
        post_standardize: tinyint unsigned  # Standardize features after decomposition?
        """
        return definition

    def add(self, features_id=1, kind='sparse_pca', params_dict=None, norm_trace: bool = False,
            stim_names='gChirp_lChirp', ncomps='20_20', pre_standardize=False, post_standardize=True,
            skip_duplicates=False):
        if params_dict is None:
            params_dict = dict()
        assert isinstance(params_dict, dict)
        key = dict(features_id=features_id, kind=kind, params_dict=params_dict, norm_trace=int(norm_trace),
                   stim_names=stim_names, ncomps=ncomps,
                   pre_standardize=int(pre_standardize), post_standardize=int(post_standardize))
        self.insert1(key, skip_duplicates=skip_duplicates)


class FeaturesTemplate(dj.Computed):
    database = ""

    _restr_filter = None  # Restriction applied to roi_quality_table if any.
    roi_filter_table = None

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
    def key_source(self):
        try:
            return self.params_table.proj()
        except (AttributeError, TypeError):
            pass

    def fetch_traces(self, key, rtol=0.15):
        norm_trace, stim_names = (self.params_table & key).fetch1('norm_trace', 'stim_names')
        average_key = 'average_norm' if norm_trace else 'average'
        stim_names = stim_names.split('_')

        tab = self.roi_table.proj()

        restr = self._restr_filter if self._restr_filter is not None else dict()

        if self.roi_filter_table is not None:
            tab = tab & self.roi_filter_table & restr

        for stim_i in stim_names:
            tab = tab * (self.averages_table & f"stim_name='{stim_i}'" & restr).proj(
                **{f'{stim_i}_avgs': average_key, f'{stim_i}_dt': 'average_dt',
                   f'{stim_i}_t0': 'average_t0', f'{stim_i}_name': 'stim_name'})

        times = [
            truncated_vstack(
                np.arange(tab.fetch(f'{stim_i}_avgs').shape[0]) * tab.fetch1(f'{stim_i}_dt')
                + tab.fetch1(f'{stim_i}_t0'), rtol=rtol)
            for stim_i in stim_names]
        traces = [
            truncated_vstack(tab.fetch(f'{stim_i}_avgs'), rtol=rtol)
            for stim_i in stim_names]
        roi_keys = tab.fetch(*self.roi_table.primary_key, as_dict=True)
        return traces, times, roi_keys, stim_names

    def make(self, key, verboselvl=1):
        kind, params_dict, ncomps, pre_standardize, post_standardize = (self.params_table & key).fetch1(
            'kind', 'params_dict', 'ncomps', 'pre_standardize', 'post_standardize')

        ncomps = [int(ncomps_i) for ncomps_i in ncomps.split('_')] if len(ncomps) > 0 else []
        traces, times, roi_keys, stim_names = self.fetch_traces(key=key)

        features, traces_reconstructed, infos = compute_features(
            traces=traces, ncomps=ncomps, kind=kind, params_dict=params_dict,
            pre_standardize=pre_standardize, post_standardize=post_standardize,
            verboselvl=verboselvl)

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

    def plot1_components(self, key=None, sort=False):
        key = get_primary_key(table=self, key=key)
        decomp_infos = (self & key).fetch1('decomp_infos')

        if 'components' not in decomp_infos[0].keys():
            print(f'No component available for key={key}')
            return

        hsize = np.max([decomp_info['components'].shape[0] for decomp_info in decomp_infos])

        fig, axs = plt.subplots(1, len(decomp_infos), sharex='all',
                                figsize=(4 * len(decomp_infos), hsize * 0.2), squeeze=False)
        axs = axs.flat
        axs[0].set_ylabel('Sorted components' if sort else 'Components')

        fig.suptitle(f"{key!r}", y=1, va='bottom')
        for ax, decomp_info in zip(axs, decomp_infos):
            components = decomp_info['components']
            if sort:
                sort_idxs = np.argsort(np.argmax(np.cumsum(np.abs(components), axis=1) > 0.5, axis=1))
            else:
                sort_idxs = np.arange(components.shape[0])

            vabsmax = np.max(np.abs(components))
            ax.set_title(f"var_exp_tot={np.sum(decomp_info['explained_variance_ratio']):.1%}")
            ax.imshow(components[sort_idxs, :],
                      aspect='auto', cmap='bwr', interpolation='none', vmin=-vabsmax, vmax=vabsmax)
        plt.show()

    def plot1_traces(self, key=None):
        """Plot traces, reconstructed traces, and reconstruction errors as heatmaps"""
        key = get_primary_key(table=self, key=key)

        stim_names = (self.params_table & key).fetch1('stim_names')
        traces, traces_reconstructed = (self & key).fetch1('traces', 'traces_reconstructed')

        sort_idxs = argsort_traces(traces[0])

        fig, axs = plt.subplots(len(traces), 3, sharex='all', sharey='all', figsize=(12, 6), squeeze=False)

        for ax_row, traces_i, reconstructed_i, stim_i in zip(axs, traces, traces_reconstructed, stim_names.split('_')):
            errors_i = reconstructed_i - traces_i
            vabsmax = np.max([np.max(np.abs(traces_i)),
                              np.max(np.abs(reconstructed_i)),
                              np.max(np.abs(errors_i))])
            ax_row[0].set_ylabel(stim_i)

            for ax, data_i, title_i in zip(ax_row,
                                           [traces_i, reconstructed_i, errors_i],
                                           ["Traces", "Reconstruction", "Error(R-T)"]):
                ax.set_title(title_i)
                im = ax.imshow(data_i[sort_idxs, :], origin='lower',
                               aspect='auto', cmap='bwr', interpolation='none', vmin=-vabsmax, vmax=vabsmax)
                plt.colorbar(im, ax=ax)

        plt.show()


def compute_features(traces: list, ncomps: list, kind: str, params_dict: dict,
                     pre_standardize=False, post_standardize=False, verboselvl=0) -> (list, list, list):
    """Reduce dimension of traces to features and stack them to feature matrix"""
    kind = kind.lower()

    if verboselvl:
        print(f"Computing features with kind={kind} and params_dict={params_dict}")

    if kind == 'sparse_pca':
        features, traces_reconstructed, infos = compute_features_sparse_pca(
            traces=traces, ncomps=ncomps, standardize=pre_standardize, verboselvl=verboselvl, **params_dict)
    elif kind == 'pca':
        assert len(params_dict) == 0
        features, traces_reconstructed, infos = compute_features_pca(
            traces=traces, ncomps=ncomps, standardize=pre_standardize)
    elif kind == 'none':
        features, traces_reconstructed, infos = deepcopy(traces), deepcopy(traces), [dict()] * len(traces)
    else:
        raise NotImplementedError(kind)

    if post_standardize:
        if verboselvl:
            print("Standardizing features after decomposition.")

        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler(with_mean=True, with_std=True)
        features = [sc.fit_transform(X=features_i) for features_i in features]

    return features, traces_reconstructed, infos


def compute_variance_explained_sparse_pca(X: np.ndarray, P: np.ndarray) -> (float, float):
    """Compute the variance explained variance of sparse PCA
    Code based on: https://github.com/scikit-learn/scikit-learn/issues/11512#issuecomment-1354299118
    Original Author: https://github.com/qbilius
    scikit-learn license: BSD 3-Clause (https://opensource.org/licenses/BSD-3-Clause)
    Idea based on: Camacho et al. (2019): explained here"""

    Xc = X - np.mean(X, axis=0)  # center data
    T = Xc @ P @ np.linalg.pinv(P.T @ P)

    explained_variance = np.trace(P @ T.T @ T @ P.T)
    total_variance = np.trace(Xc.T @ Xc)

    return explained_variance, total_variance


def compute_features_sparse_pca(
        traces: list, ncomps: list, alpha=1, standardize=False, verboselvl=1) -> (list, list, list):
    from sklearn.decomposition import SparsePCA

    features, traces_reconstructed, infos = [], [], []
    for traces_i, ncomps_i in zip(traces, ncomps):
        if standardize:
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler(with_mean=True, with_std=True)
            norm_traces_i = sc.fit_transform(X=traces_i)
        else:
            sc = None
            norm_traces_i = traces_i

        decomp = SparsePCA(n_components=ncomps_i, random_state=0, alpha=alpha, verbose=verboselvl)
        features_i = decomp.fit_transform(X=norm_traces_i)

        try:
            if standardize:
                traces_reconstructed_i = sc.inverse_transform(X=decomp.inverse_transform(X=features_i))
            else:
                traces_reconstructed_i = decomp.inverse_transform(X=features_i)
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


def compute_features_pca(traces: list, ncomps: list, standardize=False) -> (list, list, list):
    from sklearn.decomposition import PCA

    features, traces_reconstructed, infos = [], [], []
    for traces_i, ncomps_i in zip(traces, ncomps):
        if standardize:
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler(with_mean=True, with_std=True)
            norm_traces_i = sc.fit_transform(X=traces_i)
        else:
            sc = None
            norm_traces_i = traces_i

        decomp = PCA(n_components=ncomps_i, random_state=0)
        features_i = decomp.fit_transform(X=norm_traces_i)

        if standardize:
            traces_reconstructed_i = sc.inverse_transform(X=decomp.inverse_transform(X=features_i))
        else:
            traces_reconstructed_i = decomp.inverse_transform(X=features_i)

        info_i = dict()
        info_i["components"] = decomp.components_
        info_i["explained_variance"] = decomp.explained_variance_
        info_i["explained_variance_ratio"] = decomp.explained_variance_ratio_

        features.append(features_i)
        traces_reconstructed.append(traces_reconstructed_i)
        infos.append(info_i)

    return features, traces_reconstructed, infos
