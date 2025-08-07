"""
Moving Bar feature extraction similar but not the same as in Baden et al. 2016

Example usage:

from djimaging.tables import response

@schema
class OsDsIndexes(response.OsDsIndexesTemplate):
    _reduced_storage = True
    _n_shuffles = 100

    stimulus_table = Stimulus
    snippets_table = Snippets
"""
import warnings
from abc import abstractmethod

import datajoint as dj
import numpy as np
from matplotlib import pyplot as plt

from djimaging.tables.response.movingbar.orientation_utils import preprocess_mb_snippets, T_START, T_CHANGE, T_END
from djimaging.tables.response.movingbar.orientation_utils_v1 import compute_os_ds_idxs as compute_os_ds_idxs_v1
from djimaging.tables.response.movingbar.orientation_utils_v2 import compute_os_ds_idxs as compute_os_ds_idxs_v2
from djimaging.utils.dj_utils import get_primary_key


class OsDsIndexesTemplate(dj.Computed):
    database = ""
    _reduced_storage = True  # Don't save all intermediate results
    _n_shuffles = 100  # Number of shuffles for null distribution
    _version = 1  # or 2

    @property
    def definition(self):
        definition = """
        #This class computes the direction and orientation selectivity indexes 
        #as well as a quality index of DS responses as described in Baden et al. (2016)
        -> self.snippets_table
        ---
        ds_index:   float     # direction selectivity index as resulting vector length (absolute of projection on complex exponential)
        ds_pvalue:  float     # p-value indicating the percentile of the vector length in null distribution
        pref_dir:   float     # preferred direction
        os_index:   float     # orientation selectivity index in analogy to ds_index
        os_pvalue:  float     # analogous to ds_pvalue for orientation tuning
        pref_or:    float     # preferred orientation
        on_off:     float     # on off index based on time kernel
        d_qi:       float     # quality index for moving bar response
        dir_component:     blob
        time_component:    blob
        time_component_dt: float
        surrogate_v:       blob    # computed by projecting on time
        surrogate_dsi:     float   # DSI of surrogate v 
        """

        if not self._reduced_storage:
            definition += """
        ds_null:    blob      # null distribution of DSIs
        os_null:    blob      # null distribution of OSIs
        avg_sorted_resp: longblob
        """

        return definition

    @property
    @abstractmethod
    def stimulus_table(self):
        pass

    @property
    @abstractmethod
    def snippets_table(self):
        pass

    @property
    def key_source(self):
        try:
            return self.snippets_table().proj() & \
                (self.stimulus_table() & "stim_name = 'movingbar' or stim_family = 'movingbar'")
        except (AttributeError, TypeError):
            pass

    def make(self, key):
        dir_order = (self.stimulus_table() & key).fetch1('trial_info')
        snippets_dt, snippets = (self.snippets_table() & key).fetch1('snippets_dt', 'snippets')

        # Pick version
        if self._version == 1:
            compute_os_ds_idxs = compute_os_ds_idxs_v1
        elif self._version == 2:
            compute_os_ds_idxs = compute_os_ds_idxs_v2
        else:
            raise ValueError(f"Version {self._version} not supported.")

        dsi, p_dsi, null_dist_dsi, pref_dir, osi, p_osi, null_dist_osi, pref_or, \
            on_off, d_qi, time_component, dir_component, surrogate_v, dsi_s, avg_sorted_responses = \
            compute_os_ds_idxs(snippets=snippets, dir_order=dir_order, dt=snippets_dt, n_shuffles=self._n_shuffles)

        entry = dict(
            key,
            ds_index=dsi, ds_pvalue=p_dsi, pref_dir=pref_dir,
            os_index=osi, os_pvalue=p_osi, pref_or=pref_or,
            on_off=on_off, d_qi=d_qi,
            time_component=time_component.astype(np.float32), time_component_dt=snippets_dt,
            dir_component=dir_component.astype(np.float32),
            surrogate_v=surrogate_v.astype(np.float32), surrogate_dsi=dsi_s,
        )

        if not self._reduced_storage:
            entry['ds_null'] = null_dist_dsi.astype(np.float32)
            entry['os_null'] = null_dist_osi.astype(np.float32)
            entry['avg_sorted_resp'] = avg_sorted_responses.astype(np.float32)

        self.insert1(entry)

    def plot1(self, key=None):
        key = get_primary_key(table=self, key=key)

        dir_order = (self.stimulus_table() & key).fetch1('trial_info')
        sorted_directions_rad = np.deg2rad(np.sort(dir_order))

        (time_component_dt, dir_component, ds_index, ds_pvalue, os_index, os_pvalue, pref_dir, pref_or, on_off) = (
                self & key).fetch1(
            'time_component_dt', 'dir_component', 'ds_index', 'ds_pvalue', 'os_index', 'os_pvalue',
            'pref_dir', 'pref_or', 'on_off')

        fig, axs = plt.subplots(3, 3, figsize=(6, 6), facecolor='w', sharex=True, sharey=True)

        fig.suptitle(
            f"DSI: {ds_index:.2f}, Pref-Dir: {(360 + np.rad2deg(pref_dir)) % 360:.0f}°; p={ds_pvalue:.2f}\n"
            f"OSI: {ds_index:.2f}, Pref-Or: {(180 + np.rad2deg(pref_or)) % 180:.0f}°; p={os_pvalue:.2f}\n"
            f"On-Off: {on_off:.2f}")

        # Polar plot in center
        axs[1, 1].remove()
        ax = fig.add_subplot(3, 3, 5, projection='polar', frameon=False)
        temp = np.max(np.append(dir_component, ds_index))
        ax.plot((0, np.pi), (temp * 1.2, temp * 1.2), color='gray')
        ax.plot((np.pi / 2, np.pi / 2 * 3), (temp * 1.2, temp * 1.2), color='gray')
        ax.plot([0, pref_dir], [0, ds_index * np.sum(dir_component)], color='r')
        ax.plot(np.append(sorted_directions_rad, sorted_directions_rad[0]),
                np.append(dir_component, dir_component[0]), color='k')
        ax.set_rmin(0)
        ax.set_thetalim([0, 2 * np.pi])
        ax.set_yticks([])
        ax_idxs = [0, 1, 2, 3, 5, 6, 7, 8]
        dir_idxs = [3, 2, 1, 4, 0, 5, 6, 7]

        if not self._reduced_storage:
            avg_sorted_resp = (self & key).fetch1('avg_sorted_resp')
        else:
            snippets = (self.snippets_table() & key).fetch1('snippets')
            sorted_directions, sorted_responses, avg_sorted_resp = preprocess_mb_snippets(snippets, dir_order)

        for idx, (ax_idx, dir_idx) in enumerate(zip(ax_idxs, dir_idxs)):
            ax = axs.flat[ax_idx]
            ax.fill_between(np.arange(avg_sorted_resp.shape[0]) * time_component_dt,
                            avg_sorted_resp[:, dir_idx], color='red', alpha=0.5)
            ax.axvline(x=T_START, color='gray', linestyle='--')
            ax.axvline(x=T_CHANGE, color='gray', linestyle='--')
            ax.axvline(x=T_END, color='gray', linestyle='--')

            ax.spines['left'].set_visible(True)
            # Remove all other spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        plt.tight_layout()

    def plot(self, restriction=None):
        if restriction is None:
            restriction = dict()

        var_names = ['ds_index', 'ds_pvalue', 'os_index', 'os_pvalue', 'on_off', 'd_qi']
        fig, axs = plt.subplots(1, len(var_names), figsize=(len(var_names) * 2, 2), squeeze=False)
        axs = axs.flatten()
        for ax, var_name in zip(axs, var_names):
            dat = (self & restriction).fetch(var_name)
            ax.hist(dat)
            ax.set(title=var_name)
        plt.tight_layout()
        plt.show()
        return fig, axs


class OsDsIndexesTemplateV1(OsDsIndexesTemplate):
    _version = 1

    def __init__(self, *args, **kwargs):
        warnings.warn("OsDsIndexesTemplateV1 is deprecated. Use OsDsIndexesTemplate with _version=1 instead.",
                      DeprecationWarning, stacklevel=2)
        super().__init__(*args, **kwargs)

    @property
    @abstractmethod
    def stimulus_table(self):
        pass

    @property
    @abstractmethod
    def snippets_table(self):
        pass


class OsDsIndexesTemplateV2(OsDsIndexesTemplate):
    _version = 2

    def __init__(self, *args, **kwargs):
        warnings.warn("OsDsIndexesTemplateV2 is deprecated. Use OsDsIndexesTemplate with _version=2 instead.",
                      DeprecationWarning, stacklevel=2)
        super().__init__(*args, **kwargs)

    @property
    @abstractmethod
    def stimulus_table(self):
        pass

    @property
    @abstractmethod
    def snippets_table(self):
        pass
