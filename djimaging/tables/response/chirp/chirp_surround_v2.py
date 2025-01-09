"""
Example usage:
@schema
class ChirpSurround(response.ChirpSurroundTemplate):
    _l_name = 'lChirp'
    _g_name = 'gChirp'

    _t_on_step = 2
    _t_off_step = 5
    _dt = 2

    snippets_table = Snippets
    chirp_features_table = ChirpFeatures
"""

from abc import abstractmethod

import datajoint as dj
import numpy as np

from matplotlib import pyplot as plt

from djimaging.utils.dj_utils import get_primary_key, get_secondary_keys
from djimaging.utils.trace_utils import get_mean_dt


class ChirpSurroundTemplate(dj.Computed):
    database = ""

    _l_name = 'lChirp'
    _g_name = 'gChirp'

    _t_on_step = 2
    _t_off_step = 5
    _dt = 2
    _t_plot_max = 8

    @property
    def definition(self):
        definition = f"""
        -> self.chirp_features_table.proj({self._l_name}='stim_name')
        -> self.snippets_table.proj({self._l_name}='stim_name')
        -> self.snippets_table.proj({self._g_name}='stim_name')
        ---
        chirp_surround_index = NULL : float
        l_response_mus : blob  # (repetitions,) local response means
        g_response_mus : blob  # (repetitions,) global response means 
        """
        return definition

    @property
    @abstractmethod
    def snippets_table(self):
        pass

    @property
    @abstractmethod
    def chirp_features_table(self):
        pass

    @property
    def key_source(self):
        try:
            return (
                    (self.chirp_features_table & dict(stim_name=f"{self._l_name}")).proj(
                        **{self._l_name: 'stim_name'}) *
                    (self.snippets_table & dict(stim_name=f"{self._l_name}")).proj(**{self._l_name: 'stim_name'}) *
                    (self.snippets_table & dict(stim_name=f"{self._g_name}")).proj(**{self._g_name: 'stim_name'})
            )
        except (AttributeError, TypeError):
            pass

    def compute_entry(self, key, plot=False):
        l_key = {**key, 'stim_name': self._l_name}
        g_key = {**key, 'stim_name': self._g_name}

        polarity_index = (self.chirp_features_table & l_key).fetch1('polarity_index')

        l_snippets_t0, l_snippets_dt, l_snippets, l_triggertimes_snippets = (self.snippets_table() & l_key).fetch1(
            "snippets_t0", "snippets_dt", 'snippets', 'triggertimes_snippets')
        g_snippets_t0, g_snippets_dt, g_snippets, g_triggertimes_snippets = (self.snippets_table() & g_key).fetch1(
            "snippets_t0", "snippets_dt", 'snippets', 'triggertimes_snippets')

        l_snippets_times = (np.tile(np.arange(l_snippets.shape[0]) * l_snippets_dt, (len(l_snippets_t0), 1)).T
                            + l_snippets_t0)
        g_snippets_times = (np.tile(np.arange(g_snippets.shape[0]) * g_snippets_dt, (len(g_snippets_t0), 1)).T
                            + g_snippets_t0)

        dt = get_mean_dt(l_snippets_times.T)[0]
        dt_g = get_mean_dt(g_snippets_times.T)[0]
        if not np.isclose(dt_g, dt):
            raise ValueError("Sampling rate of the two traces do not match")
        fs = 1. / dt

        l_n_reps = l_snippets.shape[1]
        l_response_mus = np.full((l_n_reps,), np.nan)

        g_n_reps = g_snippets.shape[1]
        g_response_mus = np.full((g_n_reps,), np.nan)

        if polarity_index >= 0:
            n_idxs_base = int(np.round(self._t_on_step * fs))
            n_idxs_spot = int(np.round((self._t_on_step + self._dt) * fs))
        else:
            n_idxs_base = int(np.round(self._t_off_step * fs))
            n_idxs_spot = int(np.round((self._t_off_step + self._dt) * fs))

        n_reps = np.maximum(l_n_reps, g_n_reps)

        if plot:
            fig, axs = plt.subplots(n_reps, 2, figsize=(8, 5), squeeze=False)

            fig.suptitle(f"pol_index = {polarity_index:.1f}")

            axs[0, 0].set(title=self._l_name)
            axs[0, 1].set(title=self._g_name)

        for rep_i in range(n_reps):
            if plot:
                ax = axs[rep_i, 0]
                ax.plot(l_snippets_times[:, rep_i], l_snippets[:, rep_i], c='k', zorder=-10)
                ax.set_xlim(l_snippets_times[0, rep_i] - 0.1, l_snippets_times[0, rep_i] + self._t_plot_max)

                ax = axs[rep_i, 1]
                ax.plot(g_snippets_times[:, rep_i], g_snippets[:, rep_i], c='k', zorder=-10)
                ax.set_xlim(g_snippets_times[0, rep_i] - 0.1, g_snippets_times[0, rep_i] + self._t_plot_max)

            idxs_base_j = np.arange(n_idxs_base)
            idxs_r_j = np.arange(n_idxs_base, n_idxs_spot)

            if rep_i < l_n_reps:
                l_base = np.median(l_snippets[idxs_base_j, rep_i])
                l_r = np.mean(l_snippets[idxs_r_j, rep_i] - l_base)
                l_response_mus[rep_i] = l_r

            if rep_i < g_n_reps:
                g_base = np.median(g_snippets[idxs_base_j, rep_i])
                g_r = np.mean(g_snippets[idxs_r_j, rep_i] - g_base)
                g_response_mus[rep_i] = g_r

            if plot:
                if rep_i < l_n_reps:
                    ax = axs[rep_i, 0]
                    ax.plot(l_snippets_times[idxs_base_j, rep_i], l_snippets[idxs_base_j, rep_i],
                            color='C0', alpha=0.5, ls='--')
                    ax.fill_between(l_snippets_times[idxs_r_j, rep_i],
                                    np.ones(idxs_r_j.size) * l_base,
                                    l_snippets[idxs_r_j, rep_i], color='C0')

                    for tt in l_triggertimes_snippets[:, rep_i]:
                        ax.axvline(tt, c='r')

                if rep_i < g_n_reps:
                    ax = axs[rep_i, 1]
                    ax.plot(g_snippets_times[idxs_base_j, rep_i], g_snippets[idxs_base_j, rep_i],
                            color='C1', alpha=0.5, ls='--')
                    ax.fill_between(g_snippets_times[idxs_r_j, rep_i],
                                    np.ones(idxs_r_j.size) * g_base,
                                    g_snippets[idxs_r_j, rep_i], color='C1')

                    for tt in g_triggertimes_snippets[:, rep_i]:
                        ax.axvline(tt, c='r')

        l_mu = np.median(l_response_mus)
        g_mu = np.median(g_response_mus)
        surround_index = g_mu - np.clip(l_mu, 0, None)

        if plot:
            plt.tight_layout(rect=(0, 0, 0.7, 1))
            ax = fig.add_axes(rect=(0.8, 0.8, 0.15, 0.15))

            ax.scatter(np.ones(l_n_reps) * 1, l_response_mus, color='C0', alpha=0.5, clip_on=False)
            ax.scatter(np.ones(g_n_reps) * 2, g_response_mus, color='C1', alpha=0.5, clip_on=False)

            ax.axhline(l_mu, c='C0', ls='--')
            ax.axhline(g_mu, c='C1', ls='--')

            ax = fig.add_axes(rect=(0.8, 0.45, 0.15, 0.15))
            ax.scatter(np.arange(g_n_reps), g_response_mus - l_mu, color='dimgray')
            ax.axhline(surround_index, c='k', ls='--')
            ax.set(xlabel='reps', ylabel='r_g - r_l', title=f"surround index = {surround_index:.2f}")

            plt.show()

        return surround_index, l_response_mus, g_response_mus

    def make(self, key, plot=False):
        surround_index, l_response_mus, g_response_mus = self.compute_entry(key, plot=plot)

        self.insert1(dict(
            key,
            chirp_surround_index=surround_index,
            l_response_mus=l_response_mus,
            g_response_mus=g_response_mus,
        ))

    def plot1(self, key=None):
        key = get_primary_key(table=self, key=key)
        self.compute_entry(key, plot=True)
