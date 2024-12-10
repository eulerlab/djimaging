"""
Moving Bar feature extraction implemented as in Baden et al. 2016

Example usage:

from djimaging.tables import response

@schema
class OsDsIndexes(response.OsDsIndexesTemplate):
    _reduced_storage = True
    _n_shuffles = 100

    stimulus_table = Stimulus
    snippets_table = Snippets
"""

from abc import abstractmethod

import datajoint as dj
import numpy as np
from matplotlib import pyplot as plt

from djimaging.tables.response.orientation_v1 import get_si, quality_index_ds, get_on_off_index, get_dir_idx, \
    sort_response_matrix
from djimaging.utils.dj_utils import get_primary_key

T_START = 1.152
T_CHANGE = 2.432
T_END = 3.712


class OsDsIndexesTemplate(dj.Computed):
    database = ""
    _reduced_storage = True  # Don't save all intermediate results
    _n_shuffles = 100  # Number of shuffles for null distribution

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
        snippets_t0, snippets_dt, snippets = (self.snippets_table() & key).fetch1(
            'snippets_t0', 'snippets_dt', 'snippets')

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

        time_component_dt, dir_component, ds_index, pref_dir = (self & key).fetch1(
            'time_component_dt', 'dir_component', 'ds_index', 'pref_dir')

        fig, axs = plt.subplots(3, 3, figsize=(6, 6), facecolor='w', sharex=True, sharey=True)

        fig.suptitle(f"DSI: {ds_index:.2f}, Pref-Dir: {(360 + np.rad2deg(pref_dir)) % 360:.0f}")

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
            dir_idx, dir_rad = get_dir_idx(snippets, dir_order)

            sorted_responses, sorted_directions = sort_response_matrix(snippets, dir_idx, dir_rad)
            avg_sorted_resp = np.mean(sorted_responses, axis=-1)

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


def get_time_dir_kernels(sorted_responses, dt):
    """
    Performs singular value decomposition on the time x direction matrix (averaged across repetitions)
    Uses a heuristic to try to determine whether a sign flip occurred during svd
    For the time course, the mean of the first second is subtracted and then the vector is divided by the maximum
    absolute value.
    For the direction/orientation tuning curve, the vector is normalized to the range (0,1)
    Input:
    sorted_responses:   array, time x direction
    dt: 1 / sampling_rate of trace
    Outputs:
    time_kernel     array, time x 1 (time component, 1st component of U)
    direction_tuning    array, directions x 1 (direction tuning, 1st component of V)
    singular_value  float, 1st singular value
    """

    U, S, Vh = np.linalg.svd(sorted_responses)

    time_component = U[:, 0]
    dir_component = Vh[0, :]

    # the time_kernel determined by SVD should be correlated to the average response across all directions. if the
    # correlation is negative, U is likely flipped

    if np.mean((-1 * time_component - np.mean(sorted_responses, axis=-1)) ** 2) < np.mean(
            (time_component - np.mean(sorted_responses, axis=-1)) ** 2
    ):
        su = -1
    else:
        su = 1

    sv = np.sign(np.mean(np.sign(dir_component)))
    if sv == 1 and su == 1:
        s = 1
    elif sv == -1 and su == -1:
        s = -1
    elif sv == 1 and su == -1:
        s = 1
    elif sv == 0:
        s = su
    else:
        s = 1

    time_component *= s
    dir_component *= s

    # determine which entries correspond to the first second, assuming 4 seconds presentation time
    first_second_idx = np.maximum(int(np.floor(1.0 / dt)), 1)
    time_component -= np.mean(time_component[:first_second_idx])
    time_component = time_component / np.max(np.abs(time_component))

    dir_component = dir_component - np.min(dir_component)
    dir_component = dir_component / np.max(dir_component)

    return time_component, dir_component


def compute_null_dist(dirs, counts, per, iters=1000):
    """
    Test significance of orientation tuning by permutation test.

    Parameters:
        dirs (ndarray): Vector of directions (#directions x 1) in radians.
        counts (ndarray): Matrix of responses (#reps x #directions).
        per (int): Fourier component to test (1 = direction, 2 = orientation).
        iters (int): Number of permutations for the test.

    Returns:
        p (float): p-value for tuning.
        q (float): Magnitude of the Fourier component.
        qdistr (ndarray): Sampling distribution of |q| under the null hypothesis.
    """
    rep_n, dir_n = counts.shape
    k = dirs.reshape(-1)
    v = np.exp(per * 1j * k) / np.sqrt(dir_n)

    # Compute magnitude of Fourier component for original data
    q = np.abs(np.mean(counts, axis=0) @ v)

    # Initialize null distribution
    qdistr = np.zeros(iters)

    # Flatten counts for permutation
    flattened_counts = counts.flatten()

    for i in range(iters):
        # Shuffle counts
        shuffled_indices = np.random.permutation(rep_n * dir_n)
        shuffled_counts = flattened_counts[shuffled_indices]
        shuffled_counts = shuffled_counts.reshape(rep_n, dir_n)

        # Compute Fourier magnitude for shuffled data
        qdistr[i] = np.abs(np.mean(shuffled_counts, axis=0) @ v)

    # Compute p-value
    p = np.mean(qdistr > q)

    return p, q, qdistr


def compute_os_ds_idxs(snippets: np.ndarray, dir_order: np.ndarray, dt: float, n_shuffles: int = 100):
    assert snippets.ndim == 2
    assert np.asarray(dir_order).ndim == 1

    dir_idx, dir_rad = get_dir_idx(snippets, dir_order)

    sorted_responses, sorted_directions = sort_response_matrix(snippets, dir_idx, dir_rad)
    avg_sorted_responses = np.mean(sorted_responses, axis=-1)

    time_component, dir_component = get_time_dir_kernels(avg_sorted_responses, dt=dt)

    dsi, pref_dir = get_si(dir_component, sorted_directions, 1)
    osi, pref_or = get_si(dir_component, sorted_directions, 2)
    (t, d, r) = sorted_responses.shape
    temp = np.reshape(sorted_responses, (t, d * r))
    projected_flat = temp.T @ time_component  # we do this whole projection thing to make the result
    projected = np.reshape(projected_flat, (d, r))  # between the original and the shuffled comparable
    surrogate_v = np.mean(projected, axis=-1)
    surrogate_v -= np.min(surrogate_v)
    surrogate_v /= np.max(surrogate_v)

    dsi_s, pref_dir_s = get_si(surrogate_v, sorted_directions, 1)
    # osi_s, pref_or_s = get_si(surrogate_v, sorted_directions, 2)  # Not used atm

    p_dsi, null_dist_dsi, _ = compute_null_dist(sorted_directions, projected.T, 1, iters=n_shuffles)
    p_osi, null_dist_osi, _ = compute_null_dist(sorted_directions, projected.T, 2, iters=n_shuffles)

    d_qi = quality_index_ds(sorted_responses)
    on_off = get_on_off_index(time_component, dt=dt)

    return (
        dsi,
        p_dsi,
        null_dist_dsi,
        pref_dir,
        osi,
        p_osi,
        null_dist_osi,
        pref_or,
        on_off,
        d_qi,
        time_component,
        dir_component,
        surrogate_v,
        dsi_s,
        avg_sorted_responses,
    )
