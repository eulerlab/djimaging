"""
Moving Bar feature extraction as implemented in Baden et al. 2016

Example usage:

from djimaging.tables import response

@schema
class OsDsIndexes(response.OsDsIndexesTemplate):
    stimulus_table = Stimulus
    snippets_table = Snippets
"""

import cmath
from abc import abstractmethod

import datajoint as dj
import numpy as np
from djimaging.utils.trace_utils import get_mean_dt

from djimaging.utils import math_utils
from matplotlib import pyplot as plt
from scipy import stats

from djimaging.utils.dj_utils import get_primary_key


class OsDsIndexesTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = """
        #This class computes the direction and orientation selectivity indexes 
        #as well as a quality index of DS responses as described in Baden et al. (2016)
        -> self.snippets_table
        ---
        ds_index:   float     # direction selectivity index as resulting vector length (absolute of projection on complex exponential)
        ds_pvalue:  float     # p-value indicating the percentile of the vector length in null distribution
        ds_null:    blob      # null distribution of DSIs
        pref_dir:   float     # preferred direction
        os_index:   float     # orientation selectivity index in analogy to ds_index
        os_pvalue:  float     # analogous to ds_pvalue for orientation tuning
        os_null:    blob      # null distribution of OSIs
        pref_or:    float     # preferred orientation
        on_off:     float     # on off index based on time kernel
        d_qi:       float     # quality index for moving bar response
        time_component: blob
        time_component_dt: float
        dir_component:  blob
        surrogate_v:    blob    #computed by projecting on time
        surrogate_dsi:  float   #DSI of surrogate v 
        avg_sorted_resp:    longblob    # response matrix, averaged across reps
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
        snippets, snippets_times = (self.snippets_table() & key).fetch1('snippets', 'snippets_times')

        dt = float(np.mean([get_mean_dt(snippets_time)[0] for snippets_time in snippets_times.T]))

        dsi, p_dsi, null_dist_dsi, pref_dir, osi, p_osi, null_dist_osi, pref_or, \
            on_off, d_qi, time_component, dir_component, surrogate_v, dsi_s, avg_sorted_responses = \
            compute_os_ds_idxs(snippets=snippets, dir_order=dir_order, dt=dt)

        self.insert1(dict(key,
                          ds_index=dsi, ds_pvalue=p_dsi,
                          ds_null=null_dist_dsi, pref_dir=pref_dir,
                          os_index=osi, os_pvalue=p_osi,
                          os_null=null_dist_osi, pref_or=pref_or,
                          on_off=on_off, d_qi=d_qi,
                          time_component=time_component, dir_component=dir_component,
                          surrogate_v=surrogate_v, surrogate_dsi=dsi_s,
                          avg_sorted_resp=avg_sorted_responses, time_component_dt=dt))

    def plot1(self, key=None):
        key = get_primary_key(table=self, key=key)

        dir_order = (self.stimulus_table() & key).fetch1('trial_info')
        sorted_directions_rad = np.deg2rad(np.sort(dir_order))

        dir_component, ds_index, pref_dir, avg_sorted_resp = \
            (self & key).fetch1('dir_component', 'ds_index', 'pref_dir', 'avg_sorted_resp')

        fig, axs = plt.subplots(3, 3, figsize=(6, 6), facecolor='w', subplot_kw=dict(frameon=False))
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
        ax.set_xticks([])
        ax_idxs = [0, 1, 2, 3, 5, 6, 7, 8]
        dir_idxs = [3, 2, 1, 4, 0, 5, 6, 7]
        vmin, vmax = avg_sorted_resp.min(), avg_sorted_resp.max()

        for ax_idx, dir_idx in zip(ax_idxs, dir_idxs):
            ax = axs.flat[ax_idx]
            ax.plot(avg_sorted_resp[:, dir_idx], color='k')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylim([vmin - vmax * 0.2, vmax * 1.2])
            ax.set_xlim([-len(avg_sorted_resp) * 0.2, len(avg_sorted_resp) * 1.2])
        return None

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


def quality_index_ds(raw_sorted_resp_mat):
    """
    This function computes the quality index for responses to moving bar as described in
    Baden et al. 2016. QI is computed for each direction separately and the best QI is taken
    Inputs:
    raw_sorted_resp_mat:    3d array (time x directions x reps per direction)
    Output:
    qi: float               quality index
    """

    n_dirs = raw_sorted_resp_mat.shape[1]
    qis = []
    for d in range(n_dirs):
        numerator = np.var(np.mean(raw_sorted_resp_mat[:, d, :], axis=-1), axis=0)
        denom = np.mean(np.var(raw_sorted_resp_mat[:, d, :], axis=0), axis=-1)
        qis.append(numerator / denom)
    return np.max(qis)


def sort_response_matrix(snippets: np.ndarray, idxs: list, directions: np.ndarray):
    """
    Sorts the snippets according to stimulus condition and repetition into a time x direction x repetition matrix
    Inputs:
    snippets    list or array, time x (directions*repetitions)
    idxs        list of lists giving idxs into last axis of snippets. idxs[0] gives the indexes of rows in snippets
                which are responses to the direction directions[0]
    Outputs:
    sorted_responses   array, time x direction x repetitions, with directions sorted(!) (0, 45, 90, ..., 315) degrees
    sorted_directions   array, sorted directions
    """
    structured_responses = snippets[:, idxs]
    sorting = np.argsort(directions)
    sorted_responses = structured_responses[:, sorting, :]
    sorted_directions = directions[sorting]
    return sorted_responses, sorted_directions


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

    U, S, V = np.linalg.svd(sorted_responses)

    time_component = U[:, 0]
    dir_component = V[0, :]

    # the time_kernel determined by SVD should be correlated to the average response across all directions. if the
    # correlation is negative, U is likely flipped
    r, _ = stats.spearmanr(time_component, np.mean(sorted_responses, axis=-1), axis=1)
    su = np.sign(r)
    if su == 0:
        su = 1
    sv = np.sign(np.mean(np.sign(V[0, :])))
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
    first_second_idx = np.maximum(int(np.round(1. / dt)), 1)
    time_component -= np.mean(time_component[:first_second_idx])
    time_component = time_component / np.max(abs(time_component))

    dir_component = math_utils.normalize_zero_one(dir_component)

    return time_component, dir_component


def get_si(dir_component, dirs, per):
    """
    Computes direction/orientation selectivity index and preferred direction/orientation
    of a cell by projecting the tuning curve v on a
    complex exponential of the according directions dirs (as in Baden et al. 2016)
    Inputs:
    v:  array, dirs x 1, tuning curve as returned by SVD
    dirs:   array, dirs x 1, directions in radians
    per:    int (1 or 2), indicating whether direction (1) or orientation (2) shall be tested
    Output:
    index:  float, D/O si
    direction:  float, preferred D/O
    """
    bin_spacing = np.diff(per * dirs)[0]
    correction_factor = bin_spacing / (2 * (np.sin(bin_spacing / 2)))  # Zar 1999, Equation 26.16
    compl_exp = [np.exp(per * 1j * d) for d in dirs]
    vector = np.dot(compl_exp, dir_component)
    # get the absolute of the vector, normalize to make it range between 0 and 1
    index = correction_factor * np.abs(vector) / np.sum(dir_component)

    direction = cmath.phase(vector) / per
    # for orientation, the directions are mapped to the right half of a circle. Map instead to upper half
    if per == 2 and direction < 0:
        direction += np.pi
    return index, direction


def compute_null_dist(rep_dir_resps, dirs, per, iters=1000):
    """
    Compute null distribution for direction selectivity
    """
    (rep_n, dir_n) = rep_dir_resps.shape
    flattened = np.reshape(rep_dir_resps, (rep_n * dir_n))
    rand_idx = np.linspace(0, rep_n * dir_n - 1, rep_n * dir_n, dtype=int)
    null_dist = np.zeros(iters)
    for i in range(iters):
        np.random.shuffle(rand_idx)
        shuffled = flattened[rand_idx]
        shuffled = np.reshape(shuffled, (rep_n, dir_n))
        shuffled_mean = np.mean(shuffled, axis=0)
        normalized_shuffled_mean = shuffled_mean - np.min(shuffled_mean)
        normalized_shuffled_mean /= np.max(abs(normalized_shuffled_mean))
        dsi, _ = get_si(normalized_shuffled_mean, dirs, per)
        null_dist[i] = dsi

    return null_dist


def get_on_off_index(time_kernel, dt, t_start=1.152, t_change=2.432, t_end=3.712):
    """
    Computes a preliminary On-Off Index based on the responses to the On (first half) and the OFF (2nd half) part of
    the responses to the moving bars stimulus
    """

    idx_start = int(np.round(t_start / dt))
    idx_change = int(np.round(t_change / dt))
    idx_end = int(np.round(t_end / dt))

    normed_kernel = math_utils.normalize_zero_one(time_kernel)
    deriv = np.diff(normed_kernel)
    on_response = np.max(deriv[idx_start:idx_change])
    off_response = np.max(deriv[idx_change:idx_end])
    off_response = np.max((0, off_response))
    on_response = np.max((0, on_response))

    if (on_response + off_response) < 1e-9:
        on_off = 0.
    else:
        on_off = (on_response - off_response) / (on_response + off_response)
        on_off = np.round(on_off, 2)

    return on_off


def get_dir_idx(snippets, dir_order):
    """
    snippets: np.ndarray (times, dirs*reps)
    dir_order: np.ndarray (dirs, ) or (dirs*reps, )
    """
    dir_order = np.asarray(dir_order).squeeze()
    assert dir_order.ndim == 1, dir_order.shape
    assert snippets.ndim == 2, snippets.shape
    n_snippets = snippets.shape[-1]
    assert (n_snippets % dir_order.size) == 0, f"Snippet length {n_snippets} is not a multiple of {dir_order.size}"
    dir_order = np.tile(dir_order, n_snippets // dir_order.size)
    assert n_snippets == dir_order.size

    dir_deg = dir_order[:8]  # get the directions of the bars in degree
    dir_rad = np.deg2rad(dir_deg)  # convert to radians
    dir_idx = [list(np.where(dir_order == d)[0]) for d in dir_deg]

    return dir_idx, dir_rad


def compute_mb_qi(snippets, dir_order):
    assert snippets.ndim == 2
    assert np.asarray(dir_order).ndim == 1

    dir_idx, dir_rad = get_dir_idx(snippets, dir_order)
    sorted_responses, sorted_directions = sort_response_matrix(snippets, dir_idx, dir_rad)
    d_qi = quality_index_ds(sorted_responses)
    return d_qi


def compute_os_ds_idxs(snippets: np.ndarray, dir_order: np.ndarray, dt: float):
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
    projected = np.dot(np.transpose(temp), time_component)  # we do this whole projection thing to make the result
    projected = np.reshape(projected, (d, r))  # between the original and the shuffled comparable
    surrogate_v = np.mean(projected, axis=-1)
    surrogate_v -= np.min(surrogate_v)
    surrogate_v /= np.max(surrogate_v)

    dsi_s, pref_dir_s = get_si(surrogate_v, sorted_directions, 1)
    osi_s, pref_or_s = get_si(surrogate_v, sorted_directions, 2)
    null_dist_dsi = compute_null_dist(np.transpose(projected), sorted_directions, 1)
    p_dsi = np.mean(null_dist_dsi > dsi_s)
    null_dist_osi = compute_null_dist(np.transpose(projected), sorted_directions, 2)
    p_osi = np.mean(null_dist_osi > osi_s)
    d_qi = quality_index_ds(sorted_responses)
    on_off = get_on_off_index(time_component, dt=dt)

    return dsi, p_dsi, null_dist_dsi, pref_dir, osi, p_osi, null_dist_osi, pref_or, \
        on_off, d_qi, time_component, dir_component, surrogate_v, dsi_s, avg_sorted_responses
