import datajoint as dj
import numpy as np
from scipy import stats
import cmath
import matplotlib.pyplot as plt

from djimaging.utils.dj_utils import PlaceholderTable

class OsDsIndexesV22Template(dj.Computed):
    database = ""  # hack to suppress DJ error

    @property
    def definition(self):
        definition = """
        #This class computes the direction and orientation selectivity indexes
        #as well as a quality index of DS responses as described in Baden et al. (2016)
        -> self.snippets_table
        ---
        ds_index:   float   #direction selectivity index as resulting vector length (absolute of projection on complex exponential)
        ds_pvalue:  float   #p-value indicating the percentile of the vector length in null distribution
        ds_null:    longblob    #null distribution of DSIs
        pref_dir:  float    #preferred direction
        os_index:   float   #orientation selectivity index in analogy to ds_index
        os_pvalue:  float   #analogous to ds_pvalue for orientation tuning
        os_null:    longblob    #null distribution of OSIs
        pref_or:    float   #preferred orientation
        on_off:     float   #on off index based on time kernel
        d_qi:       float   #quality index for moving bar response
        u:     longblob    #time component
        v:     longblob    #direction component
        surrogate_v:    longblob    #computed by projecting on time
        surrogate_dsi:  float   #DSI of surrogate v
        avg_sorted_resp:    longblob    # response matrix, averaged across reps
        """
        return definition

    stimulus_table = PlaceholderTable
    snippets_table = PlaceholderTable

    @property
    def key_source(self):
        return self.snippets_table() & (self.stimulus_table() & "stim_name = 'movingbar' or stim_family = 'movingbar'")

    def make(self, key):
        #if (Stimulus() & key).fetch1('stimulusname') == "ds":
        dir_order = (self.stimulus_table() & key).fetch1('trial_info')
        snippets = (self.snippets_table() & key).fetch1('snippets')  # get the response snippets

        dir_order = np.tile(dir_order, 3)
        #dir_order = Stimulus().DsInfo().fetch1('trialinfo')
        dir_deg = list(dir_order[:8])   # get the directions of the bars in degree
        dir_rad = np.deg2rad(dir_deg)   # convert to radians
        #snippets = (DetrendSnippets() & key).fetch1('detrend_snippets')  # get the response snippets

        valid = False   # check whether there are 2 or 3 complete repetitions of the stimulus sequence
        if snippets.shape[-1] == len(dir_order):
            dir_idx = [list(np.nonzero(dir_order == d)[0])
                        for d in dir_deg]
            valid = True
        elif snippets.shape[-1] == 16:
            dir_idx = [list(np.nonzero(dir_order[:-8] == d)[0])
                        for d in dir_deg]
            valid = True
        if valid:
            sorted_responses, sorted_directions = \
                sort_response_matrix(snippets, dir_idx, dir_rad)
            avg_sorted_responses = np.mean(sorted_responses, axis=-1)
            u, v, s = get_time_dir_kernels(avg_sorted_responses)
            dsi, pref_dir = get_si(v, sorted_directions, 1)
            osi, pref_or = get_si(v, sorted_directions, 2)
            (t, d, r) = sorted_responses.shape
            temp = np.reshape(sorted_responses, (t, d*r))
            projected = np.dot(np.transpose(temp), u)  # we do this whole projection thing to make the result
            projected = np.reshape(projected, (d, r))  #  between the original and the shuffled comparable
            surrogate_v = np.mean(projected, axis = -1)
            surrogate_v -= np.min(surrogate_v)
            surrogate_v /= np.max(surrogate_v)
            dsi_s, pref_dir_s = get_si(surrogate_v,
                                            sorted_directions,
                                            1)
            osi_s, pref_or_s = get_si(surrogate_v,
                                            sorted_directions,
                                            2)
            null_dist_dsi = test_tuning(np.transpose(projected),
                                                sorted_directions,
                                                1)
            p_dsi = np.mean(null_dist_dsi > dsi_s)
            null_dist_osi = test_tuning(np.transpose(projected),
                                                sorted_directions,
                                                2)
            p_osi = np.mean(null_dist_osi > osi_s)
            d_qi = quality_index_ds(sorted_responses)
            on_off = get_on_off_index(u)
            self.insert1(dict(key,
                                ds_index=dsi, ds_pvalue=p_dsi,
                                ds_null=null_dist_dsi, pref_dir=pref_dir,
                                os_index=osi, os_pvalue=p_osi,
                                os_null=null_dist_osi, pref_or=pref_or,
                                on_off=on_off, d_qi=d_qi, u=u, v=v,
                                surrogate_v=surrogate_v, surrogate_dsi=dsi_s,
                                avg_sorted_resp=avg_sorted_responses))
            print("Populated", key)
        else:
            print("Skipped for ", key)

    def plot1(self, key):
        key = {k: v for k, v in key.items() if k in self.primary_key}
        dir_order = (self.stimulus_table() & key).fetch1('trial_info')
        sorted_directions_rad = np.deg2rad(np.sort(dir_order))

        v, ds_index, pref_dir, avg_sorted_resp = \
            (self & key).fetch1('v', 'ds_index', 'pref_dir', 'avg_sorted_resp')

        fig, axs = plt.subplots(3, 3, figsize=(6, 6), facecolor='w', subplot_kw=dict(frameon=False))
        axs[1, 1].remove()
        ax = fig.add_subplot(3, 3, 5, projection='polar', frameon=False)
        temp = np.max(np.append(v, ds_index))
        ax.plot((0, np.pi), (temp * 1.2, temp * 1.2), color='gray')
        ax.plot((np.pi / 2, np.pi / 2 * 3), (temp * 1.2, temp * 1.2), color='gray')
        ax.plot([0, pref_dir], [0, ds_index * np.sum(v)], color='r')
        ax.plot(np.append(sorted_directions_rad, sorted_directions_rad[0]), np.append(v, v[0]), color='k')
        ax.set_rmin(0)
        ax.set_thetalim([0, 2 * np.pi])
        ax.set_yticks([])
        ax.set_xticks([])
        ax_idxs = [0, 1, 2, 3, 5, 6, 7, 8]
        dir_idxs = [3, 2, 1, 4, 0, 5, 6, 7]
        vmin, vmax = avg_sorted_resp.min(), avg_sorted_resp.max()

        for ax_idx, dir_idx in zip(ax_idxs, dir_idxs):
            ax = axs.flat[ax_idx]
            # ax.set_title(f"{np.sort(dir_order)[dir_idx]}")
            ax.plot(avg_sorted_resp[:, dir_idx], color='k')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylim([vmin - vmax * 0.2, vmax * 1.2])
            ax.set_xlim([-len(avg_sorted_resp) * 0.2, len(avg_sorted_resp) * 1.2])
        return None

def quality_index_ds(raw_sorted_resp_mat):
    """
    This function computes the quality index for responses to moving bar as described in
    Baden et al 2016. QI is computed for each direction separately and the best QI is taken
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

def sort_response_matrix(snippets, idxs, directions):
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

def get_time_dir_kernels(sorted_responses):
    """
    Performs singular value decomposition on the time x direction matrix (averaged across repetitions)
    Uses a heuristic to try to determine whether a sign flip occurred during svd
    For the time course, the mean of the first second is subtracted and then the vector is divided by the maximum
    absolute value.
    For the direction/orientation tuning curve, the vector is normalized to the range (0,1)
    Input:
    sorted_responses:   array, time x direction

    Outputs:
    time_kernel     array, time x 1 (time component, 1st component of U)
    direction_tuning    array, directions x 1 (direction tuning, 1st component of V)
    singular_value  float, 1st singular value
    """

    U, S, V = np.linalg.svd(sorted_responses)
    u = U[:, 0]
    s = S[0]
    v = V[0, :]
    # the time_kernel determined by SVD should be correlated to the average response across all directions. if the
    # correlation is negative, U is likely flipped
    r, _ = stats.spearmanr(u, np.mean(sorted_responses, axis=-1), axis=1)
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

    u = s*u
    #determine which entries correspond to the first second, assuming 4 seconds presentation time
    idx = int(len(u)/4)
    u -= np.mean(u[:idx])
    u = u/np.max(abs(u))
    v = s*v
    v -= np.min(v)
    v /= np.max(abs(v))

    return u, v, s


def get_si(v, dirs, per):
    """
    Computes direction/orientation selectivity index and preferred direction/orientation of a cell by projecting the tuning curve v on a
    complex exponential of the according directions dirs (as in Baden et al. 2016)
    Inputs:
    v:  array, dirs x 1, tuning curve as returned by SVD
    dirs:   array, dirs x 1, directions in radians
    per:    int (1 or 2), indicating whether direction (1) or orientation (2) shall be tested

    Output:
    index:  float, D/O si
    direction:  float, preferred D/O
    """
    bin_spacing = np.diff(per*dirs)[0]
    correction_factor = bin_spacing / (2 * (np.sin(bin_spacing / 2)))  # Zar 1999, Equation 26.16
    complExp = [np.exp(per * 1j * d) for d in dirs]
    vector = np.dot(complExp, v)
    index = correction_factor * np.abs(vector)/np.sum(v)  # get the absolute of the vector, normalize to make it range between 0 and 1
    direction = cmath.phase(vector)/per
    #for orientation, the directions are mapped to the right half of a circle. Map instead to upper half
    if per == 2 and direction < 0:
        direction+=np.pi
    return index, direction

def gen_null_distribution(sorted_response, dirs, per, iters=1000):
    """
    Generates a null distribution of
    """
    (t, d, r) = sorted_response.shape
    reshaped = np.reshape(sorted_response, (t, d*r))
    null_dist = np.zeros(iters)
    rand_idx = np.linspace(0, d*r - 1, d*r, dtype=int)
    for i in range(iters):
        np.random.shuffle(rand_idx)
        shuffled_response = np.reshape(reshaped[:, rand_idx], (t, d, r))
        _, v, _ = get_time_dir_kernels(np.mean(shuffled_response, axis=-1))
        idx, _, = get_si(v, dirs, per)
        null_dist[i] = idx
    return null_dist

def test_tuning(rep_dir_resps, dirs, per, iters=1000):
    """

    """
    bin_spacing = np.diff(dirs)[0]
    # correction_factor = bin_spacing / (2 * (np.sin(bin_spacing / 2)))
    # compl_exp = [np.exp(per * 1j * d) for d in dirs]
    (rep_n, dir_n) = rep_dir_resps.shape
    flattened = np.reshape(rep_dir_resps, (rep_n*dir_n))
    rand_idx = np.linspace(0, rep_n*dir_n-1, rep_n*dir_n, dtype=int)
    null_dist = np.zeros(iters)
    for i in range(iters):
        np.random.shuffle(rand_idx)
        shuffled = flattened[rand_idx]
        shuffled = np.reshape(shuffled, (rep_n, dir_n))
        shuffled_mean = np.mean(shuffled, axis=0)
        normalized_shuffled_mean = shuffled_mean-np.min(shuffled_mean)
        normalized_shuffled_mean /= np.max(abs(normalized_shuffled_mean))
        dsi, _ = get_si(normalized_shuffled_mean, dirs, per)
        null_dist[i] = dsi

    return null_dist

def get_on_off_index(time_kernel):
    """
    Computes a preliminary On-Off Index based on the responses to the On (first half) and the OFF (2nd half) part of
    the responses to the moving bars stimulus
    :param avg_response_matrix: average response matrix (average across repetitions per condition), time x directions
    :return:
    """
    normed_kernel = time_kernel - np.min(time_kernel)
    normed_kernel = normed_kernel/np.max(normed_kernel)
    deriv = np.diff(normed_kernel)
    on_response = np.max(deriv[9:19])
    off_response = np.max(deriv[19:29])
    off_response = np.max((0, off_response))
    on_response = np.max((0, on_response))
    on_off = (on_response - off_response)/(on_response + off_response)
    on_off = np.round(on_off, 2)
    if np.isnan(on_off):
        on_off = 0
    return on_off
