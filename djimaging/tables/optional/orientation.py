import datajoint as dj
import numpy as np
from scipy import stats
import cmath

from djimaging.utils.dj_utils import PlaceholderTable
from djimaging.utils.math_utils import normalize_zero_one


class OsDsIndexesTemplate(dj.Computed):
    database = ""  # hack to suppress DJ error

    @property
    def definition(self):
        definition = """
        #This class computes the direction and orientation selectivity indexes 
        #as well as a quality index of DS responses as described in Baden et al. (2016)
        -> self.detrendsnippets_table
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
    detrendsnippets_table = PlaceholderTable

    @property
    def key_source(self):
        return self.detrendsnippets_table() & 'stim_id = 2'

    def make(self, key):

        dir_order = (self.stimulus_table() & key).fetch1('trialinfo')
        snippets = (self.detrendsnippets_table() & key).fetch1('detrend_snippets')  # get the response snippets

        sorted_responses, sorted_directions_rad = _sort_response_matrix(snippets, dir_order)
        avg_sorted_responses = np.mean(sorted_responses, axis=-1)
        try:
            u, v, s = _get_time_dir_kernels(avg_sorted_responses)
        except np.linalg.LinAlgError:
            print(f'ERROR: LinAlgError for key {key}')
            return

        dsi, pref_dir = _get_si(v, sorted_directions_rad, 1)
        osi, pref_or = _get_si(v, sorted_directions_rad, 2)

        (t, d, r) = sorted_responses.shape
        # make the result between the original and the shuffled comparable
        projected = np.dot(np.transpose(np.reshape(sorted_responses, (t, d * r))), u)
        projected = np.reshape(projected, (d, r))
        surrogate_v = normalize_zero_one(np.mean(projected, axis=-1))

        dsi_s, pref_dir_s = _get_si(surrogate_v, sorted_directions_rad, 1)
        osi_s, pref_or_s = _get_si(surrogate_v, sorted_directions_rad, 2)
        null_dist_dsi = _compute_null_dist(np.transpose(projected), sorted_directions_rad, 1)
        p_dsi = np.mean(null_dist_dsi > dsi_s)
        null_dist_osi = _compute_null_dist(np.transpose(projected), sorted_directions_rad, 2)
        p_osi = np.mean(null_dist_osi > osi_s)
        d_qi = _quality_index_ds(sorted_responses)
        on_off = _get_on_off_index(u)

        self.insert1(dict(**key,
                          ds_index=dsi, ds_pvalue=p_dsi,
                          ds_null=null_dist_dsi, pref_dir=pref_dir,
                          os_index=osi, os_pvalue=p_osi,
                          os_null=null_dist_osi, pref_or=pref_or,
                          on_off=on_off, d_qi=d_qi, u=u, v=v,
                          surrogate_v=surrogate_v, surrogate_dsi=dsi_s,
                          avg_sorted_resp=avg_sorted_responses))


def _quality_index_ds(raw_sorted_resp_mat):
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


def _sort_response_matrix(snippets, dir_order):
    """
    Sorts the snippets according to stimulus condition and repetition into a time x direction x repetition matrix
    :param snippets: arraylike, time x (directions*repetitions)
    :param dir_order: arraylike, order of directions of moving bar
    :returns sorted_responses: array, time x direction x repetitions, relative to sorted directions
    :returns sorted_directions: sorted directions in radians
    """

    nreps = snippets.shape[1]
    dir_order = dir_order.flatten()

    if nreps != dir_order.size:
        assert nreps % dir_order.size == 0, 'directions must be a multiple of reps'
        dir_order = np.tile(dir_order, int(nreps / dir_order.size))

    dir_deg, dir_counts = np.unique(dir_order, return_counts=True)
    assert np.all(dir_counts == dir_counts[0]), 'Different number of repetitions per direction not implemented'

    dir_idx = [np.where(dir_order == d)[0] for d in dir_deg]

    sorted_responses = snippets[:, dir_idx]
    sorted_directions_rad = np.deg2rad(dir_deg)

    return sorted_responses, sorted_directions_rad


def _get_time_dir_kernels(sorted_responses):
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
    v = V[0, :]
    # the time_kernel determined by SVD should be correlated to the average response across all directions. if the
    # correlation is negative, U is likely flipped
    r, _ = stats.spearmanr(a=u, b=np.mean(sorted_responses, axis=-1), axis=1)
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

    u = s * u
    # determine which entries correspond to the first second, assuming 4 seconds presentation time
    idx = int(len(u) / 4)
    u -= np.mean(u[:idx])
    u = u / np.max(abs(u))
    v = normalize_zero_one(s * v)
    return u, v, s


def _get_si(v, dirs, per):
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
    bin_spacing = np.diff(per * dirs)[0]
    correction_factor = bin_spacing / (2 * (np.sin(bin_spacing / 2)))  # Zar 1999, Equation 26.16
    complExp = [np.exp(per * 1j * d) for d in dirs]
    vector = np.dot(complExp, v)
    index = correction_factor * np.abs(vector) / np.sum(
        v)  # get the absolute of the vector, normalize to make it range between 0 and 1
    direction = cmath.phase(vector) / per
    # for orientation, the directions are mapped to the right half of a circle. Map instead to upper half
    if per == 2 and direction < 0:
        direction += np.pi
    return index, direction


def _gen_null_distribution(sorted_response, dirs, per, iters=1000):
    """
    Generates a null distribution of
    """
    (t, d, r) = sorted_response.shape
    reshaped = np.reshape(sorted_response, (t, d * r))
    null_dist = np.zeros(iters)
    rand_idx = np.linspace(0, d * r - 1, d * r, dtype=int)
    for i in range(iters):
        np.random.shuffle(rand_idx)
        shuffled_response = np.reshape(reshaped[:, rand_idx], (t, d, r))
        _, v, _ = _get_time_dir_kernels(np.mean(shuffled_response, axis=-1))
        idx, _, = _get_si(v, dirs, per)
        null_dist[i] = idx
    return null_dist


def _get_on_off_index(time_kernel):
    """
    Computes a preliminary On-Off Index based on the responses to the On (first half) and the OFF (2nd half) part of
    the responses to the moving bars stimulus
    """
    deriv = np.diff(normalize_zero_one(time_kernel))
    on_response = np.max(deriv[9:19])  # TODO: This is very hardcoded. Probably one sho
    off_response = np.max(deriv[19:29])
    off_response = np.max((0, off_response))
    on_response = np.max((0, on_response))
    on_off = (on_response - off_response) / (on_response + off_response)
    on_off = np.round(on_off, 2)
    if np.isnan(on_off):
        on_off = 0
    return on_off


def _compute_null_dist(rep_dir_resps, dirs, per, iters=1000):
    """Compute null distribution"""
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
        dsi, _ = _get_si(normalized_shuffled_mean, dirs, per)
        null_dist[i] = dsi

    return null_dist
