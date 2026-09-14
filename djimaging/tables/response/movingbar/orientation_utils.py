"""Moving-bar selectivity following the Baden et al. 2016 MATLAB implementation."""

import cmath

import numpy as np

from djimaging.utils import math_utils

T_START = 1.152
T_CHANGE = 2.432
T_END = 3.712


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


def get_on_off_index(time_kernel, dt, t_start=T_START, t_change=T_CHANGE, t_end=T_END):
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
        on_off = 0.0
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


def preprocess_mb_snippets(snippets, dir_order):
    dir_idx, dir_rad = get_dir_idx(snippets, dir_order)

    sorted_responses, sorted_directions = sort_response_matrix(snippets, dir_idx, dir_rad)
    sorted_averages = np.mean(sorted_responses, axis=-1)
    return sorted_directions, sorted_responses, sorted_averages


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
    compl_exp = np.array([np.exp(per * 1j * d) for d in dirs])
    vector = np.dot(compl_exp, dir_component)
    # get the absolute of the vector, normalize to make it range between 0 and 1
    index = correction_factor * np.abs(vector) / np.sum(dir_component)

    direction = cmath.phase(vector) / per
    # for orientation, the directions are mapped to the right half of a circle. Map instead to upper half
    if per == 2 and direction < 0:
        direction += np.pi
    return index, direction


def get_time_dir_kernels(sorted_responses: np.ndarray, dt: float):
    """
    Performs singular value decomposition on the time x direction matrix (averaged across repetitions)
    Uses a heuristic to try to determine whether a sign flip occurred during svd
    For the time course, the mean of the first second is subtracted and then the vector is divided by the maximum
    absolute value.
    For the direction/orientation tuning curve, the vector is normalized to the range (0,1)

    Parameters:
    sorted_responses (array): Time x direction matrix.
    dt (float): 1 / sampling_rate of trace.

    Returns:
    tuple: Contains time_kernel (array, time x 1), direction_tuning (array, directions x 1), and singular_value (float).
    """
    # Only the leading components are used; avoid the full time-by-time U matrix.
    U, S, Vh = np.linalg.svd(sorted_responses, full_matrices=False)

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

    dir_component = math_utils.normalize_zero_one(dir_component)

    return time_component, dir_component


def compute_null_dist(dirs: np.ndarray, counts: np.ndarray, per: int, iters=1000):
    """
    Test significance of orientation tuning by permutation test.

    Parameters:
        dirs (array): Vector of directions (#directions x 1) in radians.
        counts (array): Matrix of responses (#reps x #directions).
        per (int): Fourier component to test (1 = direction, 2 = orientation).
        iters (int): Number of permutations for the test.

    Returns:
        p (float): p-value for tuning.
        q (float): Magnitude of the Fourier component.
        qdistr (array): Sampling distribution of |q| under the null hypothesis.
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


def compute_os_ds_idxs(snippets: np.ndarray, dir_order: np.ndarray, dt: float, n_shuffles: int = 1000):
    """
    Compute orientation selectivity (OS) and direction selectivity (DS) indices.

    Parameters:
        snippets (array): 2D array of response snippets.
        dir_order (array): 1D array of direction order.
        dt (float): Time step.
        n_shuffles (int): Number of shuffles for null distribution.

    Returns:
        tuple: Contains various computed indices and components.
    """

    assert snippets.ndim == 2
    assert np.asarray(dir_order).ndim == 1

    sorted_directions, sorted_responses, sorted_averages = preprocess_mb_snippets(snippets, dir_order)

    time_component, dir_component = get_time_dir_kernels(sorted_averages, dt=dt)

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

    p_dsi, _, null_dist_dsi = compute_null_dist(sorted_directions, projected.T, 1, iters=n_shuffles)
    p_osi, _, null_dist_osi = compute_null_dist(sorted_directions, projected.T, 2, iters=n_shuffles)

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
        sorted_averages,
    )
