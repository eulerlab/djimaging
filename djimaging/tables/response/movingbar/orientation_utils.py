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
