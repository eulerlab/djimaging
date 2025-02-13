"""
This is based on the first python implementation. It introduced some changes compared to the original MATLAB code
from Baden et al. 2016. This results in fewer p-values < 0.05, but the overall results are similar.
However, for the classifier this will introduce a bias towards non DS cells.
"""

import numpy as np
from scipy import stats

from djimaging.tables.response.movingbar.orientation_utils import quality_index_ds, get_on_off_index, \
    preprocess_mb_snippets, get_si
from djimaging.utils import math_utils


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


def compute_os_ds_idxs(snippets: np.ndarray, dir_order: np.ndarray, dt: float, n_shuffles: int = 100):
    assert snippets.ndim == 2
    assert np.asarray(dir_order).ndim == 1

    sorted_directions, sorted_responses, sorted_averages = preprocess_mb_snippets(snippets, dir_order)

    time_component, dir_component = get_time_dir_kernels(sorted_averages, dt=dt)

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
    null_dist_dsi = compute_null_dist(np.transpose(projected), sorted_directions, 1, iters=n_shuffles)
    p_dsi = np.mean(null_dist_dsi > dsi_s)
    null_dist_osi = compute_null_dist(np.transpose(projected), sorted_directions, 2, iters=n_shuffles)
    p_osi = np.mean(null_dist_osi > osi_s)
    d_qi = quality_index_ds(sorted_responses)
    on_off = get_on_off_index(time_component, dt=dt)

    return dsi, p_dsi, null_dist_dsi, pref_dir, osi, p_osi, null_dist_osi, pref_or, \
        on_off, d_qi, time_component, dir_component, surrogate_v, dsi_s, sorted_averages
