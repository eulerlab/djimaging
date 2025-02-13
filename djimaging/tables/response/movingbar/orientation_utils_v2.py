"""
This code is very close to the original MATLAB code from Baden et al. 2016.
"""

import numpy as np

from djimaging.tables.response.movingbar.orientation_utils import quality_index_ds, get_on_off_index, \
    preprocess_mb_snippets, get_si


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
        sorted_averages,
    )
