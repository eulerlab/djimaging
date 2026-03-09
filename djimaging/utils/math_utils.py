import numpy as np


def normalize(x, norm_kind, **kwargs):
    """Normalize an array using the specified normalization method.

    Parameters
    ----------
    x : array-like
        Input array to normalize.
    norm_kind : str
        Normalization method. One of ``'zscore'``, ``'zero_one'``,
        ``'soft_zero_one'``, ``'amp_one'``, ``'std_one'``, or ``'none'``.
    **kwargs
        Additional keyword arguments forwarded to the selected normalization
        function (e.g. ``dq`` and ``clip`` for ``'soft_zero_one'``).

    Returns
    -------
    np.ndarray
        Normalized array.

    Raises
    ------
    NotImplementedError
        If `norm_kind` is not one of the supported methods.
    """
    if norm_kind == 'zscore':
        return normalize_zscore(x)
    elif norm_kind == 'zero_one':
        return normalize_zero_one(x)
    elif norm_kind == 'soft_zero_one':
        return normalize_soft_zero_one(x, **kwargs)
    elif norm_kind == 'amp_one':
        return normalize_amp_one(x)
    elif norm_kind == 'std_one':
        return normalize_std_one(x)
    elif norm_kind == 'none':
        return x
    else:
        raise NotImplementedError(norm_kind)


def normalize_zero_one(x):
    """Normalize array to the range [0, 1].

    Parameters
    ----------
    x : array-like
        Input array.

    Returns
    -------
    np.ndarray
        Array linearly scaled so that its minimum maps to 0 and its maximum
        maps to 1. If the range is 0, the shifted array (``x - min``) is
        returned unchanged.
    """
    x = np.asarray(x)
    vmin = np.nanmin(x)
    vmax = np.nanmax(x)
    vrng = vmax - vmin
    if vrng == 0.:
        return x - vmin
    else:
        return (x - vmin) / vrng


def normalize_soft_zero_one(x, dq=5, clip=True):
    """Normalize array to mostly fall in [0, 1] using percentile-based scaling.

    Parameters
    ----------
    x : array-like
        Input array.
    dq : float, optional
        Percentile (0–50) used to define the lower and upper anchors for
        scaling. The ``dq``-th percentile maps to 0 and the
        ``(100 - dq)``-th percentile maps to 1. Default is 5.
    clip : bool, optional
        If True (default), values outside [0, 1] are clipped after scaling.

    Returns
    -------
    np.ndarray
        Scaled (and optionally clipped) array.
    """
    x = np.asarray(x)
    x = (x - np.percentile(x, q=dq)) / np.abs((np.percentile(x, q=100 - dq) - np.percentile(x, q=dq)))
    if clip:
        x = np.clip(x, 0, 1)
    return x


def normalize_amp_one(x):
    """Scale array so that the maximum absolute value equals one.

    The zero point of the array is preserved (i.e. the array is only
    divided by a scalar, not shifted).

    Parameters
    ----------
    x : array-like
        Input array.

    Returns
    -------
    np.ndarray
        Array divided by ``max(|x|)``.
    """
    x = np.asarray(x)
    return x / np.max(np.abs(x))


def normalize_std_one(x):
    """Scale array so that its standard deviation equals one.

    The zero point of the array is preserved (i.e. the array is only
    divided by a scalar, not shifted).

    Parameters
    ----------
    x : array-like
        Input array.

    Returns
    -------
    np.ndarray
        Array divided by ``std(x)``.
    """
    x = np.asarray(x)
    return x / np.std(x)


def normalize_amp_std(x):
    """Scale array so that its standard deviation equals one.

    Alias for :func:`normalize_std_one`. The zero point of the array is
    preserved (i.e. the array is only divided by a scalar, not shifted).

    Parameters
    ----------
    x : array-like
        Input array.

    Returns
    -------
    np.ndarray
        Array divided by ``std(x)``.
    """
    x = np.asarray(x)
    return x / np.std(x)


def normalize_zscore(x):
    """Standardize array to zero mean and unit standard deviation (z-score).

    Parameters
    ----------
    x : array-like
        Input array.

    Returns
    -------
    np.ndarray
        Array with mean 0 and standard deviation 1, computed as
        ``(x - mean(x)) / std(x)``.
    """
    x = np.asarray(x)
    return (x - np.mean(x)) / np.std(x)


def truncated_vstack(traces, rtol=0.05):
    """Vertically stack traces of potentially different lengths by truncating to the shortest.

    All traces are truncated to the length of the shortest trace before
    stacking. An error is raised when the relative length spread exceeds
    `rtol`.

    Parameters
    ----------
    traces : sequence of array-like
        Collection of 1-D arrays (traces) to stack.  Each element is
        treated as a single row in the output array.
    rtol : float, optional
        Maximum allowed relative difference between the shortest and
        longest trace, computed as
        ``(max_len - min_len) / mean_len``. Default is 0.05.

    Returns
    -------
    np.ndarray
        2-D array of shape ``(n_traces, min_length)``.

    Raises
    ------
    ValueError
        If the relative length difference across traces exceeds `rtol`.
    """
    sizes = [len(trace) for trace in traces]
    min_size, max_size, mean_size = np.min(sizes), np.max(sizes), np.mean(sizes)
    rel_difference = (max_size - min_size) / mean_size
    if rel_difference > rtol:
        raise ValueError(f"Lengths ranged from {min_size} to {max_size}, {rel_difference:.0%} > rtol={rtol:.0%}")
    return np.vstack([trace[:min_size] for trace in traces])


def padded_vstack(traces, cval=np.nan):
    """Vertically stack traces of potentially different lengths by padding shorter ones.

    Shorter traces are right-padded with `cval` so that all rows reach
    the length of the longest trace before stacking.

    Parameters
    ----------
    traces : sequence of array-like
        Collection of 1-D arrays (traces) to stack.  Each element is
        treated as a single row in the output array.
    cval : scalar, optional
        Constant fill value used to pad shorter traces. Default is
        ``np.nan``.

    Returns
    -------
    np.ndarray
        2-D array of shape ``(n_traces, max_length)``.
    """
    sizes = [len(trace) for trace in traces]
    max_size = np.max(sizes)
    return np.vstack([trace if len(trace) == max_size else np.pad(trace, (0, max_size - len(trace)),
                                                                  mode='constant', constant_values=(cval, cval))
                      for trace in traces])
