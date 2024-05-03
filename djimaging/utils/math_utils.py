import numpy as np


def normalize(x, norm_kind, **kwargs):
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
    """Normalize array to be between zero and one"""
    x = np.asarray(x)
    vmin = np.nanmin(x)
    vmax = np.nanmax(x)
    vrng = vmax - vmin
    if vrng == 0.:
        return x - vmin
    else:
        return (x - vmin) / vrng


def normalize_soft_zero_one(x, dq=5, clip=True):
    """Normalize array to be mostly between zero and one. If clip is True, clip at zero and one after scaling."""
    x = np.asarray(x)
    x = (x - np.percentile(x, q=dq)) / np.abs((np.percentile(x, q=100 - dq) - np.percentile(x, q=dq)))
    if clip:
        x = np.clip(x, 0, 1)
    return x


def normalize_amp_one(x):
    """Scale array such that the maximum absolute amplitude is one. Does not change zero."""
    x = np.asarray(x)
    return x / np.max(np.abs(x))


def normalize_std_one(x):
    """Scale array such that the maximum absolute amplitude is one standard deviation. Does not change zero."""
    x = np.asarray(x)
    return x / np.std(x)


def normalize_amp_std(x):
    """Scale array such that the maximum absolute amplitude is one standard deviation. Does not change zero."""
    x = np.asarray(x)
    return x / np.std(x)


def normalize_zscore(x):
    """Standardize array to zero mean and standard deviation one."""
    x = np.asarray(x)
    return (x - np.mean(x)) / np.std(x)


def truncated_vstack(traces, rtol=0.05):
    """Vertical stack of traces. Allows to have different lengths. Raise error is lengths differ more than rtol."""
    sizes = [len(trace) for trace in traces]
    min_size, max_size, mean_size = np.min(sizes), np.max(sizes), np.mean(sizes)
    rel_difference = (max_size - min_size) / mean_size
    if rel_difference > rtol:
        raise ValueError(f"Lengths ranged from {min_size} to {max_size}, {rel_difference:.0%} > rtol={rtol:.0%}")
    return np.vstack([trace[:min_size] for trace in traces])


def padded_vstack(traces, cval=np.nan):
    """Vertical stack of traces. Allows to have different lengths. Fill with cval, which is NaN per default."""
    sizes = [len(trace) for trace in traces]
    max_size = np.max(sizes)
    return np.vstack([trace if len(trace) == max_size else np.pad(trace, (0, max_size - len(trace)),
                                                                  mode='constant', constant_values=(cval, cval))
                      for trace in traces])
