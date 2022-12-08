import numpy as np


def normalize_zero_one(x):
    """Normalize array to be between zero and one"""
    x = np.asarray(x)
    # Use absolute values in case of complex numbers, not sure if necessary
    return (x - np.min(x)) / np.abs((np.max(x) - np.min(x)))


def normalize_soft_zero_one(x, dq=5, clip=True):
    x = np.asarray(x)
    x = (x - np.percentile(x, q=dq)) / np.abs((np.percentile(x, q=100 - dq) - np.percentile(x, q=dq)))
    if clip:
        x = np.clip(x, 0, 1)
    return x


def normalize_zscore(x):
    """Normalize array to be between zero and one"""
    x = np.asarray(x)
    return (x - np.mean(x)) / np.std(x)


def truncated_vstack(traces, rtol=0.05):
    sizes = [len(trace) for trace in traces]
    min_size, max_size, mean_size = np.min(sizes), np.max(sizes), np.mean(sizes)
    rel_difference = (max_size - min_size) / mean_size
    if rel_difference > rtol:
        raise ValueError(f"Lengths ranged from {min_size} to {max_size}, {rel_difference:.0%} > rtol={rtol:.0%}")
    return np.vstack([trace[:min_size] for trace in traces])
