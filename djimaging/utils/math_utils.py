import numpy as np


def normalize_zero_one(x):
    """Normalize array to be between zero and one"""
    x = np.asarray(x)
    # Use absolute values in case of complex numbers, not sure if necessary
    return (x - np.min(x)) / np.abs((np.max(x) - np.min(x)))
