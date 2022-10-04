import numpy as np

from djimaging.utils import math_utils


def test_normalize_zero_one_1darray():
    x = np.array([1., 2./3., 7.5, np.pi])
    obs = math_utils.normalize_zero_one(x)
    assert np.isclose(np.max(obs), 1.)
    assert np.isclose(np.min(obs), 0.)


def test_normalize_zero_one_2darray():
    x = np.array([[1., 2., 7.5, np.pi], [4.333, 2.15423, 7.5, 3.]])
    obs = math_utils.normalize_zero_one(x)
    assert np.isclose(np.max(obs), 1.)
    assert np.isclose(np.min(obs), 0.)
