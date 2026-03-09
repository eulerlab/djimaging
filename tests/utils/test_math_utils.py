import numpy as np
import pytest

from djimaging.utils import math_utils


def test_normalize_zero_one_1darray():
    x = np.array([1., 2. / 3., 7.5, np.pi])
    obs = math_utils.normalize_zero_one(x)
    assert np.isclose(np.max(obs), 1.)
    assert np.isclose(np.min(obs), 0.)


def test_normalize_zero_one_2darray():
    x = np.array([[1., 2., 7.5, np.pi], [4.333, 2.15423, 7.5, 3.]])
    obs = math_utils.normalize_zero_one(x)
    assert np.isclose(np.max(obs), 1.)
    assert np.isclose(np.min(obs), 0.)


def test_normalize_zero_one_constant_array():
    """Constant array should return all zeros (vrng == 0 branch)."""
    x = np.array([3., 3., 3., 3.])
    obs = math_utils.normalize_zero_one(x)
    assert np.all(obs == 0.)


def test_normalize_zero_one_with_nan():
    """NaN values are ignored when computing min/max."""
    x = np.array([1., np.nan, 3., 5.])
    obs = math_utils.normalize_zero_one(x)
    assert np.isclose(obs[0], 0.)
    assert np.isclose(obs[2], 0.5)
    assert np.isclose(obs[3], 1.)


def test_normalize_soft_zero_one_basic():
    x = np.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
    obs = math_utils.normalize_soft_zero_one(x, dq=5, clip=True)
    assert np.all(obs >= 0.) and np.all(obs <= 1.)


def test_normalize_soft_zero_one_no_clip():
    """Without clipping, values can exceed [0, 1]."""
    x = np.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
    obs = math_utils.normalize_soft_zero_one(x, dq=10, clip=False)
    assert np.any(obs > 1.) or np.any(obs < 0.)


def test_normalize_amp_one():
    x = np.array([1., 2., -3., 0.5])
    obs = math_utils.normalize_amp_one(x)
    assert np.isclose(np.max(np.abs(obs)), 1.)
    assert np.isclose(obs[2], -1.)


def test_normalize_std_one():
    x = np.array([1., 2., 3., 4., 5.])
    obs = math_utils.normalize_std_one(x)
    assert np.isclose(np.std(obs), 1.)


def test_normalize_amp_std():
    x = np.array([1., 2., 3., 4., 5.])
    obs = math_utils.normalize_amp_std(x)
    assert np.isclose(np.std(obs), 1.)


def test_normalize_zscore():
    x = np.array([2., 4., 4., 4., 5., 5., 7., 9.])
    obs = math_utils.normalize_zscore(x)
    assert np.isclose(np.mean(obs), 0.)
    assert np.isclose(np.std(obs), 1.)


def test_normalize_dispatch_zero_one():
    x = np.array([1., 2., 3.])
    obs = math_utils.normalize(x, 'zero_one')
    assert np.isclose(np.min(obs), 0.) and np.isclose(np.max(obs), 1.)


def test_normalize_dispatch_unknown_raises():
    x = np.array([1., 2., 3.])
    with pytest.raises(NotImplementedError):
        math_utils.normalize(x, 'unknown_norm')
