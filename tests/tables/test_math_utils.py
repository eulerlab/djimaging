import numpy as np

from djimaging.utils import math_utils


def test_normalize_zero_one():
    x = np.array([1., 2. / 3., 7.5, np.pi])
    obs = math_utils.normalize_zero_one(x)
    assert np.isclose(np.max(obs), 1.)
    assert np.isclose(np.min(obs), 0.)


def test_normalize_soft_zero_one():
    x = np.array([1., 2. / 3., 7.5, np.pi])
    obs = math_utils.normalize_soft_zero_one(x, dq=5, clip=True)
    assert np.isclose(np.max(obs), 1.)
    assert np.isclose(np.min(obs), 0.)


def test_normalize_amp_one():
    x = np.array([1., 2. / 3., 7.5, np.pi])
    obs = math_utils.normalize_amp_one(x)
    assert np.isclose(np.max(np.abs(obs)), 1.)


def normalize_amp_std():
    x = np.array([1., 2. / 3., 7.5, np.pi])
    obs = math_utils.normalize_amp_std(x)
    assert np.isclose(np.std(obs), 1.)


def test_padded_vstack():
    trace_a = np.array([1., 2., 3., 4.])
    trace_b = np.array([1., 2., 3., 4., 5.])
    obs = math_utils.padded_vstack([trace_a, trace_b], cval=8)
    assert np.allclose(obs, np.array([[1., 2., 3., 4., 8.], [1., 2., 3., 4., 5.]]))


def test_truncated_vstack():
    trace_a = np.array([1., 2., 3., 4.])
    trace_b = np.array([1., 2., 3., 4., 5.])
    obs = math_utils.truncated_vstack([trace_a, trace_b], rtol=0.5)
    assert np.allclose(obs, np.array([[1., 2., 3., 4.], [1., 2., 3., 4.]]))
