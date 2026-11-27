"""Additional tests for branches of djimaging.utils.math_utils not covered by the existing math tests."""
import numpy as np
import pytest

from djimaging.utils import math_utils


@pytest.mark.parametrize('norm_kind, func', [
    ('zscore', math_utils.normalize_zscore),
    ('soft_zero_one', math_utils.normalize_soft_zero_one),
    ('amp_one', math_utils.normalize_amp_one),
    ('std_one', math_utils.normalize_std_one),
])
def test_normalize_dispatch(norm_kind, func):
    x = np.array([1., -2., 4., 0.5, 3.])
    np.testing.assert_allclose(math_utils.normalize(x, norm_kind), func(x))


def test_normalize_dispatch_none_returns_input():
    x = np.array([1., 2.])
    assert math_utils.normalize(x, 'none') is x


def test_normalize_dispatch_forwards_kwargs():
    x = np.arange(11.)
    np.testing.assert_allclose(math_utils.normalize(x, 'soft_zero_one', dq=10, clip=False), (x - 1.) / 8.)


def test_normalize_soft_zero_one_hand_computed():
    x = np.arange(11.)  # 10th percentile = 1, 90th percentile = 9
    np.testing.assert_allclose(math_utils.normalize_soft_zero_one(x, dq=10, clip=True),
                               np.clip((x - 1.) / 8., 0, 1))


def test_truncated_vstack_exceeding_rtol_raises():
    with pytest.raises(ValueError, match='rtol'):
        math_utils.truncated_vstack([np.arange(10.), np.arange(8.)], rtol=0.1)


def test_truncated_vstack_rtol_boundary_allowed():
    # (10 - 9) / 9.5 = 0.105 -> allowed for rtol=0.11
    out = math_utils.truncated_vstack([np.arange(10.), np.arange(9.)], rtol=0.11)
    assert out.shape == (2, 9)


def test_padded_vstack_default_nan():
    out = math_utils.padded_vstack([np.array([1., 2.]), np.array([3.])])
    np.testing.assert_array_equal(out, [[1., 2.], [3., np.nan]])
