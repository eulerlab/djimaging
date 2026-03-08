import numpy as np
import pytest

from djimaging.utils.dj_utils import check_unique_one, is_equal, make_hash, merge_keys


def test_check_unique_one_int():
    values = [1, 1, 1]
    name = 'values'
    v_obs = check_unique_one(values, name=name)
    assert v_obs == 1


def test_check_unique_one_float():
    values = [1., 1., 1.0000]
    name = 'values'
    v_obs = check_unique_one(values, name=name)
    assert v_obs == 1.


def test_check_unique_one_str():
    values = ["as", "as", "as"]
    v_obs = check_unique_one(values)
    assert v_obs == "as"


def test_check_unique_error():
    values = [1, 2, 3]
    name = 'values'
    with pytest.raises(ValueError):
        check_unique_one(values, name=name)


# --- is_equal ---

def test_is_equal_scalars_equal():
    assert is_equal(1, 1) is True


def test_is_equal_scalars_not_equal():
    assert is_equal(1, 2) is False


def test_is_equal_strings_equal():
    assert is_equal('abc', 'abc') is True


def test_is_equal_strings_not_equal():
    assert is_equal('abc', 'xyz') is False


def test_is_equal_arrays_equal():
    v1 = np.array([1, 2, 3])
    v2 = np.array([1, 2, 3])
    assert is_equal(v1, v2) is True


def test_is_equal_arrays_not_equal():
    v1 = np.array([1, 2, 3])
    v2 = np.array([1, 2, 4])
    assert is_equal(v1, v2) is False


# --- make_hash ---

def test_make_hash_returns_string():
    result = make_hash({'a': 1, 'b': 2})
    assert isinstance(result, str)
    assert len(result) == 32


def test_make_hash_identical_dicts_same_hash():
    h1 = make_hash({'x': 1, 'y': [1, 2, 3]})
    h2 = make_hash({'x': 1, 'y': [1, 2, 3]})
    assert h1 == h2


def test_make_hash_key_order_independent():
    """Hash should be the same regardless of dict key insertion order."""
    h1 = make_hash({'a': 1, 'b': 2})
    h2 = make_hash({'b': 2, 'a': 1})
    assert h1 == h2


def test_make_hash_different_values_different_hash():
    h1 = make_hash({'a': 1})
    h2 = make_hash({'a': 2})
    assert h1 != h2


def test_make_hash_nested_structure():
    h1 = make_hash({'a': {'b': [1, 2, 3]}})
    h2 = make_hash({'a': {'b': [1, 2, 3]}})
    assert h1 == h2


# --- merge_keys ---

def test_merge_keys_disjoint():
    k1 = {'a': 1}
    k2 = {'b': 2}
    merged = merge_keys(k1, k2)
    assert merged == {'a': 1, 'b': 2}


def test_merge_keys_overlapping_equal():
    k1 = {'a': 1, 'b': 2}
    k2 = {'b': 2, 'c': 3}
    merged = merge_keys(k1, k2)
    assert merged == {'a': 1, 'b': 2, 'c': 3}


def test_merge_keys_overlapping_conflict_raises():
    k1 = {'a': 1, 'b': 2}
    k2 = {'b': 99}
    with pytest.raises(ValueError):
        merge_keys(k1, k2)
