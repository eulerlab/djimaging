import pytest

from djimaging.utils.dj_utils import check_unique_one


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
