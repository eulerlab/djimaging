"""Conversions between python-format ROI masks (background 0, ROIs > 0) and Igor-format masks
(background 1, ROIs < 0)."""
import numpy as np
import pytest

from djimaging.utils.mask_format_utils import (
    as_igor_format, as_python_format, assert_igor_format, is_igor_format, is_python_format, to_igor_format,
    to_python_format,
)


def _python_mask():
    """Non-square python mask with three ROIs, one touching the border, one single pixel."""
    mask = np.zeros((4, 6), dtype=int)
    mask[0, 0:2] = 1  # touches the top-left border
    mask[2:4, 3] = 2
    mask[3, 5] = 3  # single pixel in the corner
    return mask


def _igor_mask():
    mask = np.ones((4, 6), dtype=int)
    mask[0, 0:2] = -1
    mask[2:4, 3] = -2
    mask[3, 5] = -3
    return mask


def test_to_igor_format_values():
    np.testing.assert_array_equal(to_igor_format(_python_mask()), _igor_mask())


def test_to_python_format_values():
    np.testing.assert_array_equal(to_python_format(_igor_mask()), _python_mask())


def test_round_trips():
    py = _python_mask()
    np.testing.assert_array_equal(to_python_format(to_igor_format(py)), py)
    igor = _igor_mask()
    np.testing.assert_array_equal(to_igor_format(to_python_format(igor)), igor)


def test_conversions_do_not_modify_input():
    py, igor = _python_mask(), _igor_mask()
    to_igor_format(py)
    to_python_format(igor)
    np.testing.assert_array_equal(py, _python_mask())
    np.testing.assert_array_equal(igor, _igor_mask())


def test_to_igor_format_rejects_igor_mask():
    with pytest.raises(ValueError, match='not in python format'):
        to_igor_format(_igor_mask())


def test_to_python_format_rejects_python_mask_with_several_rois():
    with pytest.raises(ValueError, match='unexpected values'):
        to_python_format(_python_mask())


def test_to_python_format_replaces_11_by_background():
    igor = _igor_mask()
    igor[1, 1] = 11
    np.testing.assert_array_equal(to_python_format(igor), _python_mask())


def test_to_python_format_background_zero():
    """Igor masks whose maximum is 0 use 0 as background."""
    igor = np.array([[0, -1, -1], [0, 0, -2]])
    np.testing.assert_array_equal(to_python_format(igor), [[0, 1, 1], [0, 0, 2]])


def test_to_python_format_empty_igor_mask():
    np.testing.assert_array_equal(to_python_format(np.ones((2, 3), dtype=int)), np.zeros((2, 3)))


def test_to_igor_format_float_mask():
    out = to_igor_format(np.array([[0., 1.], [2., 0.]]))
    np.testing.assert_array_equal(out, [[1, -1], [-2, 1]])


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
def test_to_igor_format_unsigned_mask(dtype):
    out = to_igor_format(_python_mask().astype(dtype))
    np.testing.assert_array_equal(out, _igor_mask())


def test_assert_igor_format():
    assert_igor_format(_igor_mask())
    with pytest.raises(ValueError, match='2D'):
        assert_igor_format(np.ones((2, 2, 2), dtype=int))
    with pytest.raises(ValueError, match='integer'):
        assert_igor_format(np.array([[1., -0.5]]))
    with pytest.raises(ValueError, match='unexpected values'):
        assert_igor_format(np.array([[2, -1]]))


def test_is_igor_and_is_python_format():
    assert is_igor_format(_igor_mask())
    assert not is_igor_format(_python_mask())
    assert not is_igor_format(np.ones((2, 2, 2), dtype=int))
    assert not is_igor_format(np.array([[1., -0.5]]))
    assert not is_igor_format(np.array([1, -1]))

    assert is_python_format(_python_mask())
    assert not is_python_format(_igor_mask())
    assert not is_python_format(np.zeros((2, 2, 2), dtype=int))
    assert not is_python_format(np.array([[0., 0.5]]))
    assert not is_python_format(np.array([0, 1]))


def test_as_formats_convert_only_when_needed():
    py, igor = _python_mask(), _igor_mask()
    assert as_python_format(py) is py
    assert as_igor_format(igor) is igor
    np.testing.assert_array_equal(as_python_format(igor), py)
    np.testing.assert_array_equal(as_igor_format(py), igor)
