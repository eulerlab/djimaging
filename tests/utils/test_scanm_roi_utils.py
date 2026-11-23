"""Tests of the ROI helpers for ScanM (Igor format) ROI masks: background 0/1, ROI i is encoded as -i."""
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from djimaging.utils.scanm import roi_utils


def _mask():
    mask = np.ones((6, 4), dtype=int)
    mask[0, :] = 0  # background can be 0 or 1
    mask[1, 1:3] = -10
    mask[3:5, 0] = -2
    mask[5, 3] = -1
    return mask


def test_extract_roi_ids_sorted_by_abs_value():
    assert_array_equal(roi_utils.extract_roi_ids(_mask()), [-1, -2, -10])
    assert roi_utils.extract_roi_ids(_mask()).dtype.kind == 'i'


def test_extract_roi_ids_skips_artifact_rows():
    assert_array_equal(roi_utils.extract_roi_ids(_mask(), npixartifact=2), [-1, -2])


def test_extract_roi_ids_float_mask_and_empty():
    assert_array_equal(roi_utils.extract_roi_ids(_mask().astype(float)), [-1, -2, -10])
    assert roi_utils.extract_roi_ids(np.ones((3, 3))).size == 0


def test_extract_roi_ids_rejects_3d():
    with pytest.raises(AssertionError):
        roi_utils.extract_roi_ids(np.ones((2, 2, 2)))


def test_get_roi_center_is_row_col():
    x, y = roi_utils.get_roi_center(_mask(), 10)
    assert (x, y) == (1., 1.5)
    x, y = roi_utils.get_roi_center(_mask(), 2)
    assert (x, y) == (3.5, 0.)


def test_get_roi_center_missing_roi():
    with pytest.raises(ValueError, match='roi_id=3'):
        roi_utils.get_roi_center(_mask(), 3)


def test_get_roi_centers():
    centers = roi_utils.get_roi_centers(_mask(), np.array([1, 2, 10]))
    assert_allclose(centers, [[5., 3.], [3.5, 0.], [1., 1.5]])
    assert roi_utils.get_roi_centers(_mask(), np.array([], dtype=int)).shape == (0, 2)


def test_get_rel_roi_pos_axes_and_pixel_sizes():
    mask = np.ones((8, 6), dtype=int)
    mask[0, 5] = -1  # first row, last column
    d1, d2 = roi_utils.get_rel_roi_pos(1, mask, pixel_size_um=2., pixel_size_d2_um=3.)
    assert d1 < 0 < d2
    # Pixels left of the ROI / above the ROI scale with the respective pixel size
    d1_b, d2_b = roi_utils.get_rel_roi_pos(1, mask, pixel_size_um=4., pixel_size_d2_um=3.)
    assert d1_b == pytest.approx(2 * d1)
    assert d2_b == pytest.approx(d2)
    # Default: isotropic
    assert roi_utils.get_rel_roi_pos(1, mask, pixel_size_um=2.) == pytest.approx((d1, d2 * 2 / 3))


def test_get_rel_roi_pos_rotation():
    mask = np.ones((10, 10), dtype=int)
    mask[0, 3] = -1
    d1, d2 = roi_utils.get_rel_roi_pos(1, mask, pixel_size_um=1.)
    r1, r2 = roi_utils.get_rel_roi_pos(1, mask, pixel_size_um=1., ang_deg=90.)
    assert (r1, r2) == pytest.approx((-d2, d1))
    f1, f2 = roi_utils.get_rel_roi_pos(1, mask, pixel_size_um=1., ang_deg=360.)
    assert (f1, f2) == pytest.approx((d1, d2))
    assert np.hypot(*roi_utils.get_rel_roi_pos(1, mask, 1., ang_deg=33.)) == pytest.approx(np.hypot(d1, d2))


def test_get_rel_roi_pos_centred_roi_is_zero():
    mask = np.ones((4, 6), dtype=int)
    mask[1:3, 2:4] = -1  # exactly in the centre of the field
    assert roi_utils.get_rel_roi_pos(1, mask, pixel_size_um=1.) == pytest.approx((0., 0.))


def test_get_rel_roi_pos_symmetric_rois_are_symmetric():
    """Two ROIs mirrored at the field centre must have mirrored positions (holds only with a correct centre)."""
    mask = np.ones((4, 4), dtype=int)
    mask[0, 0] = -1
    mask[3, 3] = -2
    p1 = np.array(roi_utils.get_rel_roi_pos(1, mask, pixel_size_um=1.))
    p2 = np.array(roi_utils.get_rel_roi_pos(2, mask, pixel_size_um=1.))
    assert_allclose(p1 + p2, [0., 0.])


def test_fix_first_or_last_n_nan():
    trace = np.array([np.nan, np.nan, 1., 2., 3., 4., np.nan, np.nan])
    fixed = roi_utils.fix_first_or_last_n_nan(trace.copy(), n=2)
    assert_array_equal(fixed, [1., 1., 1., 2., 3., 4., 4., 4.])
    # n <= 0 returns unchanged
    out = roi_utils.fix_first_or_last_n_nan(trace.copy(), n=0)
    assert np.isnan(out[0])
    # Nothing to fix
    clean = np.arange(6.)
    assert_array_equal(roi_utils.fix_first_or_last_n_nan(clean.copy(), n=3), clean)


def test_fix_first_or_last_n_nan_2d():
    trace = np.arange(12.).reshape(6, 2)
    trace[0, 1] = np.nan
    fixed = roi_utils.fix_first_or_last_n_nan(trace.copy(), n=1)
    assert_array_equal(fixed[0], trace[1])


@pytest.mark.xfail(strict=True, reason="BUG: all first/last n samples are overwritten with trace[n] if any is NaN, "
                                       "also valid samples; docstring promises NaNs replaced by nearest neighbour")
def test_fix_first_or_last_n_nan_keeps_valid_samples():
    trace = np.array([np.nan, 1., 2., 3., 4., 5., 6.])
    fixed = roi_utils.fix_first_or_last_n_nan(trace.copy(), n=3)
    assert_array_equal(fixed, [1., 1., 2., 3., 4., 5., 6.])


def test_get_roi2trace_2d():
    traces = np.arange(20.).reshape(10, 2)
    times = np.tile(np.arange(10.)[:, None], (1, 2)) + np.array([0., 0.5])
    roi2trace = roi_utils.get_roi2trace(traces.copy(), times.copy(), np.array([3, 7]))
    assert list(roi2trace) == [3, 7]
    assert_array_equal(roi2trace[7]['trace'], traces[:, 1])
    assert_array_equal(roi2trace[7]['trace_times'], times[:, 1])
    assert roi2trace[3]['trace_valid'] == 1


def test_get_roi2trace_subset_and_3d():
    traces = np.random.default_rng(0).normal(size=(10, 3, 2))
    roi2trace = roi_utils.get_roi2trace(traces, traces.copy(), np.array([1, 2]), roi_ids_subset=np.array([2]))
    assert list(roi2trace) == [2]
    assert roi2trace[2]['trace'].shape == (10, 3)


def test_get_roi2trace_more_ids_than_traces():
    traces = np.arange(10.)[:, None]
    with pytest.warns(UserWarning, match='does not match'):
        roi2trace = roi_utils.get_roi2trace(traces, traces.copy(), np.array([1, 2]))
    assert roi2trace[1]['trace_valid'] == 1
    assert roi2trace[2]['trace_valid'] == 0
    assert roi2trace[2]['trace'].size == 0


def test_get_roi2trace_nan_inside_is_invalid():
    traces = np.arange(20.).reshape(10, 2)
    traces[5, 0] = np.nan
    with pytest.warns(UserWarning, match='NaN'):
        roi2trace = roi_utils.get_roi2trace(traces, traces.copy() * 0 + 1, np.array([1, 2]))
    assert roi2trace[1]['trace_valid'] == 0
    assert roi2trace[2]['trace_valid'] == 1


def test_get_roi2trace_nan_at_edges_is_fixed():
    traces = np.arange(20.).reshape(10, 2)
    traces[0, 0] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        roi2trace = roi_utils.get_roi2trace(traces, np.ones_like(traces), np.array([1, 2]))
    assert roi2trace[1]['trace_valid'] == 1


def test_get_roi2trace_input_validation():
    traces = np.zeros((5, 2))
    with pytest.raises(AssertionError):
        roi_utils.get_roi2trace(traces, traces, np.array([0, 1]))
    with pytest.raises(AssertionError):
        roi_utils.get_roi2trace(traces, traces, np.array([-1, -2]))
    with pytest.raises(ValueError, match='shape'):
        roi_utils.get_roi2trace(traces, np.zeros((5, 3)), np.array([1, 2]))
