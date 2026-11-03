import numpy as np
import pytest

from djimaging.autorois import default_models
from djimaging.autorois.corr_roi_mask_multiscale import (
    CorrRoiMaskMultiscale, compute_corr_map_multiscale, find_seed_pixels)
from djimaging.autorois.corr_roi_mask_utils import get_roi_pixel_limits

CENTERS = [(10, 8), (10, 24), (28, 8), (28, 24)]


@pytest.fixture(scope='module')
def stack():
    """(nx, nz, nt) stack with four gaussian blobs that each have their own time course on top of noise."""
    rng = np.random.default_rng(0)
    nx, nz, nt = 40, 32, 200
    xx, zz = np.meshgrid(np.arange(nx), np.arange(nz), indexing='ij')
    stack = rng.normal(100, 5, (nx, nz, nt))
    for cx, cz in CENTERS:
        blob = np.exp(-((xx - cx) ** 2 + (zz - cz) ** 2) / (2 * 1.5 ** 2))
        stack += 40 * blob[:, :, None] * rng.normal(0, 1, nt)[None, None, :]
    return stack


def test_get_roi_pixel_limits():
    assert get_roi_pixel_limits(1, 16, (1., 1.)) == (2, 16, 5, 5)
    assert get_roi_pixel_limits(1, 16, (0.5, 0.5)) == (4, 64, 11, 11)
    assert get_roi_pixel_limits(1, 16, (1., 1.), n_pix_min=20, n_pix_max=10, n_pix_max_x=3, n_pix_max_z=4) == (
        20, 20, 3, 4)


def test_corr_map_multiscale(stack):
    corr_map = compute_corr_map_multiscale(stack)
    assert corr_map.shape == stack.shape[:2]
    cx, cz = CENTERS[0]
    assert corr_map[cx, cz] > np.percentile(corr_map, 90)
    assert compute_corr_map_multiscale(stack, clip=True).max() <= 1.


def test_find_seed_pixels_at_blobs(stack):
    corr_map = compute_corr_map_multiscale(stack)
    seed_ixs, seed_izs = find_seed_pixels(corr_map)
    for cx, cz in CENTERS:
        assert np.min(np.hypot(seed_ixs - cx, seed_izs - cz)) <= 3
    assert np.all(np.diff(corr_map[seed_ixs, seed_izs]) <= 0)


@pytest.mark.parametrize('kind', ['default_pr', 'default_pr_watershed'])
def test_default_pr_models_find_blobs(stack, kind):
    model = default_models.get_corr_roi_mask(kind)
    roi_mask = model.create_mask_from_data(stack, pixel_size_um=(1., 1.))

    assert roi_mask.shape == stack.shape[:2]
    roi_ids = np.unique(roi_mask[roi_mask > 0])
    assert np.array_equal(roi_ids, np.arange(1, len(roi_ids) + 1))
    for cx, cz in CENTERS:
        assert np.any(roi_mask[cx - 2:cx + 3, cz - 2:cz + 3] > 0)


def test_grow_respects_size_limits(stack):
    model = CorrRoiMaskMultiscale(n_pix_min=3, n_pix_max=9, n_pix_max_x=4, n_pix_max_z=4)
    roi_mask = model.create_mask_from_data(stack)
    assert roi_mask.max() > 0
    for roi_id in range(1, roi_mask.max() + 1):
        ixs, izs = np.nonzero(roi_mask == roi_id)
        assert 3 <= ixs.size <= 9
        assert np.ptp(ixs) < 4 and np.ptp(izs) < 4


def test_border_cut_is_empty(stack):
    model = CorrRoiMaskMultiscale(cut_x=(5, 5), cut_z=(5, 5))
    roi_mask = model.create_mask_from_data(stack, n_artifact=8)
    assert np.all(roi_mask[:8] == 0) and np.all(roi_mask[-5:] == 0)
    assert np.all(roi_mask[:, :5] == 0) and np.all(roi_mask[:, -5:] == 0)


def test_requires_ch1_if_not_ch0(stack):
    with pytest.raises(ValueError):
        CorrRoiMaskMultiscale(use_ch0_stack=False).create_mask_from_data(stack)
