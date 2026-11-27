import os

import numpy as np

from djimaging.utils.mask_utils import (
    load_preferred_roi_mask_file, save_roi_mask_file, to_roi_mask_file,
)


def test_roi_mask_files_default_to_numpy(tmp_path):
    data_file = str(tmp_path / 'recording.h5')
    mask_file = to_roi_mask_file(data_file)
    assert mask_file == str(tmp_path / 'recording_ROIs.npy')
    mask = np.array([[0, 1], [0, 1]], dtype=np.int32)
    save_roi_mask_file(mask_file, mask, file_format='numpy')
    loaded, source = load_preferred_roi_mask_file([data_file])
    np.testing.assert_array_equal(loaded, [[1, -1], [1, -1]])
    assert source == mask_file


def test_to_roi_mask_file_unchanged():
    data_file = '/Users/someone/Data/Pre/recording.h5'
    roi_mask_file = to_roi_mask_file(
        data_file, old_suffix=None, new_suffix='_ROIs.pkl',
        roi_mask_dir=None, old_prefix=None, new_prefix=None)

    assert os.path.normpath(roi_mask_file) == os.path.normpath('/Users/someone/Data/Pre/recording_ROIs.pkl')


def test_to_roi_mask_file_change_dir():
    data_file = '/Users/someone/Data/Pre/recording.h5'
    roi_mask_file = to_roi_mask_file(
        data_file, old_suffix=None, new_suffix='_ROIs.pkl',
        roi_mask_dir='RoiMask', old_prefix=None, new_prefix=None)

    assert os.path.normpath(roi_mask_file) == os.path.normpath('/Users/someone/Data/RoiMask/recording_ROIs.pkl')


def test_to_roi_mask_file_rm_prefix():
    data_file = '/Users/someone/Data/Pre/SMP_recording.h5'
    roi_mask_file = to_roi_mask_file(
        data_file, old_suffix=None, new_suffix='_ROIs.pkl',
        roi_mask_dir='RoiMask', old_prefix='SMP_', new_prefix=None)

    assert os.path.normpath(roi_mask_file) == os.path.normpath('/Users/someone/Data/RoiMask/recording_ROIs.pkl')


def _cells_image_and_mask(seed=0, shape=(64, 64), n_cells=30, n_artifact=3):
    """Mean image with Gaussian cells, and an Igor-format mask with a 3x3 ROI on each cell."""
    rng = np.random.default_rng(seed)
    xx, yy = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    img = rng.normal(0, 0.05, shape) + 1.
    mask = np.ones(shape, dtype=int)
    centers = []
    while len(centers) < n_cells:
        c = rng.integers(n_artifact + 6, shape[0] - 6), rng.integers(6, shape[1] - 6)
        if all(abs(c[0] - o[0]) > 5 or abs(c[1] - o[1]) > 5 for o in centers):
            centers.append(c)
    for i, (cx, cy) in enumerate(centers):
        img += np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 1.2 ** 2))
        mask[cx - 1:cx + 2, cy - 1:cy + 2] = -(i + 1)
    return img, mask


def test_mask_fit_score_aligned_vs_misaligned():
    from djimaging.utils.mask_utils import mask_fit_score, shift_array
    img, mask = _cells_image_and_mask()
    aligned = mask_fit_score(mask, img, n_artifact=3)
    shifted = mask_fit_score(shift_array(mask, (3, 0), cval=1), img, n_artifact=3)
    assert aligned > 1
    assert shifted < 0.3 * aligned


def test_best_mask_offset_recovers_offset():
    from djimaging.utils.mask_utils import best_mask_offset, shift_array
    img, mask = _cells_image_and_mask()
    res = best_mask_offset(shift_array(mask, (-1, 2), cval=1), img, max_offset=2, n_artifact=3)
    assert res['offset'] == (1, -2)
    assert res['score'] > res['score0']
    # An aligned mask is not shifted
    assert best_mask_offset(mask, img, n_artifact=3)['offset'] == (0, 0)


def test_select_field_roi_mask():
    from djimaging.utils.mask_utils import select_field_roi_mask, shift_array
    img_a, mask = _cells_image_and_mask(seed=1)
    # b: same field moved by (2, 0); the mask was copied with a wrong shift (1, 0) -> off by one pixel
    img_b = shift_array(img_a, (2, 0), cval=np.median)
    mask_b = shift_array(mask, (1, 0), cval=1)
    # c: other cells (e.g. other z), mask copied without shift
    img_c, _ = _cells_image_and_mask(seed=2)

    # The preferred presentation (mask_alias) has a mask that does not fit its image
    res = select_field_roi_mask([mask_b, mask, mask], [img_b, img_a, img_c], preferred_idx=2, n_artifact=3)
    assert res['idx'] in (0, 1) and res['selection'] == 'fit'
    assert res['scores'][2] < 0.3 * res['scores'][1]
    assert np.isclose(res['scores'][0], res['scores'][1], rtol=0.05)  # b fits after alignment
    res = select_field_roi_mask([mask_b, mask, mask], [img_b, img_a, img_c], preferred_idx=1, n_artifact=3)
    assert res['idx'] == 1 and res['offset'] == (0, 0) and np.array_equal(res['roi_mask'], mask)

    # Only the misaligned copy: chosen and aligned to its image
    res = select_field_roi_mask([mask_b, mask], [img_b, img_c], preferred_idx=1, n_artifact=3)
    assert res['idx'] == 0 and res['offset'] == (1, 0)
    assert np.array_equal(res['roi_mask'] < 0, shift_array(mask, (2, 0), cval=1) < 0)

    # Equally good fits: the presentation with a Stack_ave wins, then the preferred one
    res = select_field_roi_mask([mask, mask], [img_a, img_a], preferred_idx=0, has_stack_ave=[False, True],
                                n_artifact=3)
    assert res['idx'] == 1
    res = select_field_roi_mask([mask, mask], [img_a, img_a], preferred_idx=1, n_artifact=3)
    assert res['idx'] == 1

    # A separately drawn mask is not a candidate; no fit at all falls back to mask_alias
    other = np.ones_like(mask)
    other[20:23, 20:23] = -1
    res = select_field_roi_mask([mask, other], [img_c, img_a], preferred_idx=0, n_artifact=3)
    assert res['n_different'] == 1 and np.isnan(res['scores'][1])
    res = select_field_roi_mask([mask], [np.ones_like(img_a)], preferred_idx=0, n_artifact=3)
    assert res['selection'] == 'alias' and res['idx'] == 0
