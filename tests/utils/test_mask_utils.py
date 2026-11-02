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
