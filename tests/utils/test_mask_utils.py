from djimaging.utils.mask_utils import to_roi_mask_file


def test_to_roi_mask_file_unchanged():
    data_file = '/Users/someone/Data/Pre/recording.h5'
    roi_mask_file = to_roi_mask_file(
        data_file, old_suffix=None, new_suffix='_ROIs.pkl',
        roi_mask_dir=None, old_prefix=None, new_prefix=None)

    assert roi_mask_file == '/Users/someone/Data/Pre/recording_ROIs.pkl'


def test_to_roi_mask_file_change_dir():
    data_file = '/Users/someone/Data/Pre/recording.h5'
    roi_mask_file = to_roi_mask_file(
        data_file, old_suffix=None, new_suffix='_ROIs.pkl',
        roi_mask_dir='RoiMask', old_prefix=None, new_prefix=None)

    assert roi_mask_file == '/Users/someone/Data/RoiMask/recording_ROIs.pkl'


def test_to_roi_mask_file_rm_prefix():
    data_file = '/Users/someone/Data/Pre/SMP_recording.h5'
    roi_mask_file = to_roi_mask_file(
        data_file, old_suffix=None, new_suffix='_ROIs.pkl',
        roi_mask_dir='RoiMask', old_prefix='SMP_', new_prefix=None)

    assert roi_mask_file == '/Users/someone/Data/RoiMask/recording_ROIs.pkl'
