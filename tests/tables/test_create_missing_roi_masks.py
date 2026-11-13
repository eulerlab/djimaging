import numpy as np
import pytest

from djimaging.tables.core.roi_mask import RoiMaskTemplate
from djimaging.utils import mask_utils


class _FakeModel:
    def __init__(self):
        self.calls = []

    def create_mask_from_data(self, ch0_stack, ch1_stack=None, n_artifact=0, pixel_size_um=(1., 1.),
                              multiple_stacks=False):
        self.calls.append(dict(multiple_stacks=multiple_stacks, ch1=ch1_stack is not None))
        mask = np.zeros((20, 20), dtype=int)
        mask[5:9, 5:9] = 1
        mask[12:14, 12:15] = 2
        mask[0:2, 0:2] = 3  # in light artifact rows, removed by cleaning
        mask[16, 16] = 4  # smaller than min_size
        return mask


class _FakeRoiMask:
    _roi_file_format = 'numpy'
    _get_roi_file_format = RoiMaskTemplate._get_roi_file_format
    _save_roi_mask_files = RoiMaskTemplate._save_roi_mask_files

    def __init__(self, tmp_path, n_pres=3, two_channels=False):
        rng = np.random.default_rng(0)
        base = rng.random((20, 20, 30))
        self.output_files = [str(tmp_path / 'ROIs' / f'pres{i}.npy') for i in range(n_pres)]
        # presentation i is the base stack moved by i pixels in x
        stacks = [np.roll(base, i, axis=0) for i in range(n_pres)]
        self.data = dict(
            pres_keys=[dict(i=i) for i in range(n_pres)], pres_names=[dict(i=i) for i in range(n_pres)],
            stacks={'ch0': stacks, **({'ch1': stacks} if two_channels else {})},
            output_files=self.output_files, ch_names=['ch0', 'ch1'] if two_channels else ['ch0'],
            roi_channel='ch0', n_artifact=2, pixel_size_d1_d2=(1., 1.))

    def _load_field_stacks(self, field_key, **kwargs):
        return self.data


def _run(fake, model, **kwargs):
    params = dict(roi_channel=None, roi_mask_dir='ROIs', old_prefix=None, new_prefix=None, use_stim_onset=True,
                  use_all_stacks=False, clean=True, min_size=3, connectivity=2, auto_shift=False, max_shift=5,
                  overwrite_files=False)
    params.update(kwargs)
    return RoiMaskTemplate._create_field_roi_mask_files(fake, dict(field=1), model, **params)


def test_writes_cleaned_mask_for_every_presentation(tmp_path):
    fake, model = _FakeRoiMask(tmp_path), _FakeModel()
    assert _run(fake, model) is True
    assert model.calls == [dict(multiple_stacks=False, ch1=False)]
    for f in fake.output_files:
        mask = mask_utils.load_roi_mask_file(f)
        assert set(np.unique(mask)) == {0, 1, 2}  # artifact and tiny ROI removed
    np.testing.assert_array_equal(mask_utils.load_roi_mask_file(fake.output_files[0]),
                                  mask_utils.load_roi_mask_file(fake.output_files[1]))


def test_without_cleaning_keeps_all_rois(tmp_path):
    fake = _FakeRoiMask(tmp_path)
    _run(fake, _FakeModel(), clean=False)
    assert len(np.unique(mask_utils.load_roi_mask_file(fake.output_files[0]))) == 5


def test_existing_files_skipped_unless_overwritten(tmp_path):
    fake, model = _FakeRoiMask(tmp_path), _FakeModel()
    assert _run(fake, model) is True
    assert _run(fake, model) is False
    assert len(model.calls) == 1
    assert _run(fake, model, overwrite_files=True) is True
    assert len(model.calls) == 2


def test_auto_shift_moves_mask_with_the_recording(tmp_path):
    fake = _FakeRoiMask(tmp_path)
    _run(fake, _FakeModel(), auto_shift=True)
    masks = [mask_utils.load_roi_mask_file(f) for f in fake.output_files]
    # The recording of presentation i is the first one moved by i pixels, so is the mask file (either sign).
    centroids = [np.argwhere(m == 1).mean(axis=0)[0] for m in masks]
    assert abs(centroids[1] - centroids[0]) == pytest.approx(1, abs=0.01)
    assert abs(centroids[2] - centroids[0]) == pytest.approx(2, abs=0.01)


def test_all_stacks_and_second_channel_are_passed_to_model(tmp_path):
    fake, model = _FakeRoiMask(tmp_path, two_channels=True), _FakeModel()
    _run(fake, model, use_all_stacks=True)
    assert model.calls == [dict(multiple_stacks=True, ch1=True)]


def test_missing_root_directory_is_not_created(tmp_path):
    fake = _FakeRoiMask(tmp_path)
    fake.data['output_files'] = [str(tmp_path / 'does_not_exist' / 'ROIs' / 'a.npy')] * 3
    with pytest.raises(FileNotFoundError):
        _run(fake, _FakeModel())
