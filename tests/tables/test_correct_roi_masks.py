import os
from unittest import mock

import numpy as np

from djimaging.tables.core import roi_mask as roi_mask_module
from djimaging.tables.core.roi_mask import RoiMaskTemplate
from djimaging.utils import mask_utils
from djimaging.utils.mask_format_utils import to_igor_format


class _Rows:
    def __init__(self, n=0, **values):
        self._n, self._values = n, values

    def __len__(self):
        return self._n

    def __and__(self, other):
        return self

    def proj(self):
        return self

    def fetch1(self):
        return self._values


class _Table:
    """Table stand-in that returns the same rows for every restriction."""
    primary_key = ['pres']

    def __init__(self, rows=None):
        self._rows = rows if rows is not None else _Rows()

    def __and__(self, other):
        return self._rows

    def proj(self):
        return self


class _FakeModel:
    """Finds the two original cells and one new cell."""

    def create_mask_from_data(self, ch0_stack, ch1_stack=None, n_artifact=0, pixel_size_um=(1., 1.),
                              multiple_stacks=False):
        mask = np.zeros((20, 20), dtype=int)
        mask[5:9, 5:9] = 1
        mask[12:15, 12:15] = 2
        mask[14:17, 3:6] = 3  # new cell
        return mask


def _original_mask():
    mask = np.zeros((20, 20), dtype=int)
    mask[5:9, 5:9] = 1
    mask[12:15, 12:15] = 2
    mask[0:2, 0:2] = 3  # in the light artifact rows, to be removed
    return mask


class _FakeRoiMask:
    _roi_file_format = 'numpy'
    _get_roi_file_format = RoiMaskTemplate._get_roi_file_format
    _save_roi_mask_files = RoiMaskTemplate._save_roi_mask_files
    init_shifts = None

    def __init__(self, tmp_path, roi_mask, n_pres=2, different=False):
        self.output_files = [str(tmp_path / 'Corrected' / f'pres{i}.npy') for i in range(n_pres)]
        self.field_table = _Table(_Rows(1, field=1))
        self.RoiMaskPresentation = _Table(_Rows(int(different)))
        self.presentation_table = _Table()
        self._master = dict(pres=0, roi_mask=to_igor_format(roi_mask))
        self._shifts = [(0, 0), (1, 0)][:n_pres]
        self.data = dict(
            pres_keys=[dict(pres=i) for i in range(n_pres)], output_files=self.output_files,
            stacks={'ch0': [np.zeros((20, 20, 3))] * n_pres}, ch_names=['ch0'], roi_channel='ch0',
            n_artifact=2, pixel_size_d1_d2=(1., 1.))

    def __and__(self, other):
        return _Rows(1, **self._master)

    def _load_field_stacks(self, field_key, **kwargs):
        return self.data

    def init_shifts(self, field_key, pres_keys):
        return self._shifts


def _run(fake, model, **kwargs):
    params = dict(roi_channel=None, roi_mask_dir='Corrected', old_prefix=None, new_prefix=None, use_stim_onset=True,
                  clean=True, min_size=3, connectivity=2, iou_threshold=0.1, overwrite_files=False)
    params.update(kwargs)
    with mock.patch.object(roi_mask_module, 'load_array', side_effect=lambda x: x):
        return RoiMaskTemplate._correct_field_roi_mask(fake, dict(field=1), model, **params)


def test_cleans_and_adds_missing_roi_and_shifts_files(tmp_path):
    fake = _FakeRoiMask(tmp_path, _original_mask())
    info = _run(fake, _FakeModel())
    assert info['status'] == 'corrected'
    assert (info['n_before'], info['n_after'], info['n_added']) == (3, 3, 1)

    ref = mask_utils.load_roi_mask_file(fake.output_files[0])
    assert not np.any(ref[:2] > 0)  # artifact ROI removed
    assert np.all(ref[14:17, 3:6] > 0)  # new ROI added
    assert len(np.unique(ref[ref > 0])) == 3

    # The mask of the second presentation is the inverse-shifted field mask.
    shifted = mask_utils.load_roi_mask_file(fake.output_files[1])
    np.testing.assert_array_equal(shifted > 0, mask_utils.shift_array(ref, shift=(-1, 0), cval=0) > 0)


def test_unchanged_mask_writes_nothing(tmp_path):
    mask = np.zeros((20, 20), dtype=int)
    mask[5:9, 5:9] = 1
    mask[12:15, 12:15] = 2
    fake = _FakeRoiMask(tmp_path, mask)

    info = _run(fake, None)  # without a model nothing is added and the mask is already clean
    assert info['status'] == 'unchanged'
    assert not any(os.path.isfile(f) for f in fake.output_files)


def test_existing_corrected_files_are_skipped_unless_overwritten(tmp_path):
    fake = _FakeRoiMask(tmp_path, _original_mask())
    assert _run(fake, _FakeModel())['status'] == 'corrected'
    assert _run(fake, _FakeModel())['status'] == 'skipped'
    assert _run(fake, _FakeModel(), overwrite_files=True)['status'] == 'corrected'


def test_fields_with_different_presentation_masks_are_skipped(tmp_path):
    fake = _FakeRoiMask(tmp_path, _original_mask(), different=True)
    info = _run(fake, _FakeModel())
    assert info['status'] == 'skipped' and 'different' in info['reason']


def test_bright_alt_channel_rois_are_removed(tmp_path):
    fake = _FakeRoiMask(tmp_path, _original_mask())
    alt = np.zeros((20, 20, 3))
    alt[12:17, :, :] = 10.  # vessel-like bright band over the second cell and the new cell
    fake.data['stacks']['ch1'] = [alt] * 2
    fake.data['ch_names'] = ['ch0', 'ch1']

    info = _run(fake, _FakeModel(), max_alt_intensity=0.5)
    assert info['status'] == 'corrected'
    ref = mask_utils.load_roi_mask_file(fake.output_files[0])
    assert np.all(ref[5:9, 5:9] > 0)
    assert not np.any(ref[12:17, :] > 0)
    assert info['n_added'] == 0


def test_bright_alt_channel_needs_second_channel(tmp_path):
    fake = _FakeRoiMask(tmp_path, _original_mask())
    try:
        _run(fake, _FakeModel(), max_alt_intensity=0.5)
    except ValueError as e:
        assert 'second channel' in str(e)
    else:
        raise AssertionError('expected ValueError')


def test_light_artifact_does_not_count_for_bright_alt_channel_rois():
    mask = np.zeros((20, 20), dtype=int)
    mask[0:6, 5:9] = 1  # mostly in the artifact rows, dim below them
    alt = np.zeros((20, 20))
    alt[:2] = 100.  # light artifact
    alt[10:14] = 5.
    out = mask_utils.remove_bright_alt_channel_rois(mask, alt, n_artifact=2, max_rel_intensity=0.5, verbose=False)
    assert np.any(out > 0)
