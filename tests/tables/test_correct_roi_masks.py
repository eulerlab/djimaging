import os
from unittest import mock

import numpy as np
import pytest

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
    _presentation_masks = True
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

    def _expected_corrected_files(self, *args):
        return self.output_files


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


def _vessel_stacks():
    """Main channel with a dark vertical vessel (cols 14-16), bright in the alt channel; cells elsewhere."""
    rng = np.random.default_rng(0)
    main = 5. + rng.normal(0, .1, (20, 20, 3))
    main[:, 14:17] = 0.
    alt = rng.normal(0, .1, (20, 20, 3))
    alt[:, 14:17] += 50.
    return main, alt


def test_vessel_rois_are_not_added_but_existing_are_kept_by_default(tmp_path):
    mask = np.zeros((20, 20), dtype=int)
    mask[5:9, 5:9] = 1
    mask[10:14, 14:17] = 2  # existing ROI on the vessel
    fake = _FakeRoiMask(tmp_path, mask)
    main, alt = _vessel_stacks()
    fake.data['stacks'] = {'ch0': [main] * 2, 'ch1': [alt] * 2}
    fake.data['ch_names'] = ['ch0', 'ch1']
    fake.data['n_artifact'] = 0

    class _Model:  # finds the existing cell, a new cell and a new ROI on the vessel
        def create_mask_from_data(self, *args, **kwargs):
            pred = np.zeros((20, 20), dtype=int)
            pred[5:9, 5:9] = 1
            pred[12:16, 2:6] = 2
            pred[2:6, 14:17] = 3
            return pred

    info = _run(fake, _Model(), vessel_threshold=10., clean=False)
    ref = mask_utils.load_roi_mask_file(fake.output_files[0])
    assert info['n_added'] == 1
    assert np.all(ref[12:16, 2:6] > 0)
    assert not np.any(ref[2:6, 14:17] > 0)  # new vessel ROI rejected
    assert np.all(ref[10:14, 14:17] > 0)  # existing vessel ROI kept

    info = _run(fake, _Model(), vessel_threshold=10., clean=False, vessel_filter_existing=True, overwrite_files=True)
    ref = mask_utils.load_roi_mask_file(fake.output_files[0])
    assert not np.any(ref[10:14, 14:17] > 0)


def test_vessel_threshold_is_ignored_without_second_channel(tmp_path):
    fake = _FakeRoiMask(tmp_path, _original_mask())
    with pytest.warns(UserWarning, match='second channel'):
        info = _run(fake, _FakeModel(), vessel_threshold=50.)
    assert info['status'] == 'corrected'


def test_no_vessels_in_field_without_bright_alt_structures():
    rng = np.random.default_rng(1)
    main = 5. + rng.normal(0, .1, (20, 20))
    alt = rng.normal(0, .1, (20, 20))
    assert not mask_utils.estimate_vessel_mask(alt, main).any()


def test_vessel_mask_ignores_light_artifact():
    main, alt = _vessel_stacks()
    main, alt = main.mean(-1), alt.mean(-1)
    alt[:2] = 500.
    vessel = mask_utils.estimate_vessel_mask(alt, main, n_artifact=2)
    assert not vessel[:2].any()
    assert vessel[10, 15]


def test_cells_bright_in_both_channels_are_not_vessels():
    rng = np.random.default_rng(2)
    main = 1. + rng.normal(0, .1, (30, 30))
    alt = rng.normal(0, .1, (30, 30))
    for r, c in [(5, 5), (15, 20), (22, 2)]:  # cells with bleed-through
        main[r:r + 4, c:c + 4] += 20.
        alt[r:r + 4, c:c + 4] += 8.
    alt[:, 12] += 15.  # thin vessel, not visible in the main channel
    vessel = mask_utils.estimate_vessel_mask(alt, main)
    assert vessel[:, 12].mean() > 0.8
    for r, c in [(5, 5), (15, 20), (22, 2)]:
        assert not vessel[r:r + 4, c:c + 4].any()


def test_shrink_is_off_by_default_and_uses_the_field_mask_presentation(tmp_path):
    mask = np.zeros((20, 20), dtype=int)
    mask[5:9, 5:9] = 1
    fake = _FakeRoiMask(tmp_path, mask)
    fake.data['stacks'] = {'ch0': [np.full((20, 20, 3), i, dtype=float) for i in range(2)]}
    calls = []

    def fake_shrink(roi_mask, stack, n_artifact=0, **kws):
        calls.append((stack[0, 0, 0], n_artifact, kws))
        shrunk = roi_mask.copy()
        shrunk[5, 5:9] = 0  # one edge removed
        return shrunk, {1: dict(status='shrunk', n_removed=4)}

    with mock.patch.object(roi_mask_module, 'shrink_rois_to_fit', side_effect=fake_shrink):
        assert _run(fake, None, clean=False)['status'] == 'unchanged' and calls == []
        info = _run(fake, None, clean=False, shrink=True, shrink_kws=dict(margin=0.5))
    assert info['status'] == 'corrected' and info['n_shrunk'] == 1 and info['n_added'] == 0
    assert calls == [(0., 2, dict(margin=0.5))]  # stack of the presentation of the field mask (pres 0)
    assert not np.any(mask_utils.load_roi_mask_file(fake.output_files[0])[5, 5:9])
