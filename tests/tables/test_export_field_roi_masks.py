import os
from unittest import mock

import numpy as np
import pytest

from djimaging.tables.core import roi_mask as roi_mask_module
from djimaging.tables.core.roi_mask import RoiMaskTemplate
from djimaging.utils.mask_format_utils import to_igor_format
from djimaging.utils.mask_utils import load_roi_mask_file, save_roi_mask_file


class _Rows:
    def __init__(self, rows, primary_key):
        self._rows, self._primary_key = rows, primary_key

    def proj(self):
        return self

    def to_dicts(self):
        return [{k: r[k] for k in self._primary_key} for r in self._rows]

    def fetch1(self):
        assert len(self._rows) == 1
        return self._rows[0]


class _PresentationTable:
    primary_key = ['field', 'pres']

    def __init__(self, n_pres):
        self._rows = [dict(field=1, pres=i) for i in range(n_pres)]

    def __and__(self, key):
        return _Rows([r for r in self._rows if r['field'] == key['field']], self.primary_key)


def _field_mask():
    mask = np.zeros((12, 12), dtype=int)
    mask[3:6, 3:6] = 1
    mask[7:10, 6:9] = 2
    return mask


class _FakeRoiMask:
    """Motion-mode RoiMask of one field whose field mask belongs to presentation 1."""
    _roi_file_format = 'numpy'
    _presentation_masks = False
    _selects_field_mask = True
    _filepath_store = 'reference'
    primary_key = ['field']
    _get_roi_file_format = RoiMaskTemplate._get_roi_file_format
    _save_roi_mask_files = RoiMaskTemplate._save_roi_mask_files
    export_field_roi_masks = RoiMaskTemplate.export_field_roi_masks
    _export_field_roi_mask = RoiMaskTemplate._export_field_roi_mask

    def __init__(self, tmp_path, n_pres=3):
        self.presentation_table = _PresentationTable(n_pres)
        self.row = dict(field=1, pres=1, roi_mask=to_igor_format(_field_mask()), roi_mask_file='Pre/orig.h5',
                        roi_mask_source='igor_h5', align_dx=1, align_dy=-1)
        self.files = [str(tmp_path / 'Prepared' / f'pres{i}_ROIs.npy') for i in range(n_pres)]
        self.updates = []

    def __and__(self, key):
        return _Rows([self.row], self.primary_key)

    def _presentation_roi_mask_files(self, pres_keys, roi_mask_dir, old_prefix, new_prefix):
        return [self.files[k['pres']] for k in pres_keys]

    def update1(self, row):
        self.updates.append(row)
        self.row = {**self.row, **row}


@pytest.fixture(autouse=True)
def _relative_paths():
    with mock.patch.object(roi_mask_module, 'relative_store_path', lambda path, store: os.path.basename(path)):
        yield


def test_writes_field_mask_to_every_presentation_and_links_reference_file(tmp_path):
    tab = _FakeRoiMask(tmp_path)
    result = tab.export_field_roi_masks(verbose=False)

    assert result['exported'] == [dict(field=1)] and result['linked'] == [dict(field=1)]
    for f in tab.files:
        assert np.array_equal(load_roi_mask_file(f).astype(int), _field_mask())
    assert tab.updates == [dict(field=1, roi_mask_file='pres1_ROIs.npy', roi_mask_source='numpy',
                                align_dx=0, align_dy=0)]


def test_files_only_without_database_update(tmp_path):
    tab = _FakeRoiMask(tmp_path)
    result = tab.export_field_roi_masks(update_database=False, verbose=False)
    assert result['exported'] == [dict(field=1)] and result['linked'] == []
    assert tab.updates == [] and all(os.path.isfile(f) for f in tab.files)


def test_existing_files_with_same_mask_count_as_exported(tmp_path):
    tab = _FakeRoiMask(tmp_path)
    tab.export_field_roi_masks(update_database=False, verbose=False)
    mtimes = [os.path.getmtime(f) for f in tab.files]
    result = tab.export_field_roi_masks(verbose=False)
    assert result['exported'] == [] and result['skipped'] == [] and result['linked'] == [dict(field=1)]
    assert [os.path.getmtime(f) for f in tab.files] == mtimes


def test_files_with_different_mask_are_kept_unless_overwritten(tmp_path):
    tab = _FakeRoiMask(tmp_path)
    edited = _field_mask()
    edited[edited == 2] = 0  # e.g. reviewed and edited by hand
    os.makedirs(os.path.dirname(tab.files[0]))
    save_roi_mask_file(tab.files[0], edited, 'numpy')

    result = tab.export_field_roi_masks(verbose=False)
    assert len(result['skipped']) == 1 and result['linked'] == [] and tab.updates == []
    assert np.array_equal(load_roi_mask_file(tab.files[0]).astype(int), edited)

    result = tab.export_field_roi_masks(overwrite_files=True, verbose=False)
    assert result['exported'] == [dict(field=1)]
    assert np.array_equal(load_roi_mask_file(tab.files[0]).astype(int), _field_mask())


def test_only_in_motion_mode(tmp_path):
    tab = _FakeRoiMask(tmp_path)
    tab._presentation_masks = True
    with pytest.raises(NotImplementedError):
        tab.export_field_roi_masks(verbose=False)
