"""Motion mode: one field ROI mask file per field (<stem>_FieldROIs.npy) instead of per-presentation copies."""
import os
from unittest import mock

import numpy as np
import pytest

from djimaging.tables.core import roi_mask as roi_mask_module
from djimaging.tables.core.roi_mask import RoiMaskTemplate
from djimaging.utils.mask_format_utils import to_igor_format
from djimaging.utils.mask_utils import save_roi_mask_file, to_field_roi_mask_file, to_roi_mask_file


def test_field_mask_file_is_not_a_presentation_mask_file():
    data_file = os.path.join('exp', 'Pre', 'SMP_M1_GCL0_chirp_C1.h5')
    field_file = to_field_roi_mask_file(data_file, roi_mask_dir='ROIs')
    assert field_file == os.path.join('exp', 'ROIs', 'SMP_M1_GCL0_chirp_C1_FieldROIs.npy')
    for file_format in ('numpy', 'pickle'):
        assert to_roi_mask_file(data_file, roi_mask_dir='ROIs', file_format=file_format) != field_file


def _mask():
    mask = np.zeros((16, 16), dtype=int)
    mask[4:7, 4:7] = 1
    mask[10:13, 9:12] = 2
    return mask


class _Rows:
    def __init__(self, row=None, n=0):
        self._row, self._n = row, n

    def __len__(self):
        return self._n

    def fetch1(self, *attrs):
        return tuple(self._row[a] for a in attrs) if attrs else self._row

    def keys(self):
        return self._row


class _PresentationTable:
    primary_key = ['field', 'stim']
    _filepath_store = 'reference'

    def __init__(self, pres_keys):
        self._pres_keys = pres_keys

    def __and__(self, key):
        if 'stim' in key:
            return _Rows(dict(pres_data_file=f"{key['stim']}.h5"), 1)
        return _Rows(self._pres_keys, len(self._pres_keys))


class _FakeRoiMask:
    _presentation_masks = False
    _selects_field_mask = True
    _filepath_store = 'reference'
    _find_field_roi_mask_file = RoiMaskTemplate._find_field_roi_mask_file
    _insert_field_roi_mask_file = RoiMaskTemplate._insert_field_roi_mask_file
    _add_field_roi_masks = RoiMaskTemplate._add_field_roi_masks

    def __init__(self, tmp_path):
        self.pres_keys = [dict(field=1, stim=s) for s in ('chirp', 'mb', 'ds')]
        self.presentation_table = _PresentationTable(self.pres_keys)
        self.files = [str(tmp_path / 'ROIs' / f'{s}_FieldROIs.npy') for s in ('chirp', 'mb', 'ds')]
        self.inserted = []
        image = np.zeros((16, 16))
        image[_mask() > 0] = 1.  # the cells are where the mask is
        self.image = image

    def __and__(self, key):
        return _Rows(n=0)  # field not in the database yet

    def _field_roi_mask_files(self, pres_keys, roi_mask_dir, old_prefix=None, new_prefix=None):
        return [self.files[[k['stim'] for k in self.pres_keys].index(pk['stim'])] for pk in pres_keys]

    def _get_stack_names(self, field_key):
        return ['wDataCh0'], 'wDataCh0'

    def _presentation_mean_image(self, pres_key, data_file, roi_channel):
        return self.image, False, 0

    def insert1(self, row, skip_duplicates=False):
        self.inserted.append(row)

    def write(self, idx, mask=None):
        os.makedirs(os.path.dirname(self.files[idx]), exist_ok=True)
        save_roi_mask_file(self.files[idx], _mask() if mask is None else mask, 'numpy')


@pytest.fixture(autouse=True)
def _paths():
    with mock.patch.object(roi_mask_module, 'relative_store_path', lambda path, store: os.path.basename(path)), \
            mock.patch.object(roi_mask_module, 'local_file_path', lambda value, store: value):
        yield


def test_find_field_mask_file(tmp_path):
    tab = _FakeRoiMask(tmp_path)
    assert tab._find_field_roi_mask_file(tab.pres_keys, 'ROIs') == (None, None)
    tab.write(2)
    assert tab._find_field_roi_mask_file(tab.pres_keys, 'ROIs') == (tab.pres_keys[2], tab.files[2])
    tab.write(0)
    with pytest.raises(ValueError, match='Several field ROI mask files'):
        tab._find_field_roi_mask_file(tab.pres_keys, 'ROIs')


def test_field_mask_file_names_presentation_without_choice_or_alignment(tmp_path):
    tab = _FakeRoiMask(tmp_path)
    tab.write(2)
    tab._add_field_roi_masks(dict(field=1), roi_mask_dir='ROIs')  # no raw params / Igor access needed

    assert len(tab.inserted) == 1
    row = tab.inserted[0]
    assert row['stim'] == 'ds' and row['field'] == 1
    assert row['mask_selection'] == 'file' and row['align_dx'] == 0 and row['align_dy'] == 0
    assert row['mask_fit'] > 0
    assert np.array_equal(row['roi_mask'], to_igor_format(_mask()))
    assert row['roi_mask_file'] == 'ds_FieldROIs.npy' and row['roi_mask_source'] == 'numpy'
