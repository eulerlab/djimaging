from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from djimaging.tables.core import roi_mask as roi_mask_module
from djimaging.tables.core.roi_mask import RoiMaskTemplate


class _Rows:
    """Minimal stand-in for a restricted DataJoint table."""

    def __init__(self, n=0, **values):
        self._n, self._values = n, values

    def __len__(self):
        return self._n

    def fetch1(self, name):
        return self._values[name]


class _FakeRoiMask:
    """Only provides what `_resolve_field_source` touches."""
    _filepath_store = "reference"
    _get_roi_file_format = RoiMaskTemplate._get_roi_file_format
    _roi_file_format = 'numpy'

    def __init__(self, n_existing=0, existing_source=None):
        self._rows = _Rows(n_existing, roi_mask_source=existing_source)
        self.presentation_table = mock.MagicMock()

    def __and__(self, other):
        return self._rows

    def __len__(self):
        return len(self._rows)


def _resolve(fake, tmp_path, pres_files):
    pres_keys = [dict(i=i) for i in range(len(pres_files))]
    fake.presentation_table.__and__.side_effect = lambda key: _Rows(1, pres_data_file=str(pres_files[key['i']]))
    with mock.patch.object(roi_mask_module, 'local_file_path', side_effect=lambda v, store: Path(v)):
        return RoiMaskTemplate._resolve_field_source(
            fake, dict(field='f'), pres_keys, roi_mask_dir='AutoROIs')


@pytest.fixture
def pres_files(tmp_path):
    files = [tmp_path / 'Pre' / f'SMP_{i}.h5' for i in range(2)]
    (tmp_path / 'Pre').mkdir()
    return files


def test_either_uses_igor_without_mask_file(tmp_path, pres_files):
    assert _resolve(_FakeRoiMask(), tmp_path, pres_files) == 'igor'


def test_either_prefers_mask_file(tmp_path, pres_files):
    mask_dir = tmp_path / 'Pre' / 'AutoROIs'
    mask_dir.mkdir()
    np.save(mask_dir / 'SMP_1_ROIs.npy', np.zeros((2, 2)))
    with mock.patch.object(roi_mask_module, 'to_roi_mask_file',
                           side_effect=lambda f, **kw: str(mask_dir / (Path(f).stem + '_ROIs.npy'))):
        assert _resolve(_FakeRoiMask(), tmp_path, pres_files) == 'autorois'


@pytest.mark.parametrize('stored,expected', [('igor_h5', 'igor'), ('numpy', 'autorois'), ('pickle', 'autorois')])
def test_either_keeps_source_of_existing_field_entry(tmp_path, pres_files, stored, expected):
    assert _resolve(_FakeRoiMask(1, stored), tmp_path, pres_files) == expected
