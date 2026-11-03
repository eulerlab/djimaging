from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from djimaging.tables.core import roi_mask


class Relation:
    def __init__(self, rows, primary_key=()):
        self.rows = rows
        self.primary_key = primary_key

    def __and__(self, key):
        return Relation([row for row in self.rows
                         if all(row[k] == v for k, v in key.items() if k in row)], self.primary_key)

    def __len__(self):
        return len(self.rows)

    def __call__(self):
        return self

    def fetch1(self, *attrs):
        assert len(self.rows) == 1
        values = tuple(self.rows[0][attr] for attr in attrs)
        return values[0] if len(values) == 1 else values

    def to_dicts(self):
        return list(self.rows)

    def to_arrays(self, attr):
        return np.array([row[attr] for row in self.rows])

    def keys(self):
        return [{k: row[k] for k in self.primary_key} for row in self.rows]

    def insert1(self, row, **kwargs):
        self.rows.append(row)


def test_roi_lookup_scopes_directory():
    first, second = np.array([[-1, 0]]), np.array([[-2, 0]])
    table = Relation([dict(field_id=1, roi_mask_dir='ROIs', roi_mask=first),
                      dict(field_id=1, roi_mask_dir='AutoROIs', roi_mask=second)],
                     primary_key=('field_id', 'roi_mask_dir'))
    table._include_roi_mask_dir = True
    np.testing.assert_array_equal(
        roi_mask.RoiMaskTemplate.load_field_roi_mask_database(table, {'field_id': 1}, 'AutoROIs'), second)


def test_roi_adds_new_directory_even_when_presentation_exists(monkeypatch):
    mask = np.array([[-1, 0]])
    table = Relation([dict(field_id=1, roi_mask_dir='ROIs')], primary_key=('field_id', 'roi_mask_dir'))
    table._include_roi_mask_dir = True
    table._max_shift = 5
    table.presentation_table = Relation([dict(field_id=1, presentation_id=1, pres_data_file='recording.h5')],
                                        primary_key=('field_id', 'presentation_id'))
    table.presentation_table._filepath_store = 'reference'
    table.RoiMaskPresentation = Relation([dict(field_id=1, presentation_id=1, roi_mask_dir='ROIs')])
    table.userinfo_table = Relation([dict(mask_alias='chirp', highres_alias='hr')])
    table._load_presentation_roi_mask = Mock(return_value=(mask, 'AutoROIs/mask.npy', 'numpy'))
    monkeypatch.setattr(roi_mask, 'local_file_path', lambda value, store: value)
    monkeypatch.setattr(roi_mask, 'sort_roi_mask_files', lambda **kwargs: [0])
    roi_mask.RoiMaskTemplate._add_field_roi_masks(table, {'field_id': 1}, roi_mask_dir='AutoROIs')
    assert len(table.rows) == len(table.RoiMaskPresentation.rows) == 2
    for row in (table.rows[-1], table.RoiMaskPresentation.rows[-1]):
        assert row['roi_mask_dir'] == 'AutoROIs'
        assert row['roi_mask_file'] == 'AutoROIs/mask.npy'
        assert row['roi_mask_source'] == 'numpy'
    table._load_presentation_roi_mask.assert_called_once()


def test_roi_mask_dir_definition_uses_v2_types_and_resolved_store():
    table = type('MultiDirectory', (roi_mask.RoiMaskTemplate,), {'_include_roi_mask_dir': True})()
    assert 'roi_mask_dir : varchar(31)' in table.definition
    assert '<filepath@reference>' in table.definition
    assert '{self.' not in table.definition
