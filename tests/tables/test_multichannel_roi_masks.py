from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from djimaging.tables.core import traces, roi_mask


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

    def to_arrays(self, attr):
        return np.array([row[attr] for row in self.rows])

    def keys(self):
        return [{k: row[k] for k in self.primary_key} for row in self.rows]

    def insert1(self, row, **kwargs):
        self.rows.append(row)


@pytest.mark.parametrize('channels', [None, [('wDataCh0', 0), ('wDataCh1', 1)]])
def test_channel_extraction_uses_live_staged_path(channels, monkeypatch, tmp_path):
    staged = tmp_path / 'recording.h5'
    active = False

    @contextmanager
    def local_path(value, store):
        nonlocal active
        assert (value, store) == ('recording.h5', 'reference')
        active = True
        yield staged
        active = False

    calls = []

    def compute(key, filepath, roi_ids, precision, raw, data_stack_name, verboselvl):
        assert active and filepath == staged
        calls.append(data_stack_name)
        return {1: {'trace': np.array([1., 2., 3.]), 'trace_times': np.array([0., 1., 2.]),
                    'trace_valid': True}}, 1.

    monkeypatch.setattr(traces, 'local_path', local_path)
    presentation = Relation([{'pres_data_file': 'recording.h5', 'triggertimes': np.array([1.])}])
    presentation._filepath_store = 'reference'
    inserted = []
    table = SimpleNamespace(
        _channels_name_id=channels, _include_motion_correction=False,
        raw_params_table=Relation([dict(include_artifacts=False, compute_from_stack=True,
                                       trace_precision='line', from_raw_data=False)]),
        presentation_table=presentation, roi_table=Relation([{'roi_id': 1}]),
        userinfo_table=Relation([{'data_stack_name': 'wDataCh0'}]),
        _compute_roi2trace_from_stack=compute, insert1=inserted.append,
    )
    traces.TracesTemplate.make(table, {'presentation_id': 1})
    assert not active
    assert calls == (['wDataCh0'] if channels is None else ['wDataCh0', 'wDataCh1'])
    assert [row.get('ch_id') for row in inserted] == ([None] if channels is None else [0, 1])
    assert all(row['trigger_valid'] for row in inserted)


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


def test_optional_definitions_use_v2_types_and_resolved_store():
    table = type('MultiChannel', (traces.TracesTemplate,), {'_channels_name_id': [('wDataCh0', 0)]})()
    assert 'ch_id : uint8' in table.definition
    table = type('MultiDirectory', (roi_mask.RoiMaskTemplate,), {'_include_roi_mask_dir': True})()
    assert 'roi_mask_dir : varchar(31)' in table.definition
    assert '<filepath@reference>' in table.definition
    assert '{self.' not in table.definition
