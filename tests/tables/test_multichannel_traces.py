from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from djimaging.tables.core import traces


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
        _use_channels=channels is not None, _include_motion_correction=False,
        channel_table=Relation([dict(ch_name=n, ch_id=i) for n, i in channels or []]),
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


def test_multichannel_requires_compute_from_stack():
    table = SimpleNamespace(
        _use_channels=True, _include_motion_correction=False,
        raw_params_table=Relation([dict(include_artifacts=False, compute_from_stack=False,
                                       trace_precision='line', from_raw_data=False)]))
    with pytest.raises(ValueError, match='compute_from_stack'):
        traces.TracesTemplate.make(table, {'presentation_id': 1})


def test_channels_are_off_by_default():
    assert traces.TracesTemplate._use_channels is False
    table = type('SingleChannel', (traces.TracesTemplate,), {})()
    assert 'ch_id' not in table.definition and 'channel_table' not in table.definition


def test_channel_definitions_use_v2_types():
    table = type('MultiChannel', (traces.TracesTemplate,), {'_use_channels': True})()
    assert '-> self.channel_table' in table.definition


def test_features_channels_off_by_default_and_per_channel_when_on():
    from djimaging.tables.clustering.decomposition import FeaturesTemplate
    assert FeaturesTemplate._use_channels is False
    table = type('F', (FeaturesTemplate,), {})()
    assert 'channel_table' not in table.definition
    table = type('F', (FeaturesTemplate,), {'_use_channels': True})()
    assert '-> self.channel_table' in table.definition
