import datetime
from types import SimpleNamespace

import pytest

from djimaging.tables.core.stim_logs import PresentationLogTemplate
from djimaging.tables.receptivefield.glm import _get_shared_model_shift


class _Relation:
    def __init__(self, rows, primary_key=()):
        self.rows = list(rows)
        self.primary_key = tuple(primary_key)

    def __and__(self, restriction):
        if isinstance(restriction, dict):
            rows = [
                row for row in self.rows
                if all(row.get(name) == value for name, value in restriction.items() if name in row)
            ]
        elif restriction == 'aborted = 0':
            rows = [row for row in self.rows if not row['aborted']]
        else:
            raise NotImplementedError(restriction)
        return _Relation(rows, primary_key=self.primary_key)

    def fetch1(self, *attrs):
        if len(self.rows) != 1:
            raise RuntimeError(f"Expected one row, found {len(self.rows)}")
        row = self.rows[0]
        if attrs == ('KEY',):
            return {name: row[name] for name in self.primary_key}
        values = tuple(row[name] for name in attrs)
        return values[0] if len(values) == 1 else values

    def to_dicts(self, order_by=None):
        rows = [row.copy() for row in self.rows]
        if order_by:
            rows.sort(key=lambda row: tuple(row[name] for name in order_by))
        return rows


def test_get_shared_model_shift_supports_multiple_channels():
    assert _get_shared_model_shift({
        'shift': {'ch0': -4, 'ch1': -4},
        'channel_names': ['ch0', 'ch1'],
    }) == -4


def test_get_shared_model_shift_rejects_different_channel_shifts():
    with pytest.raises(ValueError, match="different shifts"):
        _get_shared_model_shift({
            'shift': {'ch0': -4, 'ch1': -3},
            'channel_names': ['ch0', 'ch1'],
        })


def test_presentation_log_uses_configured_scan_params_table():
    inserted = []
    key = {'experiment_id': 1, 'stim_name': 'noise', 'presentation_id': 2}
    log_rows = [
        {
            'experiment_id': 1,
            'log_idx': 0,
            'stim_idx': 0,
            'stim_md5': 'abc',
            'stim_file_name': 'noise.py',
            'aborted': False,
            't_start': datetime.time(10, 0),
            't_end': datetime.time(10, 1),
        },
        {
            'experiment_id': 1,
            'log_idx': 0,
            'stim_idx': 1,
            'stim_md5': 'abc',
            'stim_file_name': 'noise.py',
            'aborted': False,
            't_start': datetime.time(11, 0),
            't_end': datetime.time(11, 1),
        },
    ]
    table = SimpleNamespace(
        exp_table=_Relation([{'experiment_id': 1}], primary_key=('experiment_id',)),
        stimulus_table=_Relation([{
            'stim_name': 'noise',
            'stim_hash': 'abc',
            'stim_dict': {},
        }]),
        log_table=_Relation(log_rows),
        scan_params_table=_Relation([{
            **key,
            'scan_params_dict': {
                'datestamp_y_m_d': '2026-08-05',
                'timestamp_h_m_s_ms': '11-00-05-000',
            },
        }]),
        presentation_table=object(),
        insert1=inserted.append,
    )

    PresentationLogTemplate.make(table, key)

    assert inserted == [{
        **key,
        'log_idx': 0,
        'stim_idx': 1,
        'match_method': 'md5+time',
        't_start': datetime.time(11, 0),
        't_end': datetime.time(11, 1),
        'aborted': False,
    }]
