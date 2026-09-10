import datetime
from types import SimpleNamespace
from unittest.mock import Mock

import datajoint as dj
import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from djimaging.tables.classifier.celltype_assignment import CelltypeAssignmentTemplate
from djimaging.tables.classifier_v2 import celltype_assignment_v2
from djimaging.tables.core.stim_logs import PresentationLogTemplate
from djimaging.tables.motion_correction.motion_detection import MotionDetectionTemplate
from djimaging.tables.receptivefield.fast_sta import FastStaTemplate
from djimaging.tables.receptivefield.glm import _get_shared_model_shift


class _Relation:
    def __init__(self, rows, primary_key=()):
        self.rows = list(rows)
        self.primary_key = tuple(primary_key)

    def __call__(self):
        return self

    def __len__(self):
        return len(self.rows)

    def __and__(self, restriction):
        if isinstance(restriction, dj.AndList):
            relation = self
            for condition in restriction:
                relation = relation & condition
            return relation
        elif isinstance(restriction, dict):
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


@pytest.mark.parametrize('table_class', [
    celltype_assignment_v2.CelltypeAssignmentV2Template,
    CelltypeAssignmentTemplate,
    MotionDetectionTemplate,
    FastStaTemplate,
])
@pytest.mark.parametrize('reserve_jobs', [False, True])
def test_custom_populate_calls_datajoint_v2(table_class, reserve_jobs, monkeypatch):
    train_x = np.zeros((75, 4))
    train_y = np.arange(1, 76)
    classifier = DummyClassifier().fit(train_x, train_y)
    classifier_dict = {
        'classifier': classifier,
        'chirp_feats': np.ones((3, 1)),
        'bar_feats': np.ones((2, 1)),
        'feature_names': ['chirp', 'bar', 'ds', 'size'],
        'train_x': train_x,
        'train_y': train_y,
        'y_names': {label: str(label) for label in train_y},
    }
    monkeypatch.setattr(celltype_assignment_v2, 'load_classifier_from_file',
                        Mock(return_value=classifier_dict))
    rows = [{'model_id': 1, 'variant': 1, 'classifier_file': 'classifier.pkl'}]
    restrictions = ()
    if reserve_jobs:
        # Both conditions must hold when selecting the model/parameter set.
        rows += [dict(rows[0], model_id=2), dict(rows[0], variant=2)]
        restrictions = ({'model_id': 1}, {'variant': 1})

    params_table = _Relation([
        dict(row, x_stimulus=np.ones((4, 2)), dt=0.1, rf_time=np.arange(2),
             burn_in=1, shift=0, fit_kind='trace', dims=(2,),
             fupsample_stim=1, fupsample_trace=1, lowpass_cutoff=0,
             pre_blur_sigma_s=0, post_blur_sigma_s=0)
        for row in rows
    ])
    params_table.stimulus_table = _Relation([
        dict(row, stim_dict={'nframes_per_trigger': 2}) for row in rows
    ])
    summary = {'success_count': 1, 'error_list': []}
    worker = Mock(return_value=summary)
    table = type('TestPopulate', (table_class,), {
        'connection': SimpleNamespace(in_transaction=False),
        'classifier_table': _Relation(rows),
        'params_table': params_table,
        '_populate_direct': worker,
        '_populate_distributed': worker,
    })()
    make_kwargs = {}
    # Execute the real DataJoint populate() entry point; only its workers are stubbed.
    result = table.populate(
        *restrictions, reserve_jobs=reserve_jobs, make_kwargs=make_kwargs,
        max_calls=2, display_progress=True, priority=3, refresh=False,
    )

    assert result is summary
    worker.assert_called_once()
    assert worker.call_args.args == restrictions
    forwarded = worker.call_args.kwargs
    assert forwarded['max_calls'] == 2
    assert forwarded['display_progress'] is True
    assert forwarded['processes'] == 1
    if reserve_jobs:
        assert forwarded['priority'] == 3
        assert forwarded['refresh'] is False
    if table_class is celltype_assignment_v2.CelltypeAssignmentV2Template:
        assert forwarded['make_kwargs']['classifier'] is classifier
        assert forwarded['make_kwargs']['chirp_feats'] is classifier_dict['chirp_feats']
        assert forwarded['make_kwargs']['bar_feats'] is classifier_dict['bar_feats']
    elif table_class is FastStaTemplate:
        sta_params = forwarded['make_kwargs']['sta_params']
        assert sta_params['dt'] == 0.1
        assert sta_params['nframes_per_trigger'] == 2
        np.testing.assert_array_equal(sta_params['rf_time'], np.arange(2))
        np.testing.assert_array_equal(sta_params['x_stimulus'], np.ones((4, 2)))
        assert sta_params['x_stimulus'].dtype == np.float32
    assert make_kwargs == {}


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
