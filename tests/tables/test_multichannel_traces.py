from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from djimaging.tables.core import traces
from tests.autorois.test_roi_canvas_channels import FakeChannelTable


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


class FakeUserInfo:
    def __init__(self, functional):
        self.functional = functional

    def __call__(self):
        return self

    def get_functional_ch_names(self, key):
        return self.functional


def make_channel_table():
    """Two functional channels and one background channel (e.g. SR-101)."""
    channels = FakeChannelTable()
    channels.add_channels([('wDataCh0', 0), ('wDataCh1', 1), ('wDataCh2', 2)], functional=['wDataCh0', 'wDataCh1'])
    return channels


def make_traces_table(monkeypatch, tmp_path, use_channels, userinfo_functional, calls, inserted):
    staged = tmp_path / 'recording.h5'

    @contextmanager
    def local_path(value, store):
        assert (value, store) == ('recording.h5', 'reference')
        yield staged

    def compute(key, filepath, roi_ids, precision, raw, data_stack_name, verboselvl):
        assert filepath == staged
        calls.append(data_stack_name)
        return {1: {'trace': np.array([1., 2., 3.]), 'trace_times': np.array([0., 1., 2.]),
                    'trace_valid': True}}, 1.

    monkeypatch.setattr(traces, 'local_path', local_path)
    presentation = Relation([{'pres_data_file': 'recording.h5', 'triggertimes': np.array([1.])}])
    presentation._filepath_store = 'reference'
    return SimpleNamespace(
        _use_channels=use_channels, _include_motion_correction=False,
        channel_table=make_channel_table(),
        raw_params_table=Relation([dict(include_artifacts=False, compute_from_stack=True,
                                       trace_precision='line', from_raw_data=False)]),
        presentation_table=presentation, roi_table=Relation([{'roi_id': 1}]),
        userinfo_table=FakeUserInfo(userinfo_functional),
        _compute_roi2trace_from_stack=compute, insert1=inserted.append,
    )


def test_single_functional_channel_without_ch_id(monkeypatch, tmp_path):
    calls, inserted = [], []
    table = make_traces_table(monkeypatch, tmp_path, False, ['wDataCh0'], calls, inserted)
    traces.TracesTemplate.make(table, {'presentation_id': 1})
    assert calls == ['wDataCh0']
    assert [row.get('ch_id') for row in inserted] == [None]
    assert all(row['trigger_valid'] for row in inserted)


def test_traces_only_for_functional_channels(monkeypatch, tmp_path):
    calls, inserted = [], []
    table = make_traces_table(monkeypatch, tmp_path, True, ['wDataCh0', 'wDataCh1'], calls, inserted)
    traces.TracesTemplate.make(table, {'presentation_id': 1})
    assert calls == ['wDataCh0', 'wDataCh1']  # Not the background channel wDataCh2
    assert [row['ch_id'] for row in inserted] == [0, 1]
    assert all(row['trigger_valid'] for row in inserted)


def test_several_functional_channels_require_ch_id(monkeypatch, tmp_path):
    table = make_traces_table(monkeypatch, tmp_path, False, ['wDataCh0', 'wDataCh1'], [], [])
    with pytest.raises(ValueError, match='exactly one functional channel'):
        traces.TracesTemplate.make(table, {'presentation_id': 1})


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


class FakeTable:
    """Minimal stand-in for a table with a restrictable list of rows."""

    def __init__(self, rows, primary_key):
        self.rows = rows
        self.primary_key = primary_key

    def __and__(self, restriction):
        return FakeTable([r for r in self.rows if all(r[k] == v for k, v in restriction.items() if k in r)],
                         self.primary_key)

    def __call__(self):
        return self

    def __len__(self):
        return len(self.rows)

    def proj(self, *attrs):
        return self

    def to_dicts(self):
        return list(self.rows)

    def to_arrays(self, attr):
        return np.array([r[attr] for r in self.rows])

    def fetch1(self, *attrs):
        assert len(self.rows) == 1
        return tuple(self.rows[0][a] for a in attrs)


@pytest.fixture
def count_figures(monkeypatch):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    monkeypatch.setattr(plt, 'show', lambda: None)
    plt.close('all')
    return plt


def test_get_ch_ids():
    from djimaging.utils.dj_utils import get_ch_ids
    assert get_ch_ids(FakeTable([], ['roi_id']), {}) == [None]
    table = FakeTable([dict(roi_id=1, ch_id=1), dict(roi_id=1, ch_id=0)], ['roi_id', 'ch_id'])
    assert get_ch_ids(table, {'roi_id': 1}) == [0, 1]
    with pytest.raises(ValueError):
        get_ch_ids(table, {'roi_id': 2})


@pytest.mark.parametrize('n_channels', [1, 2])
def test_averages_plot_one_figure_with_column_per_channel(n_channels, count_figures):
    from djimaging.tables.core import averages
    pk = ['roi_id', 'ch_id'] if n_channels > 1 else ['roi_id']
    rows = [dict(roi_id=i, average=np.arange(5.) + i, average_norm=np.arange(5.) + i,
                 **({'ch_id': c} if n_channels > 1 else {}))
            for i in range(3) for c in range(n_channels)]
    averages.ResampledAveragesTemplate.plot(FakeTable(rows, pk), {})
    assert len(count_figures.get_fignums()) == 1
    titles = [ax.get_title() for ax in count_figures.gcf().axes if ax.get_title()]
    assert len([t for t in titles if t.startswith('average_norm')]) == n_channels
    assert len([t for t in titles if t.startswith('average') and not t.startswith('average_norm')]) == n_channels


@pytest.mark.parametrize('n_channels', [1, 2])
def test_averages_plot1_and_snippets_plot1_column_per_channel(n_channels, monkeypatch, count_figures):
    from djimaging.tables.core import averages, snippets
    pk = ['roi_id', 'ch_id'] if n_channels > 1 else ['roi_id']
    times = np.arange(10.)[:, None] + np.zeros((1, 3))
    fetched = (np.random.rand(10, 3), times, np.array([[0.], [0.], [0.]]).T + times[:1, :])
    monkeypatch.setattr(averages, 'fetch_snippets_and_times', lambda table, key: fetched)
    monkeypatch.setattr(snippets, 'fetch_snippets_and_times', lambda table, key: fetched)

    rows = [dict(roi_id=1, average=np.arange(5.), average_norm=np.arange(5.), average_t0=0., average_dt=1.,
                 triggertimes_rel=np.array([0.]), **({'ch_id': c} if n_channels > 1 else {}))
            for c in range(n_channels)]
    table = FakeTable(rows, pk)
    table.snippets_table = None

    def n_columns(fig):
        return len([ax for ax in fig.axes if ax.get_xlabel() == 'rel. to trigger'])

    averages.ResampledAveragesTemplate.plot1(table, key={'roi_id': 1})
    assert n_columns(count_figures.gcf()) == n_channels
    count_figures.close('all')

    snippets.SnippetsTemplate.plot1(table, key={'roi_id': 1})
    assert n_columns(count_figures.gcf()) == n_channels
