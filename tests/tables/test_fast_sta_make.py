import inspect
import time

import deepdiff
import numpy as np
import pytest

from djimaging.tables.receptivefield.fast_sta import FastStaTemplate, _FetchedData
from djimaging.tables.receptivefield.glm import RfGlmTemplate
from djimaging.utils.receptive_fields.fit_rf_utils import compute_rf_sta, compute_whitening_matrix


def _deephash(fetched_data):
    # As in datajoint.autopopulate.AutoPopulate._populate1
    return deepdiff.DeepHash(fetched_data, ignore_iterable_order=False)[fetched_data]


def _fetched(traces, **unhashed):
    return _FetchedData(
        triggertimes=np.arange(100.), rf_keys=[dict(roi_id=i) for i in range(len(traces))],
        trace_t0s=np.zeros(len(traces)), trace_dts=np.full(len(traces), 0.1), traces=traces, unhashed=unhashed)


@pytest.mark.parametrize('table_class', [FastStaTemplate, RfGlmTemplate])
def test_make_is_tripartite(table_class):
    """The computation must run outside of the transaction, see `FastStaTemplate.make_fetch`."""
    assert 'make' not in vars(table_class)
    assert inspect.isgeneratorfunction(table_class.make)


def test_fetched_data_hash_is_fast_and_sensitive():
    rng = np.random.default_rng(0)
    traces = [rng.normal(size=30_000).astype(np.float32) for _ in range(200)]

    t_start = time.time()
    fetched = (_fetched(traces, sta_params=dict(x_stimulus=np.ones((10, 10)))),)
    hash1 = _deephash(fetched)
    assert time.time() - t_start < 5.

    # Unhashed data does not change the hash, the fetched data does.
    assert _deephash((_fetched(traces, sta_params=None),)) == hash1
    traces[3] = traces[3].copy()
    traces[3][100] += 1
    assert _deephash((_fetched(traces),)) != hash1
    assert fetched[0].data['traces'] is traces and 'sta_params' in fetched[0].data


def test_make_generator_inserts_computed_entries():
    """Run the tripartite make as `AutoPopulate._populate1` does."""
    inserted = []
    entries = [dict(roi_id=1, rf=np.ones(3))]

    class Table(FastStaTemplate):
        def make_fetch(self, key, **kwargs):
            return _fetched([np.ones(5)], **kwargs),

        def make_compute(self, key, fetched):
            assert fetched.data['n_jobs'] == 3
            return entries,

        def make_insert(self, key, rf_entries):
            inserted.extend(rf_entries)

    table = object.__new__(Table)
    gen = table.make(dict(field_id=1), n_jobs=3)
    fetched_data = next(gen)
    fetch_hash = _deephash(fetched_data)
    computed_result = next(gen)
    gen = table.make(dict(field_id=1), n_jobs=3)
    assert _deephash(next(gen)) == fetch_hash
    gen.send(computed_result)
    assert inserted == entries


@pytest.mark.parametrize('whiten', [False, True])
def test_batched_sta_matches_single_traces(whiten):
    rng = np.random.default_rng(1)
    dims = (4, 3, 5)
    x_stimulus = rng.normal(size=(200, int(np.prod(dims)))).astype(np.float32)
    # Traces of different lengths are zero-padded in the batched computation.
    traces = [rng.random(n).astype(np.float32) for n in (200, 190, 200, 150)]
    whitening_matrix = compute_whitening_matrix(x_stimulus) if whiten else None
    sta_params = dict(x_stimulus=x_stimulus, dims=dims, whitening_matrix=whitening_matrix)

    entries = FastStaTemplate._compute_entries(FastStaTemplate, traces, sta_params, n_jobs=2)

    for entry, trace in zip(entries, traces):
        rf = compute_rf_sta(x_stimulus[:trace.size], trace)
        if whiten:
            rf = whitening_matrix @ rf
        assert entry['rf'].shape == dims
        np.testing.assert_allclose(entry['rf'], rf.reshape(dims), rtol=1e-4, atol=1e-6)
