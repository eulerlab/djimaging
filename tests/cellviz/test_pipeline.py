"""Opening CellViz on a populated schema: finding the tables, choosing the field, export folder in the processed store.

Uses the offline table substitute of ``test_dj_adapter`` (no database, SYNTHETIC rows, every write raises).
"""
import sys
import types

import datajoint as dj
import pandas as pd
import pytest

from djimaging.cellviz import dj_adapter
from djimaging.cellviz.dj_adapter import CellVizTables, DataJointFieldSource, list_fields, resolve_field_key
from djimaging.cellviz.export import EXPORT_SUBDIR, FALLBACK_EXPORT_DIR, processed_export_dir
from tests.cellviz.test_dj_adapter import FIELD, FakeTable, build_tables


def _with_second_field(tables):
    """Add field 'SYN2' (mask + one trace row) to the tables of ``build_tables``."""
    other = dict(FIELD, field='SYN2')
    tables.roi_mask._rows.append(dict(other, raw_id=1))
    row = dict(tables.traces._rows[0], field='SYN2')
    tables.traces._rows.append(row)
    return tables


def _schema_module(tables, name='cellviz_test_schema'):
    """A module like ``djimaging.user.<project>.schemas.<project>_schema``: a ``schema`` object + the tables."""
    module = types.ModuleType(name)
    module.schema = object()
    module.Presentation, module.RoiMask, module.Traces = tables.presentation, tables.roi_mask, tables.traces
    module.Stimulus, module.PreprocessTraces, module.ChirpQI = tables.stimulus, tables.preprocess_traces, \
        tables.chirp_qi
    module.Channel = tables.channel
    return module


@pytest.fixture
def stores(tmp_path):
    """A DataJoint config with local reference and processed stores (restored afterwards)."""
    old = dj.config['stores']
    paths = {name: tmp_path / name for name in ('reference', 'processed')}
    dj.config['stores'] = {'default': 'processed', 'filepath_default': 'reference',
                           **{n: {'protocol': 'file', 'location': str(p)} for n, p in paths.items()}}
    try:
        yield paths
    finally:
        dj.config['stores'] = old


# ------------------------------------------------------------------ field selection
def test_list_and_resolve_fields():
    tables, _ = build_tables()
    assert list_fields(tables) == [dict(FIELD, raw_id=1)]
    assert resolve_field_key(tables, None) == (FIELD, [])
    assert resolve_field_key(tables, dict(field='SYN'))[0] == FIELD  # partial key
    longer = dict(FIELD, stim_name='chirp', cond1='drug', raw_id=1, roi_id=5)  # e.g. a Traces key
    assert resolve_field_key(tables, longer)[0] == FIELD
    assert resolve_field_key(tables, dict(FIELD, date=pd.Timestamp('2099-01-01').date().isoformat()))[0] == FIELD
    with pytest.raises(ValueError, match='available'):
        resolve_field_key(tables, dict(field='nope'))
    with pytest.raises(ValueError, match='none of the field attributes'):
        resolve_field_key(tables, dict(stim_name='chirp'))


def test_ambiguous_field_key_asks_to_be_more_specific():
    tables = _with_second_field(build_tables()[0])
    with pytest.raises(ValueError, match='2 fields match'):
        resolve_field_key(tables, dict(date='2099-01-01'))
    assert resolve_field_key(tables, dict(field='SYN2'))[0] == dict(FIELD, field='SYN2')
    key, notes = resolve_field_key(tables, None)
    assert key == FIELD and 'first of 2 fields' in notes[0]


def test_fields_without_traces_are_not_offered():
    tables, _ = build_tables()
    tables.roi_mask._rows.append(dict(FIELD, field='MASK_ONLY', raw_id=1))  # mask, but Traces not populated
    assert [k['field'] for k in list_fields(tables)] == ['SYN']


# ------------------------------------------------------------------ finding the tables
def test_tables_from_module_namespace_and_schema_object():
    tables, _ = build_tables()
    module = _schema_module(tables)
    for schema in (module, vars(module)):
        t = CellVizTables.from_schema(schema)
        assert t.presentation is tables.presentation and t.chirp_qi is tables.chirp_qi and t.highres is None
    block = types.ModuleType('cellviz_test_block')  # a block module sharing the schema object, with fewer tables
    block.schema, block.Presentation = module.schema, tables.presentation
    sys.modules[module.__name__], sys.modules[block.__name__] = module, block
    try:
        assert CellVizTables.from_schema(module.schema).chirp_qi is tables.chirp_qi  # the module with most tables
    finally:
        del sys.modules[module.__name__], sys.modules[block.__name__]
    with pytest.raises(ValueError, match='no imported module'):
        CellVizTables.from_schema(object())
    with pytest.raises(ValueError, match='not found'):
        CellVizTables.from_schema({'Presentation': tables.presentation})


def test_cellviz_from_schema_opens_the_chosen_field(monkeypatch, tmp_path):
    pytest.importorskip('ipywidgets')
    from djimaging.cellviz import CellViz
    tables, loader = build_tables(n_channels=1, preprocess_ids=(1,))
    tables = _with_second_field(tables)
    monkeypatch.setattr(dj_adapter, '_movie_from_file', lambda t, *args: loader(*args))
    module = _schema_module(tables)
    fields = CellViz.list_fields(module)
    assert isinstance(fields, pd.DataFrame) and sorted(fields['field']) == ['SYN', 'SYN2']
    v = CellViz.from_schema(module, field_key=dict(field='SYN'), backend='image', export_dir=tmp_path,
                            overlay='mask')
    assert v.source.field_key == FIELD and v.source.raw_id == 1
    assert v.pres.info.key['cond1'] == 'control' and v.state.overlay == 'mask'
    assert sorted(v.pres.signals) == ['preprocessed', 'raw']


def test_writes_are_impossible():
    tables, loader = build_tables()
    src = DataJointFieldSource(tables, FIELD, preprocess_id=2, movie_loader=loader, data_roots=())
    src.load_presentation(src.list_presentations()[0].name, 'wDataCh0')
    for table in (tables.presentation, tables.roi_mask, tables.traces):
        assert isinstance(table, FakeTable)  # FakeTable raises on insert/delete/update: nothing was written


# ------------------------------------------------------------------ export folder
def test_exports_go_to_the_processed_store_and_never_to_the_reference_store(stores):
    path, how = processed_export_dir('ageuler_me_proj')
    assert path == stores['processed'].resolve() / EXPORT_SUBDIR / 'ageuler_me_proj' and 'processed' in how
    tables, loader = build_tables()
    tables.presentation.database = 'ageuler_me_proj'
    src = DataJointFieldSource(tables, FIELD, preprocess_id=2, movie_loader=loader)
    assert src.data_roots == (stores['reference'].resolve(),)
    assert src.default_export_dir()[0] == path


def test_export_folder_falls_back_without_processed_store():
    old = dj.config['stores']
    dj.config['stores'] = {}
    try:
        path, how = processed_export_dir('x')
    finally:
        dj.config['stores'] = old
    assert path == FALLBACK_EXPORT_DIR / 'x' and 'fallback' in how


def test_viewer_uses_the_processed_store_and_refuses_the_reference_store(stores, monkeypatch):
    pytest.importorskip('ipywidgets')
    from djimaging.cellviz import CellViz
    tables, loader = build_tables(n_channels=1, preprocess_ids=(1,))
    tables.presentation.database = 'ageuler_me_proj'
    monkeypatch.setattr(dj_adapter, '_movie_from_file', lambda t, *args: loader(*args))
    v = CellViz.from_schema(_schema_module(tables), backend='image')
    assert v.export_dir == stores['processed'].resolve() / 'cellviz' / 'ageuler_me_proj'
    assert 'processed' in v.w_export_dir.tooltip and str(v.export_dir) in v.w_notes.value
    pytest.importorskip('imageio_ffmpeg')
    with pytest.raises(ValueError, match='read-only'):
        v.export(stores['reference'] / 'Data' / 'x.mp4', t_start=0, t_stop=0.2)
    meta = v.export(t_start=0, t_stop=0.2)
    assert meta['file'].startswith(str(v.export_dir))
