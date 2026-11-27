"""exposure_history_dj with in-memory fake tables (no database)."""
import copy
import os
from collections.abc import Mapping
from datetime import date, datetime, timedelta

import datajoint as dj
import numpy as np
import pandas as pd
import pytest

from djimaging.utils import exposure_history_dj as ehdj

EXP_KEY = dict(experimenter='Name', date=date(2024, 3, 1), exp_num=1)
EXP_PK = list(EXP_KEY)
T0 = datetime(2024, 3, 1, 10, 0, 0)


class FakeTable:
    """Rows in memory with the parts of the DataJoint 2 query API used by exposure_history_dj."""

    def __init__(self, rows=(), primary_key=(), **attrs):
        self.rows = [dict(r) for r in rows]
        self.primary_key = list(primary_key)
        self.__dict__.update(attrs)

    def __call__(self):
        return self

    def _with_rows(self, rows):
        out = copy.copy(self)
        out.rows = rows
        return out

    def __and__(self, restriction):
        assert isinstance(restriction, Mapping)
        return self._with_rows([r for r in self.rows if all(r[k] == v for k, v in restriction.items() if k in r)])

    def __len__(self):
        return len(self.rows)

    def __mul__(self, other):
        rows = [{**a, **b} for a in self.rows for b in other.rows
                if all(a[k] == b[k] for k in set(a) & set(b))]
        extra_pk = [k for k in other.primary_key if k not in self.primary_key]
        return FakeTable(rows, primary_key=self.primary_key + extra_pk)

    def proj(self, *attrs):
        keep = self.primary_key + [a for a in attrs if a not in self.primary_key]
        return self._with_rows([{k: r[k] for k in keep if k in r} for r in self.rows])

    def to_dicts(self):
        return [dict(r) for r in self.rows]

    def to_arrays(self, *attrs):
        arrays = tuple(np.array([r[a] for r in self.rows]) for a in attrs)
        return arrays[0] if len(attrs) == 1 else arrays

    def to_pandas(self):
        return pd.DataFrame(self.rows).set_index(self.primary_key) if self.primary_key else pd.DataFrame(self.rows)

    def fetch1(self, *attrs):
        assert len(self.rows) == 1, f'{len(self.rows)} rows'
        row = self.rows[0]
        if not attrs:
            return dict(row)
        if attrs == ('KEY',):
            return {k: row[k] for k in self.primary_key}
        return row[attrs[0]] if len(attrs) == 1 else tuple(row[a] for a in attrs)


def _rec(rec_name, kind, offset_min, h5_file=None, smp_file=None, **kw):
    row = {**EXP_KEY, **{a: None for a in ehdj._REC_ATTRS}}
    row.update(rec_name=rec_name, kind=kind, rec_start=T0 + timedelta(minutes=offset_min), rec_dur_s=30.,
               absx=0., absy=0., absz=30., zoom=3., pixel_size_um=1.5, nxpix=64, nypix=64, scan_type='xy',
               setupid=1, h5_file=h5_file, smp_file=smp_file, smh_file=None, name_info=None, angle_deg=0.)
    row.update(kw)
    return row


def _tables():
    recs = [
        _rec('SMP_GCL0_chirp', 'response', 0, h5_file='exp/Pre/SMP_GCL0_chirp.h5', smp_file='exp/Raw/a.smp',
             stim_token='chirp', smh_file='exp/Raw/a.smh',
             name_info={'h5': dict(field='GCL0', region=None, stimulus='chirp'), 'smp': dict(field='GCL0raw')}),
        _rec('SMP_GCL1_noise', 'response', 5, h5_file='exp/Pre/SMP_GCL1_noise.h5', stim_token='noise', setupid=3,
             name_info={'h5': dict(field='GCL1', region='LR', stimulus='noise')}),
        _rec('SMP_stack', 'hr', 10, scan_type='xyz', absx=100., absy=50., name_info={}),
    ]
    recording = FakeTable(recs, primary_key=EXP_PK + ['rec_name'])
    experiment = FakeTable([{**EXP_KEY, 'header_path': os.path.join('data', 'exp')}], primary_key=EXP_PK)
    exp_history = FakeTable([{**EXP_KEY, 'data_dirs': ['Pre', 'Raw'], 'unscanned_files': ['x.txt']}],
                            primary_key=EXP_PK, Recording=recording, experiment_table=experiment,
                            _filepath_store='not_configured_store')

    field_pk = EXP_PK + ['field', 'region', 'raw_id']
    averages = FakeTable([{**EXP_KEY, 'field': 'GCL0', 'region': 'N/A', 'raw_id': 0, 'ch_name': ch,
                           'ch_average': np.full((64, 32), v)} for ch, v in (('wDataCh0', 1.), ('wDataCh2', 2.))],
                         primary_key=field_pk + ['ch_name'])
    field = FakeTable([{**EXP_KEY, 'field': 'GCL0', 'region': 'N/A', 'raw_id': 0, 'absx': 1., 'absy': 2., 'absz': 3.,
                        'scan_type': 'xy', 'pixel_size_um': 0.5, 'nxpix': 64, 'nypix': 32,
                        'field_data_file': r'exp\Pre\SMP_GCL0_chirp.h5'}],
                      primary_key=field_pk, new_primary_keys=['field', 'region'], StackAverages=averages)
    field_rec = FakeTable([{**EXP_KEY, 'field': 'GCL0', 'region': 'N/A', 'raw_id': 0, 'rec_name': 'SMP_GCL0_chirp'}],
                          primary_key=field_pk + ['rec_name'])
    exp_history_field = FakeTable([], primary_key=field_pk, field_table=field, Recording=field_rec)

    pres_pk = EXP_PK + ['field', 'stim_name']
    pres = FakeTable([{**EXP_KEY, 'field': 'GCL0', 'stim_name': 'chirp',
                       'pres_data_file': 'exp\\Pre\\SMP_GCL0_chirp.h5'}],
                     primary_key=pres_pk)
    traces = FakeTable([{**EXP_KEY, 'field': 'GCL0', 'stim_name': 'chirp', 'roi_id': i} for i in range(1, 4)],
                       primary_key=pres_pk + ['roi_id'])
    motion = FakeTable([{**EXP_KEY, 'field': 'GCL0', 'stim_name': 'chirp', 'slow_range_px': v} for v in (0.5, 1.5)],
                       primary_key=pres_pk)
    trust = FakeTable([{**EXP_KEY, 'field': 'GCL0', 'stim_name': 'chirp', 'trust_median': m, 'n_rois_flagged': n}
                       for m, n in ((0.9, 1), (0.7, 2), (0.8, 0))], primary_key=pres_pk)
    return dict(exp_history_table=exp_history, exp_history_field_table=exp_history_field, presentation_table=pres,
                traces_table=traces, motion_detection_table=motion, motion_trust_table=trust)


def test_field_label():
    assert ehdj._field_label(dict(field='GCL0', region='LR'), ['field', 'region']) == 'GCL0/LR'
    assert ehdj._field_label(dict(field='GCL0', region='N/A'), ['field', 'region']) == 'GCL0'
    assert ehdj._field_label(dict(field='GCL0', region=None), ['field', 'region']) == 'GCL0'
    assert ehdj._field_label(dict(field='GCL0'), ['field', 'region']) == 'GCL0'
    assert ehdj._field_label(dict(field=1, cond1=''), ['field', 'cond1']) == '1'


def test_abs_path(tmp_path):
    assert ehdj._abs_path(None, 'x') is None
    assert ehdj._abs_path(np.nan, 'x') is None
    # unconfigured store: store-relative posix path
    assert ehdj._abs_path('a\\b\\c.h5', 'not_configured_store') == 'a/b/c.h5'
    with dj.config.override(stores={'default': 'ref', 'ref': {'protocol': 'file', 'location': str(tmp_path)}}):
        assert ehdj._abs_path('a/c.h5', 'ref') == str(tmp_path / 'a' / 'c.h5')


def test_field_labels():
    t = _tables()
    recs = pd.DataFrame(dict(rec_name=['SMP_GCL0_chirp', 'SMP_stack']))
    assert ehdj._field_labels(EXP_KEY, recs, None) == [None, None]
    assert ehdj._field_labels(EXP_KEY, recs.iloc[:0], t['exp_history_field_table']) == []
    assert ehdj._field_labels(EXP_KEY, recs, t['exp_history_field_table']) == ['GCL0', None]


def test_processing_status():
    t = _tables()
    assert ehdj._processing_status(EXP_KEY, None, None, None, None) == []
    rows = ehdj._processing_status(EXP_KEY, t['presentation_table'], t['traces_table'], t['motion_detection_table'],
                                   t['motion_trust_table'])
    assert rows == [dict(path='exp/Pre/SMP_GCL0_chirp.h5', n_traces=3, motion_slow_range_px=1.5,
                         motion_trust_median=0.8, motion_n_rois_flagged=3)]
    empty = FakeTable([], primary_key=['field'])
    rows = ehdj._processing_status(EXP_KEY, t['presentation_table'], empty, empty, empty)
    assert rows == [dict(path='exp/Pre/SMP_GCL0_chirp.h5', n_traces=0, motion_slow_range_px=None,
                         motion_trust_median=None, motion_n_rois_flagged=None)]


def test_stimulator_and_imaging_setup_meta():
    assert ehdj._stimulator_meta(EXP_KEY, None) == dict(stimulator_id=None)
    assert ehdj._imaging_setup_meta(EXP_KEY, None) == dict(imaging_setup_id=None)
    stimulator = FakeTable([dict(stimulator_id='s1', source='manual', projection='direct', um_per_px=2.)],
                           primary_key=['stimulator_id'])
    exp_stim = FakeTable([{**EXP_KEY, 'stimulator_id': 's1', 'confidence': 'high', 'reason': 'log'}],
                         primary_key=EXP_PK, stimulator_table=stimulator)
    meta = ehdj._stimulator_meta(EXP_KEY, exp_stim)
    assert meta['stimulator_id'] == 's1' and meta['stimulator_confidence'] == 'high'
    assert meta['stimulator_reason'] == 'log' and meta['stimulator_source'] == 'manual'
    assert meta['stimulator_um_per_px'] == 2. and meta['stimulator_projection'] == 'direct'
    assert meta['stimulator_color_map'] is None
    assert ehdj._stimulator_meta(dict(EXP_KEY, exp_num=2), exp_stim) == dict(stimulator_id=None)

    setup = FakeTable([dict(imaging_setup_id='i1', scan_width_um=120.)], primary_key=['imaging_setup_id'])
    exp_setup = FakeTable([{**EXP_KEY, 'imaging_setup_id': 'i1', 'confidence': 'low', 'reason': 'r'}],
                          primary_key=EXP_PK, imaging_setup_table=setup)
    assert ehdj._imaging_setup_meta(EXP_KEY, exp_setup) == dict(
        imaging_setup_id='i1', imaging_setup_confidence='low', imaging_setup_reason='r', scan_width_um=120.)


def _retinal_tables(od_rows, orientation_rows, retinal_rows=()):
    od = FakeTable(od_rows, primary_key=EXP_PK)
    rel = FakeTable([], opticdisk_table=od)
    orient = FakeTable(orientation_rows, primary_key=EXP_PK)
    return FakeTable(retinal_rows, primary_key=EXP_PK + ['field'], relativefieldlocation_table=rel,
                     retina_orientation_table=orient)


def _fields():
    return pd.DataFrame(dict(field=['A', 'B', 'C'], x_um=[110., 100., None], y_um=[20., 30., 0.],
                             ventral_dorsal_pos_um=[None, 5., None], temporal_nasal_pos_um=[None, 6., None],
                             retinal_source=[None, 'RetinalFieldLocation', None]))


def test_retinal_from_table():
    table = _retinal_tables([], [], [{**EXP_KEY, 'field': 'A', 'ventral_dorsal_pos_um': 1.,
                                      'temporal_nasal_pos_um': 2.}])
    out = ehdj._retinal_from_table({'A': {**EXP_KEY, 'field': 'A'}, 'B': {**EXP_KEY, 'field': 'B'}}, table)
    assert out == {'A': (1., 2., 'RetinalFieldLocation')}


def test_add_computed_retinal_right_eye_no_rotation():
    table = _retinal_tables([{**EXP_KEY, 'odx': 100., 'ody': 0.}],
                            [{**EXP_KEY, 'eye': 'right', 'orientation_deg': 0.}])
    fields = _fields()
    ehdj._add_computed_retinal(fields, EXP_KEY, table)
    # rotation 0, right eye: ventral_dorsal = -(x - odx), temporal_nasal = y - ody
    assert fields.loc[0, 'ventral_dorsal_pos_um'] == pytest.approx(-10.)
    assert fields.loc[0, 'temporal_nasal_pos_um'] == pytest.approx(20.)
    assert fields.loc[0, 'retinal_source'] == 'computed from OpticDisk'
    # existing positions and fields without stage position are left alone
    assert fields.loc[1, 'ventral_dorsal_pos_um'] == 5. and fields.loc[1, 'retinal_source'] == 'RetinalFieldLocation'
    assert pd.isna(fields.loc[2, 'ventral_dorsal_pos_um'])


@pytest.mark.parametrize('od_rows, orientation_rows', [
    ([], [dict(EXP_KEY, eye='right', orientation_deg=0.)]),  # no optic disc
    ([dict(EXP_KEY, odx=0., ody=0.), dict(EXP_KEY, odx=1., ody=0.)],
     [dict(EXP_KEY, eye='right', orientation_deg=0.)]),  # two optic discs
    ([dict(EXP_KEY, odx=0., ody=0.)], [dict(EXP_KEY, eye='unknown', orientation_deg=0.)]),
    ([dict(EXP_KEY, odx=0., ody=0.)], [dict(EXP_KEY, eye='left', orientation_deg=None)]),
])
def test_add_computed_retinal_unknown(od_rows, orientation_rows):
    fields = _fields()
    ehdj._add_computed_retinal(fields, EXP_KEY, _retinal_tables(od_rows, orientation_rows))
    pd.testing.assert_frame_equal(fields, _fields())


def test_add_computed_retinal_missing_tables_warns():
    fields = _fields()
    with pytest.warns(UserWarning, match='Retinal position'):
        ehdj._add_computed_retinal(fields, EXP_KEY, FakeTable([]))
    pd.testing.assert_frame_equal(fields, _fields())


def test_images_from_file_unreadable(tmp_path):
    assert ehdj._images_from_file(None, np.nan, 'xy', ('wDataCh0',), 'x') == (None, 'no file')
    missing = str(tmp_path / 'missing.h5')
    imgs, issue = ehdj._images_from_file(missing, None, 'xy', ('wDataCh0',), 'x')
    assert imgs is None and issue.startswith('missing.h5: ')


def test_report_dir_next_to_ini():
    t = _tables()
    assert ehdj.report_dir_next_to_ini(EXP_KEY, t['exp_history_table']) == \
        os.path.join('data', 'exp', ehdj.INI_REPORT_DIR)


def test_file_paths():
    t = _tables()
    recs = pd.DataFrame([dict(rec_name='SMP_GCL0_chirp', field='GCL0', kind='response',
                              h5_file='exp/Pre/SMP_GCL0_chirp.h5', smp_file=None)])
    out = ehdj._file_paths(EXP_KEY, t['exp_history_table'], recs)
    assert out['header_path'] == os.path.join('data', 'exp')
    assert out['files'] == [dict(field='GCL0', rec_name='SMP_GCL0_chirp', kind='response',
                                 h5_file='exp/Pre/SMP_GCL0_chirp.h5', smp_file=None, smh_file='exp/Raw/a.smh')]


def test_fetch_exposure_history_with_fake_tables():
    t = _tables()
    events, meta = ehdj.fetch_exposure_history(EXP_KEY, **t)
    assert meta['experiment'] == '2024-03-01_1_Name'
    assert meta['n_recordings'] == 3
    assert meta['data_dirs'] == ['Pre', 'Raw'] and meta['unscanned_files'] == ['x.txt']
    assert meta['setupid'] == '1;3'
    assert meta['has_stim_log'] is False
    assert meta['stimulator_id'] is None and meta['imaging_setup_id'] is None
    assert [f['rec_name'] for f in meta['files']] == ['SMP_GCL0_chirp', 'SMP_GCL1_noise', 'SMP_stack']
    assert meta['files'][0]['field'] == 'GCL0' and pd.isna(meta['files'][1]['field'])
    rec_events = events[events['kind'] == 'response'].set_index('rec_name')  # laser events
    # Presentation file stored with backslashes is matched to the recording's .h5 file
    assert rec_events.loc['SMP_GCL0_chirp', 'n_traces'] == 3
    assert rec_events.loc['SMP_GCL0_chirp', 'motion_trust_median'] == pytest.approx(0.8)
    assert pd.isna(rec_events.loc['SMP_GCL1_noise', 'n_traces'])


def test_fetch_exposure_history_without_recordings():
    t = _tables()
    t['exp_history_table'].Recording.rows = []
    events, meta = ehdj.fetch_exposure_history(EXP_KEY, exp_history_table=t['exp_history_table'])
    assert meta['n_recordings'] == 0 and meta['files'] == [] and meta['setupid'] is None


def test_fetch_field_map_from_file_names():
    t = _tables()
    fields = ehdj.fetch_field_map(EXP_KEY, t['exp_history_table'], images=False)
    by_field = fields.set_index('field')
    # region None in the name info -> default 'N/A', which is not part of the label
    assert {'GCL0', 'GCL1/LR'} <= set(by_field.index)
    assert by_field.loc['GCL0', 'status'] == 'response'
    fields_raw = ehdj.fetch_field_map(EXP_KEY, t['exp_history_table'], images=False, from_raw_data=True)
    assert 'GCL0raw' in set(fields_raw['field'])


def test_field_table_info():
    field = _tables()['exp_history_field_table'].field_table
    key = {**EXP_KEY, 'field': 'GCL0', 'region': 'N/A', 'raw_id': 0}
    info = ehdj._field_table_info(field, key, images=True, image_channels=('wDataCh0', 'wDataCh1'))
    assert (info['x_um'], info['y_um'], info['z_um']) == (1., 2., 3.)
    assert info['width_um'] == 32. and info['height_um'] == 16.
    assert info['image_rec'] == 'SMP_GCL0_chirp.h5'
    assert list(info['images']) == ['wDataCh0'] and info['image_source'] == 'Field.StackAverages'
    field.rows[0]['pixel_size_um'] = 0.
    info = ehdj._field_table_info(field, key, images=False, image_channels=())
    assert 'width_um' not in info and 'images' not in info


def test_fetch_field_map_linked_to_field_table():
    t = _tables()
    fields = ehdj.fetch_field_map(EXP_KEY, t['exp_history_table'], exp_history_field_table=t['exp_history_field_table'],
                                  images=True, images_from_files=False)
    by_field = fields.set_index('field')
    assert by_field.loc['GCL0', 'x_um'] == 1. and by_field.loc['GCL0', 'image_source'] == 'Field.StackAverages'
    # recordings not linked to the Field table are grouped by their file names
    assert 'GCL1/LR' in by_field.index
    t['exp_history_table'].Recording.rows = []
    assert len(ehdj.fetch_field_map(EXP_KEY, t['exp_history_table'])) == 0


def test_fetch_exposure_history_listed_and_stim_log():
    from djimaging.tables.history.exp_history import ExpHistoryTemplate

    class FakeExpHistory(FakeTable):
        listed = ExpHistoryTemplate.listed

    t = _tables()
    exp_history = t['exp_history_table']
    t['exp_history_table'] = FakeExpHistory(exp_history.rows, exp_history.primary_key, **{
        k: v for k, v in exp_history.__dict__.items() if k not in ('rows', 'primary_key')})
    stimulus = FakeTable([dict(stim_name='chirp', alias='chirp_globalchirp', stim_hash='m_chirp', stim_dict={}),
                          dict(stim_name='noise', alias='dn_noise', stim_hash='', stim_dict=dict(
                              stim_name_list=['noise_']))], primary_key=['stim_name'])
    log_idx = dict(log_idx=0)
    qdspy = timedelta(hours=10, seconds=-60)  # QDSpy clock: ScanM time - 60 s
    stim_log_entries = FakeTable([
        dict(log_idx, stim_idx=0, stim_file_name='RGC_Chirp', stim_md5='m_chirp', t_start=qdspy + timedelta(seconds=2),
             t_end=qdspy + timedelta(seconds=30), t_dur_s=28., aborted=False),
        dict(log_idx, stim_idx=1, stim_file_name='noise_20um', stim_md5='m_x', t_start=qdspy + timedelta(minutes=5),
             t_end=qdspy + timedelta(minutes=5, seconds=30), t_dur_s=30., aborted=False),
    ], primary_key=['log_idx', 'stim_idx'])
    stim = FakeTable([{**EXP_KEY, **log_idx, 'stim_idx': 0, 'rec_name': 'SMP_GCL0_chirp', 'status': 'matched'},
                      {**EXP_KEY, **log_idx, 'stim_idx': 1, 'rec_name': 'SMP_GCL1_noise', 'status': 'matched'}],
                     primary_key=EXP_PK + ['log_idx', 'stim_idx'], stim_log_table=stim_log_entries)
    stim_log = FakeTable([{**EXP_KEY, 'clock_offset_s': 60., 'n_stims': 2, 'n_matched': 2, 'n_outside': 0}],
                         primary_key=EXP_PK, Stim=stim)

    events, meta = ehdj.fetch_exposure_history(EXP_KEY, stimulus_table=stimulus, stim_log_table=stim_log, **t)
    assert meta['has_stim_log'] is True and meta['clock_offset_s'] == 60.
    assert (meta['n_stims_logged'], meta['n_stims_matched'], meta['n_stims_other_experiments']) == (2, 2, 0)
    laser = events[events['kind'] == 'response'].set_index('rec_name')
    assert laser.loc['SMP_GCL0_chirp', 'listed'] == 'listed' and laser.loc['SMP_GCL1_noise', 'listed'] == 'listed'
    assert laser.loc['SMP_GCL0_chirp', 'stim_name'] == 'chirp'
    assert laser.loc['SMP_GCL0_chirp', 'has_presentation'] == True  # noqa: E712
    assert laser.loc['SMP_GCL1_noise', 'has_presentation'] == False  # noqa: E712
    # logged stimuli are named via the md5 (chirp) or the file name prefix (noise)
    assert {'chirp', 'noise'} <= set(events['stim_name'].dropna())


def test_export_exposure_history(tmp_path):
    t = _tables()
    results = ehdj.export_exposure_history(str(tmp_path), [EXP_KEY], html=True, field_map=True,
                                           field_map_kwargs=dict(images=False), **t)
    assert len(results) == 1 and results[0]['next_to_ini'] is None
    exp_dir = tmp_path / '2024-03-01_1_Name'
    assert exp_dir.is_dir() and any(p.suffix == '.html' for p in exp_dir.iterdir())
    # existing report is skipped by default
    with pytest.warns(UserWarning, match='Report exists'):
        assert ehdj.export_exposure_history(str(tmp_path), [EXP_KEY], field_map=False, **t)[0]['out_dir'] is None
