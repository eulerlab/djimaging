"""Read-only DataJoint adapter, tested with an offline substitute for the tables (no database, SYNTHETIC rows).

The fake tables implement only the read API the adapter uses (``&``, ``keys``, ``fetch1``, ``to_dicts``,
``to_arrays``, ``len``, ``primary_key``); every write method raises, so a test fails if the adapter writes.
"""
import numpy as np
import pytest

from djimaging.inspector.dj_adapter import DataJointFieldSource, InspectorTables
from djimaging.inspector.quality import classify
from tests.inspector.synthetic import FRAME_DT, LINE_OFFSETS, ROI_IDS, TRIGGERS, roi_labels_xy, make_presentation

FIELD = dict(experimenter='Syn', date='2099-01-01', exp_num=1, field='SYN')


class FakeTable:
    def __init__(self, rows, primary_key, parts=None, **attrs):
        self._rows = [dict(r) for r in rows]
        self.primary_key = list(primary_key)
        for name, part in (parts or {}).items():
            setattr(self, name, part)
        for k, v in attrs.items():
            setattr(self, k, v)

    def _restrict(self, restriction):
        if isinstance(restriction, FakeTable):
            keys = [{k: r[k] for k in restriction.primary_key} for r in restriction._rows]
            rows = [r for r in self._rows if any(all(r.get(k) == v for k, v in key.items() if k in r) for key in keys)]
        else:
            rows = [r for r in self._rows if all(r[k] == v for k, v in dict(restriction).items() if k in r)]
        clone = FakeTable(rows, self.primary_key)
        clone.__dict__.update({k: v for k, v in self.__dict__.items() if k not in ('_rows',)})
        clone._rows = rows
        return clone

    def __and__(self, restriction):
        return self._restrict(restriction)

    def __len__(self):
        return len(self._rows)

    def keys(self):
        return [{k: r[k] for k in self.primary_key} for r in self._rows]

    def to_dicts(self):
        return [dict(r) for r in self._rows]

    def to_arrays(self, attr):
        return np.array([r[attr] for r in self._rows])

    def fetch1(self, *attrs):
        if len(self._rows) != 1:
            raise AssertionError(f"fetch1 on {len(self._rows)} rows")
        row = self._rows[0]
        return row[attrs[0]] if len(attrs) == 1 else tuple(row[a] for a in attrs)

    def _write(self, *args, **kwargs):
        raise AssertionError("the inspector must not write to the database")

    insert = insert1 = delete = update1 = drop = _write


def build_tables(n_channels=2, preprocess_ids=(1, 2), qi_rows=True):
    pres_pk = list(FIELD) + ['stim_name', 'cond1', 'raw_id']
    synthetic = {'control': make_presentation('c', 'control', 0., 0, {}), 'drug': make_presentation('d', 'drug', 1e4, 1, {})}
    pres_rows, mask_rows, trace_rows, pp_rows, qi = [], [], [], [], []
    movies = {}
    for cond, sp in synthetic.items():
        key = dict(FIELD, stim_name='chirp', cond1=cond, raw_id=1)
        pres_rows.append(dict(key, pres_data_file=f'/nowhere/{cond}.h5', triggertimes=TRIGGERS, npixartifact=0,
                              pixel_size_um=2.0, absx=10.0, absy=20.0, nxpix=20, nypix=16, scan_type='xy'))
        movies[f'/nowhere/{cond}.h5'] = sp
        mask_rows.append(dict(key, roi_mask=roi_labels_xy(0 if cond == 'control' else 1),
                              as_field_mask='same' if cond == 'control' else 'shifted',
                              shift_dx=0 if cond == 'control' else 1, shift_dy=0))
        for ch in range(n_channels):
            for k, tr in sp.signals['raw'].traces.items():
                row = dict(key, roi_id=k, trace=tr.values + 1e6 * ch, trace_t0=tr.times[0], trace_dt=FRAME_DT,
                           trace_valid=1, trigger_valid=1)
                if n_channels > 1:
                    row['ch_id'] = ch
                trace_rows.append(row)
                for pid in preprocess_ids:
                    pp_rows.append(dict(row, preprocess_id=pid, pp_trace=tr.values * 0 + pid + 10 * ch,
                                        pp_trace_t0=tr.times[0], pp_trace_dt=FRAME_DT))
                    if qi_rows and k != 11:  # ROI 11 has no QI entry -> missing
                        qi.append(dict(row, preprocess_id=pid, qidx=0.1 * k + pid + ch, min_qidx=0.2))
    ch_pk = ['ch_id'] if n_channels > 1 else []
    trace_pk = pres_pk + ['roi_id'] + ch_pk
    tables = InspectorTables(
        presentation=FakeTable(pres_rows, pres_pk, parts=dict(ScanInfo=FakeTable([], pres_pk)),
                               _filepath_store='reference'),
        roi_mask=FakeTable([dict(FIELD, raw_id=1)], list(FIELD) + ['raw_id'],
                           parts=dict(RoiMaskPresentation=FakeTable(mask_rows, pres_pk))),
        traces=FakeTable(trace_rows, trace_pk),
        stimulus=FakeTable([dict(stim_name='chirp', stim_family='chirp', isrepeated=1, ntrigger_rep=2,
                                 trial_info=None)], ['stim_name']),
        preprocess_traces=FakeTable(pp_rows, trace_pk + ['preprocess_id']),
        chirp_qi=FakeTable(qi, trace_pk + ['preprocess_id']),
        channel=FakeTable([dict(ch_id=c, ch_name=f'wDataCh{c}') for c in range(n_channels)], ['ch_id']),
    )

    def movie_loader(pres_data_file, channel, from_raw_data, precision):
        sp = movies[pres_data_file]
        return np.transpose(sp.frames, (2, 1, 0)), sp.frame_times, sp.frame_dt  # stored [x, y, t] like ScanM stacks

    return tables, movie_loader


def test_lists_both_presentations_of_the_same_stimulus():
    tables, loader = build_tables()
    src = DataJointFieldSource(tables, FIELD, preprocess_id=2, movie_loader=loader)
    names = [i.name for i in src.list_presentations()]
    assert names == ['chirp  [cond1=control]', 'chirp  [cond1=drug]']
    assert [i.channels for i in src.list_presentations()] == [('wDataCh0', 'wDataCh1')] * 2


def test_loads_matching_mask_channel_traces_and_quality():
    tables, loader = build_tables()
    src = DataJointFieldSource(tables, FIELD, preprocess_id=2, movie_loader=loader)
    pres = src.load_presentation('chirp  [cond1=drug]', 'wDataCh1')
    assert pres.mask_info == dict(as_field_mask='shifted', shift_dx=1, shift_dy=0)
    np.testing.assert_array_equal(pres.rois.labels, roi_labels_xy(1).T)  # the drug (shifted) mask
    assert pres.roi_ids.tolist() == list(ROI_IDS)
    raw = pres.signals['raw'].traces
    for k in ROI_IDS:  # drug movie offset 1e4 and channel 1 offset 1e6: no mixing of presentations or channels
        first = 10 if k == 11 else 0
        assert raw[k].values[0] == 1e6 + 1e4 + 100 * k + first
        assert raw[k].times[0] == pytest.approx(first * FRAME_DT + LINE_OFFSETS[k])
    assert set(np.unique(pres.signals['preprocessed'].traces[2].values)) == {2 + 10}  # preprocess_id 2, channel 1
    q = pres.quality
    assert q.table == 'ChirpQI' and q.metric == 'qidx' and 'preprocess_id=2' in q.source
    assert q.values == {2: pytest.approx(0.2 + 2 + 1), 5: pytest.approx(0.5 + 2 + 1)}
    assert classify(pres.roi_ids, q, 3.4).missing == (11,)
    np.testing.assert_array_equal(pres.frames[0], loader('/nowhere/drug.h5', 'wDataCh1', False, 'line')[0][..., 0].T)


def test_ambiguous_preprocess_id_is_not_mixed():
    tables, loader = build_tables()
    src = DataJointFieldSource(tables, FIELD, movie_loader=loader)
    with pytest.raises(ValueError, match='several preprocess_id'):
        src.load_presentation('chirp  [cond1=control]', 'wDataCh0')


def test_single_channel_schema_and_missing_qi_table():
    tables, loader = build_tables(n_channels=1, preprocess_ids=(1,))
    tables.chirp_qi = None
    src = DataJointFieldSource(tables, FIELD, movie_loader=loader)
    pres = src.load_presentation('chirp  [cond1=control]', 'wDataCh0')
    assert pres.quality is None and 'ChirpQI is not in this schema' in pres.quality_unavailable
    assert pres.signals['raw'].traces[5].values[0] == 500


def test_unpopulated_qi_means_all_missing():
    tables, loader = build_tables(n_channels=1, preprocess_ids=(1,), qi_rows=False)
    src = DataJointFieldSource(tables, FIELD, movie_loader=loader)
    pres = src.load_presentation('chirp  [cond1=control]', 'wDataCh0')
    assert pres.quality.values == {} and 'not populated' in pres.quality.source
    c = classify(pres.roi_ids, pres.quality, 0.1)
    assert c.eligible == () and c.missing == ROI_IDS


def test_from_namespace_picks_schema_names():
    tables, _ = build_tables()
    ns = dict(Presentation=tables.presentation, RoiMask=tables.roi_mask, Traces=tables.traces, ChirpQI=tables.chirp_qi)
    t = InspectorTables.from_namespace(ns)
    assert t.presentation is tables.presentation and t.chirp_qi is tables.chirp_qi and t.os_ds_indexes is None
