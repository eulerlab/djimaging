"""Read-only DataJoint adapter: pipeline results of one field -> inspector data objects.

You pass the table classes (or instances) of your own, already activated schema; this module never connects,
activates or declares anything and only *reads* (``&``, ``keys``, ``fetch1``, ``to_dicts``, ``to_arrays``, the same
access patterns as the tables' own ``make`` methods). No inserts, updates or deletes, no new tables.

What is read from where:

- ``Presentation``: one entry per presentation of the field (full primary key: field key + ``stim_name`` + conditions
  + ``raw_id``), ``pres_data_file``, ``triggertimes`` (as stored, incl. stimulator delay), scan geometry,
  ``Presentation.ScanInfo`` (zoom, angle).
- ``RoiMask.RoiMaskPresentation``: the presentation's own mask (Igor format, already shifted if ``as_field_mask`` is
  'shifted'); this is the mask ``Traces`` was computed with. ROI ID = ``-label``.
- ``Traces`` (``trace``, ``trace_t0``, ``trace_dt``, ``trace_valid``, ``trigger_valid``), restricted to the
  presentation key and, with several functional channels, to ``ch_id``.
- ``PreprocessTraces`` (``pp_trace``, ...) and the QI tables (``ChirpQI.qidx``/``min_qidx``,
  ``OsDsIndexes.d_qi``/``ds_index``/``os_index``), restricted to one ``preprocess_id``.
- Movie: not stored in the database; loaded from ``pres_data_file`` with ``read_utils.load_stacks`` and timed with
  ``wparams_utils.compute_frame_times``, i.e. the code that computed the traces.
- ``HighRes.StackAverages`` for the HighRes view.
- ``Stimulus`` (``isrepeated``, ``ntrigger_rep``, ``trial_info``, ``stim_dict['trigger_delay']``) and
  ``Snippets._pad_trace``: the trial average cuts the traces exactly like ``Snippets.make``.

Keys are never mixed: if several ``raw_id`` or ``preprocess_id`` exist and none is given, loading raises and asks for
one. Two presentations of the same stimulus (different conditions) are separate entries of the selector.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

import numpy as np

from djimaging.inspector.data import (
    HighResImage, PresentationData, PresentationInfo, QualityValues, RoiLabels, Signal, Trace, to_display)
from djimaging.inspector.offline import stimulus_info_for, trigger_labels_for
from djimaging.inspector.quality import QUALITY_METRICS, quality_family, unsupported_reason
from djimaging.inspector.timing import sample_times

SCHEMA_NAMES = dict(presentation='Presentation', roi_mask='RoiMask', traces='Traces', stimulus='Stimulus',
                    raw_params='RawDataParams', preprocess_traces='PreprocessTraces', highres='HighRes',
                    chirp_qi='ChirpQI', os_ds_indexes='OsDsIndexes', channel='Channel', userinfo='UserInfo',
                    snippets='Snippets')


@dataclass
class InspectorTables:
    """Tables of an activated schema. Only ``presentation``, ``roi_mask`` and ``traces`` are required."""
    presentation: Any
    roi_mask: Any
    traces: Any
    stimulus: Any = None
    raw_params: Any = None
    preprocess_traces: Any = None
    highres: Any = None
    chirp_qi: Any = None
    os_ds_indexes: Any = None
    channel: Any = None
    userinfo: Any = None
    snippets: Any = None  # only its ``_pad_trace`` setting is read, for the trial average

    @classmethod
    def from_namespace(cls, namespace: Mapping) -> "InspectorTables":
        """Pick the tables by their example-schema names from e.g. ``globals()`` after ``from my_schema import *``."""
        return cls(**{attr: namespace.get(name) for attr, name in SCHEMA_NAMES.items()})


def _movie_from_file(tables: InspectorTables, pres_data_file, channel: str, from_raw_data: bool, precision: str):
    """Default movie loader: the recording file referenced by the presentation, read like ``Traces.make`` does."""
    from djimaging.utils.dj_storage import local_path
    from djimaging.utils.scanm import read_utils, wparams_utils
    store = getattr(tables.presentation, '_filepath_store', 'reference')
    with local_path(pres_data_file, store) as filepath:
        ch_stacks, wparams = read_utils.load_stacks(str(filepath), from_raw_data, ch_names=(channel,))
    stack = ch_stacks[channel]
    frame_times, _, frame_dt = wparams_utils.compute_frame_times(wparams, n_frames=stack.shape[2], precision=precision)
    return stack, frame_times, frame_dt


class DataJointFieldSource:
    """All presentations of one field from the database.

    Parameters
    ----------
    tables : ``InspectorTables`` (e.g. ``InspectorTables.from_namespace(globals())``)
    field_key : key of the field (e.g. one of ``Field().keys()``)
    raw_id, preprocess_id : pin these keys if the schema has several parameter sets.
    movie_loader : ``(pres_data_file, channel, from_raw_data, precision) -> (stack [x, y, t], frame_times, frame_dt)``;
        default reads the file of the presentation. Replaceable for tests or when files are elsewhere.
    data_roots : folders that video exports must not be written into.
    """

    def __init__(self, tables: InspectorTables, field_key: Mapping, raw_id: Optional[int] = None,
                 preprocess_id: Optional[int] = None, movie_loader: Optional[Callable] = None,
                 data_roots: tuple = (), label: Optional[str] = None):
        self.tables = tables
        self.field_key = dict(field_key)
        self.preprocess_id = preprocess_id
        self.movie_loader = movie_loader or (lambda *args: _movie_from_file(tables, *args))
        self.data_roots = tuple(data_roots)
        self.notes = ["database mode: traces, triggers, masks and QI are the stored values (read-only)"]
        self.label = label or ' '.join(f"{k}={v}" for k, v in self.field_key.items())

        restriction = dict(self.field_key, **({} if raw_id is None else {'raw_id': raw_id}))
        pres_keys = list((tables.presentation & restriction).keys())
        if len(pres_keys) == 0:
            raise ValueError(f"no Presentation for {restriction}")
        raw_ids = sorted({k['raw_id'] for k in pres_keys if 'raw_id' in k})
        if len(raw_ids) > 1:
            raise ValueError(f"several raw_id {raw_ids} for this field; pass raw_id=... to choose one")
        self._infos = {}
        for key in pres_keys:
            info = self._make_info(key)
            self._infos[info.name] = info

    # --------------------------------------------------------------- listing
    def _make_info(self, key: Mapping) -> PresentationInfo:
        t = self.tables
        stim_name = key['stim_name']
        stim_family = ''
        if t.stimulus is not None:
            stim_family = (t.stimulus & dict(stim_name=stim_name)).fetch1('stim_family') or ''
        extra = [f"{k}={v}" for k, v in key.items() if k not in self.field_key and k not in ('stim_name', 'raw_id')]
        name = stim_name + (f"  [{', '.join(extra)}]" if extra else '')
        has_mask = len(t.roi_mask.RoiMaskPresentation & key) > 0
        notes = [] if has_mask else ["no RoiMask.RoiMaskPresentation entry: no ROIs or traces for this presentation"]
        return PresentationInfo(name=name, key=dict(key), stim_name=stim_name, stim_family=stim_family,
                                channels=tuple(self._channels(key)), has_mask=has_mask, notes=tuple(notes))

    def _channels(self, key: Mapping) -> list:
        t = self.tables
        if 'ch_id' in t.traces.primary_key:
            ch_ids = sorted({int(c) for c in (t.traces & key).to_arrays('ch_id')})
            if t.channel is None:
                return [f'wDataCh{c}' for c in ch_ids]
            return [(t.channel & dict(ch_id=c)).fetch1('ch_name') for c in ch_ids]
        if t.userinfo is not None:
            return list(t.userinfo().get_functional_ch_names(key))
        return ['wDataCh0']

    def _ch_restriction(self, channel: str) -> dict:
        t = self.tables
        if 'ch_id' not in t.traces.primary_key:
            return {}
        if t.channel is not None:
            return dict(ch_id=int((t.channel & dict(ch_name=channel)).fetch1('ch_id')))
        return dict(ch_id=int(channel.replace('wDataCh', '')))

    def list_presentations(self) -> list:
        return list(self._infos.values())

    def _pick_preprocess_id(self, table, restriction) -> Optional[int]:
        ids = sorted({int(i) for i in (table & restriction).to_arrays('preprocess_id')})
        if self.preprocess_id is not None:
            return self.preprocess_id
        if len(ids) > 1:
            raise ValueError(f"several preprocess_id {ids}; pass preprocess_id=... to choose one")
        return ids[0] if ids else None

    # --------------------------------------------------------------- loading
    def load_presentation(self, name: str, channel: str = 'wDataCh0') -> PresentationData:
        t = self.tables
        from djimaging.utils.dj_storage import load_array
        info = self._infos[name]
        key = dict(info.key)
        notes = list(info.notes)
        pres_data_file, triggertimes, npixartifact, pixel_size_um, absx, absy, nxpix, nypix, scan_type = \
            (t.presentation & key).fetch1('pres_data_file', 'triggertimes', 'npixartifact', 'pixel_size_um',
                                          'absx', 'absy', 'nxpix', 'nypix', 'scan_type')
        meta = dict(absx=absx, absy=absy, nxpix=nxpix, nypix=nypix, scan_type=scan_type, file=str(pres_data_file))
        if hasattr(t.presentation, 'ScanInfo') and len(t.presentation.ScanInfo & key) == 1:
            meta['zoom'], meta['angle_deg'] = (t.presentation.ScanInfo & key).fetch1('zoom', 'angle_deg')
        triggers = np.asarray(load_array(triggertimes), dtype=float).ravel()

        stim = None
        if t.stimulus is not None:
            rows = (t.stimulus & dict(stim_name=info.stim_name)).to_dicts()
            if len(rows) != 1:
                raise ValueError(f"expected one Stimulus entry for {info.stim_name!r}, found {len(rows)}")
            stim = {k: rows[0].get(k) for k in ('isrepeated', 'ntrigger_rep', 'trial_info', 'stim_dict')}
        # Snippets cut repeats with its own _pad_trace (template default False); use the same for the trial average
        pad_trace = bool(getattr(t.snippets, '_pad_trace', False)) if t.snippets is not None else False
        if t.snippets is None:
            notes.append("Snippets table not given: trial average uses pad_trace=False (SnippetsTemplate default)")

        from_raw_data, precision = str(pres_data_file).lower().endswith('.smp'), 'line'
        if t.raw_params is not None:
            from_raw_data, precision = (t.raw_params & key).fetch1('from_raw_data', 'trace_precision')
        stack, frame_times, frame_dt = self.movie_loader(pres_data_file, channel, bool(from_raw_data), precision)
        if getattr(t.traces, '_include_motion_correction', False):
            notes.append("Traces are motion corrected; the movie is shown without motion correction")

        rois, mask_info, signals, roi_notes = None, {}, {}, {}
        quality, quality_unavailable = None, ''
        if info.has_mask:
            roi_mask, as_field_mask, shift_dx, shift_dy = (t.roi_mask.RoiMaskPresentation & key).fetch1(
                'roi_mask', 'as_field_mask', 'shift_dx', 'shift_dy')
            rois = RoiLabels.from_igor_xy(load_array(roi_mask))
            mask_info = dict(as_field_mask=as_field_mask, shift_dx=int(shift_dx), shift_dy=int(shift_dy))
            ch = self._ch_restriction(channel)
            signals, roi_notes = self._signals(key, ch, rois, channel, frame_dt, notes)
            quality, quality_unavailable = self._quality(info, key, ch)
        else:
            quality_unavailable = "no ROI mask for this presentation"

        return PresentationData(
            info=info, channel=channel, frames=to_display(stack).astype(np.float32), frame_times=frame_times,
            frame_dt=frame_dt, rois=rois, signals=signals, triggers=triggers,
            trigger_labels=trigger_labels_for(triggers, stim), stimulus=stimulus_info_for(stim, pad_trace),
            quality=quality,
            quality_unavailable=quality_unavailable, roi_notes=roi_notes, mask_info=mask_info,
            pixel_size_um=float(pixel_size_um), npixartifact=int(npixartifact), meta=meta, notes=tuple(notes),
            source='database (read-only)')

    def _signals(self, key, ch, rois, channel, frame_dt, notes):
        t = self.tables
        from djimaging.utils.dj_storage import load_array
        roi_ids = set(rois.roi_ids.tolist())
        raw, roi_notes = {}, {}
        for row in (t.traces & key & ch).to_dicts():
            roi_id = int(row['roi_id'])
            if roi_id not in roi_ids:
                raise ValueError(f"Traces has ROI {roi_id} that is not in the presentation mask of {key}")
            values = np.asarray(load_array(row['trace']), dtype=float)
            raw[roi_id] = Trace(sample_times(row['trace_t0'], row['trace_dt'], values.size), values)
            if not np.isclose(row['trace_dt'], frame_dt, rtol=1e-4):
                notes.append(f"ROI {roi_id}: trace_dt {row['trace_dt']:.6f} != movie frame_dt {frame_dt:.6f}")
            if not row.get('trace_valid', 1):
                roi_notes[roi_id] = 'trace_valid=0'
            elif not row.get('trigger_valid', 1):
                roi_notes[roi_id] = 'trigger_valid=0'
        for r in roi_ids - set(raw):
            roi_notes.setdefault(r, 'no Traces entry (e.g. light artifact with include_artifacts=0)')
        signals = {}
        if raw:
            signals['raw'] = Signal('raw', f'a.u. (ROI mean of {channel})', raw, 'Traces.trace (database)')
        if t.preprocess_traces is not None:
            pid = self._pick_preprocess_id(t.preprocess_traces, dict(key, **ch))
            if pid is not None:
                pp = {}
                for row in (t.preprocess_traces & key & ch & dict(preprocess_id=pid)).to_dicts():
                    values = np.asarray(load_array(row['pp_trace']), dtype=float)
                    pp[int(row['roi_id'])] = Trace(sample_times(row['pp_trace_t0'], row['pp_trace_dt'], values.size),
                                                   values)
                if pp:
                    signals['preprocessed'] = Signal('preprocessed', f'preprocessed (preprocess_id={pid})', pp,
                                                     f'PreprocessTraces.pp_trace, preprocess_id={pid} (database)')
        return signals, roi_notes

    def _quality(self, info: PresentationInfo, key, ch):
        t = self.tables
        family = quality_family(info.stim_name, info.stim_family)
        if family is None:
            return None, unsupported_reason(info.stim_name, info.stim_family)
        table_name, metric = QUALITY_METRICS[family]
        table = t.chirp_qi if family == 'chirp' else t.os_ds_indexes
        if table is None:
            return None, f"{table_name} is not in this schema (add the 'basic' block) or was not passed"
        pid = self._pick_preprocess_id(table, dict(key, **ch))
        rows = [] if pid is None else (table & key & ch & dict(preprocess_id=pid)).to_dicts()
        values = {int(r['roi_id']): float(r[metric]) for r in rows}
        extra = {}
        if family == 'movingbar':
            extra = {m: {int(r['roi_id']): float(r[m]) for r in rows} for m in ('ds_index', 'os_index')}
        reference = float(rows[0]['min_qidx']) if rows and 'min_qidx' in rows[0] else None
        source = f"database {table_name}" + ('' if pid is None else f", preprocess_id={pid}")
        if not rows:
            source += " (not populated for this presentation: all values missing)"
        return QualityValues(table_name, metric, values, source, reference, extra), ''

    def load_highres(self, channel: str = 'wDataCh0') -> tuple[Optional[HighResImage], str]:
        t = self.tables
        if t.highres is None:
            return None, "HighRes table not given"
        from djimaging.utils.dj_storage import load_array
        rows = (t.highres.StackAverages & self.field_key & dict(ch_name=channel)).to_dicts()
        if len(rows) == 0:
            return None, f"no HighRes.StackAverages entry for channel {channel} of this field"
        row = rows[0]
        hr_key = {k: row[k] for k in t.highres.primary_key if k in row}
        absx, absy, nxpix, nypix, pixel_size_um, scan_type = (t.highres & hr_key).fetch1(
            'absx', 'absy', 'nxpix', 'nypix', 'pixel_size_um', 'scan_type')
        meta = dict(absx=absx, absy=absy, nxpix=nxpix, nypix=nypix, scan_type=scan_type)
        image = to_display(np.asarray(load_array(row['ch_average']), dtype=float))
        return HighResImage(image, channel, float(pixel_size_um), meta, source=f"HighRes {hr_key}"), \
            ('' if len(rows) == 1 else f"{len(rows)} HighRes entries, using {hr_key}")
