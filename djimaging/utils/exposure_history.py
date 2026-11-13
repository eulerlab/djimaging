"""Light exposure history of an experiment: when and where the retina was exposed to the laser and the stimulator.

Built from the experiment history (`ExpHistory`, `ExpHistoryField`, `ExpHistoryStimLog`) to study adaptation effects
(e.g. field 2 saw the stimuli of fields 0 and 1). Everything here works on plain DataFrames, the DataJoint fetching is
in `djimaging.utils.exposure_history_dj`.

Tables
------
- **events**: one row per exposure event, in time order on the ScanM clock.
  `event='laser_scan'`: a ScanM recording (laser exposure of the scan area).
  `event='led_stimulus'`: a stimulus of the stimulator, from the QDSpy log (`time_source='qdspy+offset'`) or, for
  response recordings without a logged stimulus, from the stimulus part of the file name
  (`time_source='recording'`, assumed to run for the whole recording).
- **prior**: one row per recording, the exposure before the recording started (total, at this field, elsewhere).
- **meta**: dict with experiment-level information (setup, stimulator, calibration, clock offset) and the known gaps.

Gaps are explicit: values that are not known are NaN/None here and written as `unknown` (and `n/a` where a value does
not apply to the event type). Each row has an `incomplete` column with `;`-separated reasons.

Position of a stimulus: the stimulator projects through the objective, so a stimulus is centered on the current scan
field. Without a simultaneous recording, the position is taken from the next recording
(`position_source='next_recording'`); this is an assumption and is tagged as such.
"""
from __future__ import annotations

import html
import json
import os
from collections.abc import Iterable, Mapping
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd

UNKNOWN = 'unknown'
NOT_APPLICABLE = 'n/a'

ALWAYS_GAPS = (
    'Laser power is not recorded digitally (only on paper).',
    'Focus and live scans that were not saved are not registered.',
    'Stimulator intensity is given in input units (0-255) per LED; see the calibration of the stimulator.',
)

EVENT_COLUMNS = [
    'event', 't_start', 't_end', 't_start_rel_s', 't_end_rel_s', 'dur_s', 'time_source',
    'rec_name', 'kind', 'field', 'field_source', 'stimulus', 'stim_name', 'listed', 'aborted',
    'x_um', 'y_um', 'z_um', 'position_source',
    'scan_type', 'zoom', 'scan_width_um', 'scan_height_um', 'scan_depth_um', 'pixel_dwell_us',
    'mean_uv', 'mean_green', 'extent_x_um', 'extent_y_um', 'stim_md5', 'log_idx', 'stim_idx',
    'has_presentation', 'n_traces', 'motion_slow_range_px', 'motion_trust_median', 'motion_n_rois_flagged',
    'incomplete',
]

NUMERIC_EVENT_COLUMNS = [
    'dur_s', 'x_um', 'y_um', 'z_um', 'zoom', 'scan_width_um', 'scan_height_um', 'scan_depth_um', 'pixel_dwell_us',
    'mean_uv', 'mean_green', 'extent_x_um', 'extent_y_um', 'log_idx', 'stim_idx', 'n_traces',
    'motion_slow_range_px', 'motion_trust_median', 'motion_n_rois_flagged']

# Columns that do not apply to one of the event types: written as `n/a` instead of `unknown`
_LASER_ONLY = ['kind', 'scan_type', 'zoom', 'scan_width_um', 'scan_height_um', 'scan_depth_um', 'pixel_dwell_us',
               'has_presentation', 'n_traces', 'motion_slow_range_px', 'motion_trust_median',
               'motion_n_rois_flagged']
_LED_ONLY = ['mean_uv', 'mean_green', 'extent_x_um', 'extent_y_um', 'stim_md5', 'log_idx', 'stim_idx']

PRIOR_COLUMNS = [
    'rec_name', 'field', 'kind', 'stimulus', 'listed', 't_start', 't_start_rel_s',
    'prior_laser_s', 'prior_laser_s_this_field', 'prior_laser_s_other_fields', 'prior_laser_s_unknown_field',
    'prior_led_s', 'prior_led_s_this_field', 'prior_led_s_this_field_inferred', 'prior_led_s_other_fields',
    'prior_led_s_unknown_field', 'prior_led_uv_dose', 'prior_led_green_dose', 'prior_led_s_levels_unknown',
    'prior_fields', 'prior_stimuli_this_field', 'incomplete',
]


# ----------------------------------------------------------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------------------------------------------------------
def build_events(recordings: pd.DataFrame, stims: pd.DataFrame | None = None, clock_offset_s: float | None = None,
                 exp_date: date | None = None, light_levels: pd.DataFrame | None = None) -> pd.DataFrame:
    """All laser and stimulator exposure events of one experiment in time order.

    Parameters
    ----------
    recordings : pd.DataFrame
        One row per recording (`ExpHistory.Recording`). Required: `rec_name`, `rec_start` (datetime or None),
        `rec_dur_s`. Optional: `kind`, `field` (field label, None if not linked to a field), `stim_token`,
        `stim_name` (listed stimulus names joined by `;`, empty or None if unlisted), `listed` (bool, None: unknown),
        `stopped_early`, `absx`, `absy`, `absz`, `zoom`, `pixel_size_um`, `nxpix`, `nypix`, `nzpix`, `z_step_um`,
        `real_pix_dur_us`, `scan_type`, `header_issue`, and the processing status `has_presentation`, `n_traces`,
        `motion_slow_range_px`, `motion_trust_median`, `motion_n_rois_flagged`.
    stims : pd.DataFrame | None
        One row per logged stimulus (`ExpHistoryStimLog.Stim` joined with the QDSpy log). Required: `t_start`
        (time of day, QDSpy clock), `t_dur_s`, `rec_name` (recording it started in, None if none). Optional:
        `t_end`, `stim_file_name`, `stim_md5`, `stim_name` (Stimulus name if listed), `aborted`, `log_idx`,
        `stim_idx`. None: no QDSpy log for this experiment.
    clock_offset_s : float | None
        Add to QDSpy times to get ScanM times (`ExpHistoryStimLog.clock_offset_s`). None: unknown, the QDSpy times
        are used unchanged and tagged.
    exp_date : date | None
        Date of the QDSpy times. Default: date of the first recording start.
    light_levels : pd.DataFrame | None
        Mean light level and extent per stimulus, see `lookup_light_levels`.

    Returns
    -------
    pd.DataFrame
        Columns `EVENT_COLUMNS`, sorted by `t_start` (events without time last).
    """
    recs = _prepare_recordings(recordings)
    if exp_date is None:
        starts = recs['rec_start'].dropna()
        exp_date = starts.min().date() if len(starts) > 0 else None

    rows = [_laser_row(r) for r in recs.to_dict('records')]

    logged_recs = set()
    if stims is not None and len(stims) > 0:
        for s in stims.to_dict('records'):
            rows.append(_logged_stim_row(s, recs, clock_offset_s, exp_date, light_levels))
            if _notna(s.get('rec_name')):
                logged_recs.add(s['rec_name'])

    # Response recordings without a logged stimulus: the stimulus of the file name was (presumably) shown
    for r in recs.to_dict('records'):
        if r.get('kind') == 'response' and _notna(r.get('stim_token')) and r['rec_name'] not in logged_recs:
            rows.append(_file_name_stim_row(r, has_log=stims is not None, light_levels=light_levels))

    events = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    events = _sort_events(events)
    t0 = events['t_start'].dropna().min() if events['t_start'].notna().any() else None
    events['t_start_rel_s'] = [_rel_s(t, t0) for t in events['t_start']]
    events['t_end_rel_s'] = [_rel_s(t, t0) for t in events['t_end']]
    return events.reset_index(drop=True)


def _prepare_recordings(recordings: pd.DataFrame) -> pd.DataFrame:
    recs = recordings.copy()
    for col in ['kind', 'field', 'stim_token', 'stim_name', 'listed', 'stopped_early', 'absx', 'absy', 'absz', 'zoom',
                'pixel_size_um', 'nxpix', 'nypix', 'nzpix', 'z_step_um', 'real_pix_dur_us', 'scan_type',
                'header_issue', 'has_presentation', 'n_traces', 'motion_slow_range_px', 'motion_trust_median',
                'motion_n_rois_flagged']:
        if col not in recs:
            recs[col] = None
    recs['rec_start'] = pd.to_datetime(recs['rec_start'])
    recs = recs.astype(object).where(recs.notna(), None)
    recs['rec_start'] = pd.Series([None if v is None else pd.Timestamp(v).to_pydatetime() for v in recs['rec_start']],
                                  index=recs.index, dtype=object)
    starts = [v if v is not None else datetime.max for v in recs['rec_start']]
    return recs.iloc[np.argsort(starts, kind='stable')].reset_index(drop=True)


def _laser_row(r: dict) -> dict:
    incomplete = []
    t_start = r.get('rec_start')
    dur = _float(r.get('rec_dur_s'))
    if t_start is None:
        incomplete.append('start_time')
    if dur is None:
        incomplete.append('duration')
    t_end = t_start + timedelta(seconds=dur) if (t_start is not None and dur is not None) else None
    if r.get('field') is None:
        incomplete.append('field')
    pos = [_float(r.get(k)) for k in ('absx', 'absy', 'absz')]
    if any(p is None for p in pos):
        incomplete.append('position')
    pixel_size = _float(r.get('pixel_size_um'))
    width = _mul(pixel_size, r.get('nxpix'))
    height = _mul(pixel_size, r.get('nypix')) if r.get('scan_type') in ('xy', 'xyz') else None
    depth = _mul(_float(r.get('z_step_um')), r.get('nzpix')) if r.get('scan_type') in ('xz', 'xyz') else None
    if r.get('scan_type') == 'xz':
        height = None
    if width is None:
        incomplete.append('scan_area')
    if r.get('stopped_early'):
        incomplete.append('stopped_early')
    if r.get('header_issue'):
        incomplete.append('header_issue')
    listed = _listed_label(r.get('listed'), r.get('kind'))

    has_pres = r.get('has_presentation')
    if listed == 'listed' and not has_pres:
        incomplete.append('no_presentation' if has_pres is False else 'processing_unknown')

    return dict(
        event='laser_scan', t_start=t_start, t_end=t_end, dur_s=dur, time_source='scanm' if t_start else UNKNOWN,
        rec_name=r['rec_name'], kind=r.get('kind'), field=r.get('field'),
        field_source='recording' if r.get('field') is not None else UNKNOWN,
        stimulus=r.get('stim_token'), stim_name=r.get('stim_name') if _notna(r.get('stim_name')) else None,
        listed=listed,
        aborted=r.get('stopped_early'),
        x_um=pos[0], y_um=pos[1], z_um=pos[2], position_source='recording' if None not in pos else UNKNOWN,
        scan_type=r.get('scan_type'), zoom=_float(r.get('zoom')), scan_width_um=width, scan_height_um=height,
        scan_depth_um=depth, pixel_dwell_us=_float(r.get('real_pix_dur_us')),
        has_presentation=has_pres if listed == 'listed' else None,
        n_traces=r.get('n_traces'), motion_slow_range_px=_float(r.get('motion_slow_range_px')),
        motion_trust_median=_float(r.get('motion_trust_median')),
        motion_n_rois_flagged=r.get('motion_n_rois_flagged'),
        incomplete=';'.join(incomplete))


def _logged_stim_row(s: dict, recs: pd.DataFrame, clock_offset_s: float | None, exp_date: date | None,
                     light_levels: pd.DataFrame | None) -> dict:
    incomplete = []
    sod = _seconds_of_day(s.get('t_start'))
    dur = _float(s.get('t_dur_s'))
    if dur is not None and dur < 0:
        dur = None
    if dur is None:
        sod_end = _seconds_of_day(s.get('t_end'))
        dur = sod_end - sod if (sod is not None and sod_end is not None and sod_end >= sod) else None
    if dur is None:
        incomplete.append('duration')

    if sod is None or exp_date is None:
        t_start, time_source = None, UNKNOWN
        incomplete.append('start_time')
    else:
        t_start = datetime.combine(exp_date, time()) + timedelta(seconds=sod + (clock_offset_s or 0.))
        time_source = 'qdspy+offset' if clock_offset_s is not None else 'qdspy_unaligned'
        if clock_offset_s is None:
            incomplete.append('clock_offset')
    t_end = t_start + timedelta(seconds=dur) if (t_start is not None and dur is not None) else None

    rec = _recording_for_stim(s.get('rec_name'), t_start, recs)
    if rec is None:
        position_source = UNKNOWN
        incomplete.append('position')
    else:
        position_source = 'simultaneous_recording' if rec['rec_name'] == s.get('rec_name') else 'next_recording'
    if rec is not None and position_source == 'next_recording':
        incomplete.append('position_from_next_recording')
    field = rec.get('field') if rec is not None else None
    if field is None:
        incomplete.append('field')

    levels = lookup_light_levels(light_levels, md5=s.get('stim_md5'), file_name=s.get('stim_file_name'),
                                 stim_name=s.get('stim_name'))
    if levels is None or levels.get('mean_uv') is None or levels.get('mean_green') is None:
        incomplete.append('light_levels')
    levels = levels or {}

    return dict(
        event='led_stimulus', t_start=t_start, t_end=t_end, dur_s=dur, time_source=time_source,
        rec_name=s.get('rec_name') if _notna(s.get('rec_name')) else None,
        field=field, field_source=position_source if field is not None else UNKNOWN,
        stimulus=s.get('stim_file_name'), stim_name=s.get('stim_name') if _notna(s.get('stim_name')) else None,
        listed='listed' if _notna(s.get('stim_name')) else 'unlisted', aborted=_bool(s.get('aborted')),
        x_um=_float(rec.get('absx')) if rec is not None else None,
        y_um=_float(rec.get('absy')) if rec is not None else None,
        z_um=_float(rec.get('absz')) if rec is not None else None,
        position_source=position_source,
        mean_uv=levels.get('mean_uv'), mean_green=levels.get('mean_green'),
        extent_x_um=levels.get('extent_x_um'), extent_y_um=levels.get('extent_y_um'),
        stim_md5=s.get('stim_md5'), log_idx=s.get('log_idx'), stim_idx=s.get('stim_idx'),
        incomplete=';'.join(incomplete))


def _file_name_stim_row(r: dict, has_log: bool, light_levels: pd.DataFrame | None) -> dict:
    incomplete = ['stimulus_from_file_name']
    if has_log:
        incomplete.append('not_in_log')
    t_start = r.get('rec_start')
    dur = _float(r.get('rec_dur_s'))
    if t_start is None:
        incomplete.append('start_time')
    if dur is None:
        incomplete.append('duration')
    stim_names = [n for n in str(r.get('stim_name') or '').split(';') if n]
    levels = lookup_light_levels(light_levels, stim_name=stim_names[0] if len(stim_names) == 1 else None,
                                 file_name=r.get('stim_token'))
    if levels is None or levels.get('mean_uv') is None or levels.get('mean_green') is None:
        incomplete.append('light_levels')
    levels = levels or {}
    pos = [_float(r.get(k)) for k in ('absx', 'absy', 'absz')]
    if any(p is None for p in pos):
        incomplete.append('position')
    if r.get('field') is None:
        incomplete.append('field')
    return dict(
        event='led_stimulus', t_start=t_start,
        t_end=t_start + timedelta(seconds=dur) if (t_start is not None and dur is not None) else None,
        dur_s=dur, time_source='recording' if t_start is not None else UNKNOWN,
        rec_name=r['rec_name'], field=r.get('field'),
        field_source='simultaneous_recording' if r.get('field') is not None else UNKNOWN,
        stimulus=r.get('stim_token'), stim_name=r.get('stim_name') if _notna(r.get('stim_name')) else None,
        listed=_listed_label(r.get('listed'), r.get('kind')), aborted=r.get('stopped_early'),
        x_um=pos[0], y_um=pos[1], z_um=pos[2],
        position_source='simultaneous_recording' if None not in pos else UNKNOWN,
        mean_uv=levels.get('mean_uv'), mean_green=levels.get('mean_green'),
        extent_x_um=levels.get('extent_x_um'), extent_y_um=levels.get('extent_y_um'),
        incomplete=';'.join(incomplete))


def _recording_for_stim(rec_name, t_start: datetime | None, recs: pd.DataFrame) -> dict | None:
    """The simultaneous recording, else the next recording (by start time) with a known position."""
    if _notna(rec_name):
        match = recs[recs['rec_name'] == rec_name]
        if len(match) > 0:
            return match.iloc[0].to_dict()
    if t_start is None:
        return None
    for r in recs.to_dict('records'):
        if r['rec_start'] is not None and r['rec_start'] >= t_start and r.get('absx') is not None:
            return r
    return None


def lookup_light_levels(light_levels: pd.DataFrame | None, md5: str | None = None, file_name: str | None = None,
                        stim_name: str | None = None) -> dict | None:
    """Mean light level and extent of a stimulus.

    Parameters
    ----------
    light_levels : pd.DataFrame | None
        One row per stimulus with `match_md5`, `match_file_name`, `match_stim_name` (each may be None) and
        `mean_uv`, `mean_green` (time-averaged mean of the LED input in 0..255), `extent_x_um`, `extent_y_um`
        (size of the stimulated area on the retina; None: unknown).
    md5, file_name, stim_name : str | None
        QDSpy md5, QDSpy file name (or stimulus part of the file name), Stimulus name. Matched in this order;
        names are compared case-insensitively.

    Returns
    -------
    dict | None
        The values of the first matching row (NaN as None), None if no row matches.
    """
    if light_levels is None or len(light_levels) == 0:
        return None
    for col, value in (('match_md5', md5), ('match_file_name', file_name), ('match_stim_name', stim_name)):
        if col not in light_levels or not _notna(value):
            continue
        values = light_levels[col].astype(object).where(light_levels[col].notna(), None)
        hits = [i for i, v in enumerate(values) if v is not None and str(v).lower() == str(value).lower()]
        if hits:
            row = light_levels.iloc[hits[0]]
            return {k: (None if not _notna(row.get(k)) else float(row.get(k)))
                    for k in ('mean_uv', 'mean_green', 'extent_x_um', 'extent_y_um')}
    return None


def summarize_prior_exposure(events: pd.DataFrame) -> pd.DataFrame:
    """Exposure before each recording started: laser and stimulator time in s (total, this field, other fields,
    unknown field), the light dose of the stimulator (mean LED input x time, in input units x s), the fields
    scanned before and the stimuli shown at this field before.

    Events that overlap the recording start are counted up to the start. Events without start time are not counted
    (tag `events_without_time`), stimulator events with unknown light level are not part of the dose
    (`prior_led_s_levels_unknown`), and stimulator events with a position from the next recording are counted
    separately for this field (`prior_led_s_this_field_inferred`, also part of `prior_led_s_this_field`).
    """
    rows = []
    timed = events[events['t_start'].notna()]
    n_untimed = int(events['t_start'].isna().sum())
    for rec in events[events['event'] == 'laser_scan'].to_dict('records'):
        row = {k: rec.get(k) for k in ('rec_name', 'field', 'kind', 'stimulus', 'listed', 't_start',
                                       't_start_rel_s')}
        incomplete = []
        if rec['t_start'] is None or pd.isna(rec['t_start']):
            rows.append({**row, 'incomplete': 'start_time'})
            continue
        t_rec = rec['t_start']
        before = timed[timed['t_start'] < t_rec]
        durs = np.array([_clipped_dur(e, t_rec) for e in before.to_dict('records')], dtype=float)
        if np.any(np.isnan(durs)):
            incomplete.append('duration')
        durs = np.nan_to_num(durs)
        is_laser = (before['event'] == 'laser_scan').to_numpy()
        is_led = ~is_laser
        field = rec.get('field')
        fields = before['field'].to_numpy(dtype=object)
        unknown_field = np.array([not _notna(f) for f in fields], dtype=bool)
        if field is None or not _notna(field):
            incomplete.append('field')
            this_field = np.zeros(len(before), dtype=bool)
            other_field = np.zeros(len(before), dtype=bool)
            unknown_field = np.ones(len(before), dtype=bool)
        else:
            this_field = np.array([_notna(f) and f == field for f in fields], dtype=bool)
            other_field = ~this_field & ~unknown_field
        inferred = (before['position_source'] == 'next_recording').to_numpy()

        uv = pd.to_numeric(before['mean_uv'], errors='coerce').to_numpy(dtype=float)
        green = pd.to_numeric(before['mean_green'], errors='coerce').to_numpy(dtype=float)
        levels_unknown = is_led & (np.isnan(uv) | np.isnan(green))

        row.update(
            prior_laser_s=durs[is_laser].sum(),
            prior_laser_s_this_field=durs[is_laser & this_field].sum(),
            prior_laser_s_other_fields=durs[is_laser & other_field].sum(),
            prior_laser_s_unknown_field=durs[is_laser & unknown_field].sum(),
            prior_led_s=durs[is_led].sum(),
            prior_led_s_this_field=durs[is_led & this_field].sum(),
            prior_led_s_this_field_inferred=durs[is_led & this_field & inferred].sum(),
            prior_led_s_other_fields=durs[is_led & other_field].sum(),
            prior_led_s_unknown_field=durs[is_led & unknown_field].sum(),
            prior_led_uv_dose=float(np.nansum(uv[is_led] * durs[is_led])),
            prior_led_green_dose=float(np.nansum(green[is_led] * durs[is_led])),
            prior_led_s_levels_unknown=durs[levels_unknown].sum(),
            prior_fields=';'.join(_unique([f for f, la in zip(fields, is_laser) if la and _notna(f)])),
            prior_stimuli_this_field=';'.join(
                f"{e['stimulus']}{'*' if e['position_source'] == 'next_recording' else ''}"
                for e, m in zip(before.to_dict('records'), is_led & this_field) if m),
        )
        if row['prior_led_s_levels_unknown'] > 0:
            incomplete.append('light_levels')
        if row['prior_led_s_this_field_inferred'] > 0:
            incomplete.append('position_from_next_recording')
        if row['prior_led_s_unknown_field'] > 0 or row['prior_laser_s_unknown_field'] > 0:
            incomplete.append('events_without_field')
        if n_untimed > 0:
            incomplete.append('events_without_time')
        if (before['time_source'] == 'qdspy_unaligned').any():
            incomplete.append('clock_offset')
        row['incomplete'] = ';'.join(incomplete)
        rows.append(row)
    return pd.DataFrame(rows, columns=PRIOR_COLUMNS)


def experiment_gaps(events: pd.DataFrame, meta: Mapping) -> list[str]:
    """Experiment-level gaps: always-known limits and the gaps found in this experiment."""
    gaps = list(ALWAYS_GAPS)
    if not meta.get('has_stim_log'):
        gaps.append('No QDSpy log: stimuli are taken from the file names of the recordings only.')
    elif meta.get('clock_offset_s') is None:
        gaps.append('Clock offset between QDSpy and ScanM unknown: stimulus times are not aligned.')
    if meta.get('stimulator_id') is None:
        gaps.append('No stimulator assigned: geometry, color map and calibration unknown.')
    elif meta.get('stimulator_confidence') not in (None, 'high'):
        gaps.append(f"Stimulator confidence {meta['stimulator_confidence']}: {meta.get('stimulator_reason')}")
    if meta.get('unscanned_files'):
        gaps.append(f"{len(meta['unscanned_files'])} ScanM file(s) in other folders are not registered.")
    led = events[events['event'] == 'led_stimulus']
    laser = events[events['event'] == 'laser_scan']
    counts = [
        (int(events['t_start'].isna().sum()), 'event(s) without start time'),
        (int((laser['field'].isna()).sum()), 'recording(s) not linked to a field'),
        (int((led['position_source'] == 'next_recording').sum()),
         'stimulus/stimuli without simultaneous recording (position from the next recording)'),
        (int((led['position_source'] == UNKNOWN).sum()), 'stimulus/stimuli with unknown position'),
        (int(led['incomplete'].str.contains('light_levels').sum()), 'stimulus/stimuli with unknown light level'),
        (int((led['time_source'] == 'recording').sum()),
         'stimulus/stimuli only known from the file name (timing = recording)'),
    ]
    gaps += [f"{n} {text}." for n, text in counts if n > 0]
    return gaps


# ----------------------------------------------------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------------------------------------------------
def to_display(df: pd.DataFrame) -> pd.DataFrame:
    """Copy with `unknown` for missing values and `n/a` for values that do not apply to the event type."""
    out = df.astype(object).where(df.notna(), None)
    if 'event' in out:
        for cols, event in ((_LASER_ONLY, 'led_stimulus'), (_LED_ONLY, 'laser_scan')):
            mask = out['event'] == event
            for col in cols:
                if col in out:
                    out.loc[mask, col] = NOT_APPLICABLE
    if 'kind' in out:  # recordings without a stimulus
        no_stim = out['kind'].isin(['hr', 'od', 'outline'])
        for col in ('listed', 'stimulus', 'stim_name'):
            if col in out:
                out.loc[no_stim & out[col].isna(), col] = NOT_APPLICABLE
    for col in out.columns:
        if col == 'incomplete':
            out[col] = out[col].fillna('')
        else:
            out[col] = out[col].apply(lambda v: UNKNOWN if v is None else v)
    return out


def write_csv(out_dir: str, events: pd.DataFrame, prior: pd.DataFrame, meta: Mapping) -> dict:
    """Write `exposure_events.csv`, `prior_exposure.csv` and `experiment_meta.json` to `out_dir` (created).
    Returns the file paths."""
    os.makedirs(out_dir, exist_ok=True)
    files = dict(events=os.path.join(out_dir, 'exposure_events.csv'),
                 prior=os.path.join(out_dir, 'prior_exposure.csv'),
                 meta=os.path.join(out_dir, 'experiment_meta.json'))
    to_display(events).to_csv(files['events'], index=False)
    to_display(prior).to_csv(files['prior'], index=False)
    with open(files['meta'], 'w', encoding='utf-8') as f:
        json.dump(dict(meta), f, indent=2, default=str)
    return files


def read_csv(out_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Read the files of `write_csv`; `unknown` and `n/a` become NaN."""
    na = [UNKNOWN, NOT_APPLICABLE]
    events = pd.read_csv(os.path.join(out_dir, 'exposure_events.csv'), na_values=na, keep_default_na=False)
    prior = pd.read_csv(os.path.join(out_dir, 'prior_exposure.csv'), na_values=na, keep_default_na=False)
    for df in (events, prior):
        for col in ('t_start', 't_end'):
            if col in df:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        if 'incomplete' in df:
            df['incomplete'] = df['incomplete'].fillna('').astype(str)
    with open(os.path.join(out_dir, 'experiment_meta.json'), encoding='utf-8') as f:
        meta = json.load(f)
    return events, prior, meta


def write_nwb(nwbfile, events: pd.DataFrame, meta: Mapping | None = None, name: str = 'light_exposure_history',
              t_ref: datetime | None = None, this_field: str | None = None):
    """Add the events as a `TimeIntervals` table to an NWB file (needs `pynwb`).

    `start_time`/`stop_time` are seconds since `t_ref` (default: the first event, as `t_start_rel_s`); NaN if
    unknown. Missing text values are `unknown`/`n/a` as in the CSV, missing numbers NaN. `meta` is stored as JSON in
    the description. `this_field`: add a boolean column `is_this_field` (e.g. for one NWB file per field).
    """
    from pynwb.epoch import TimeIntervals

    disp = to_display(events)
    offset = 0.
    if t_ref is not None and events['t_start'].notna().any():
        offset = (events['t_start'].dropna().min() - t_ref).total_seconds()
    start = pd.to_numeric(events['t_start_rel_s'], errors='coerce').to_numpy(dtype=float) + offset
    stop = pd.to_numeric(events['t_end_rel_s'], errors='coerce').to_numpy(dtype=float) + offset

    description = 'Light exposure history (laser scans and stimulator), see djimaging.utils.exposure_history.'
    if meta is not None:
        description += ' meta=' + json.dumps(dict(meta), default=str)
    table = TimeIntervals(name=name, description=description)
    cols = [c for c in EVENT_COLUMNS if c not in ('t_start_rel_s', 't_end_rel_s')]
    numeric = set(NUMERIC_EVENT_COLUMNS)
    for c in cols:
        table.add_column(name=c, description=c)
    if this_field is not None:
        table.add_column(name='is_this_field', description=f'event at field {this_field}')
    for i in range(len(events)):
        row = {}
        for c in cols:
            if c in numeric:
                row[c] = float(events[c].iloc[i]) if _notna(events[c].iloc[i]) else np.nan
            else:
                row[c] = str(disp[c].iloc[i])
        if this_field is not None:
            row['is_this_field'] = bool(events['field'].iloc[i] == this_field)
        table.add_row(start_time=start[i], stop_time=stop[i], **row)
    nwbfile.add_time_intervals(table)
    return table


def read_nwb(nwbfile, name: str = 'light_exposure_history') -> tuple[pd.DataFrame, dict]:
    """Events and meta from an NWB file written with `write_nwb`. Times are relative (`t_start_rel_s`)."""
    table = nwbfile.get_time_intervals(name) if hasattr(nwbfile, 'get_time_intervals') else nwbfile.intervals[name]
    df = table.to_dataframe().reset_index(drop=True)
    df = df.rename(columns=dict(start_time='t_start_rel_s', stop_time='t_end_rel_s'))
    df = df.replace({UNKNOWN: np.nan, NOT_APPLICABLE: np.nan})
    df['incomplete'] = df['incomplete'].fillna('').astype(str)
    meta = {}
    if ' meta=' in table.description:
        meta = json.loads(table.description.split(' meta=', 1)[1])
    return df, meta


def render_html(events: pd.DataFrame, prior: pd.DataFrame | None, meta: Mapping, path: str | None = None) -> str:
    """Self-contained HTML page: header with setup and gaps, timeline per field, event and prior-exposure tables.

    Works with the output of `build_events`, `read_csv` or `read_nwb` (uses `t_start_rel_s`/`t_end_rel_s`).
    Writes to `path` if given and returns the HTML.
    """
    title = f"Light exposure history {meta.get('experiment', '')}".strip()
    gaps = meta.get('gaps') or experiment_gaps(events, meta)
    info_keys = ['experiment', 'setupid', 'stimulator_id', 'stimulator_projection', 'stimulator_color_map',
                 'stimulator_confidence', 'stimulator_reason', 'stimulator_calibration', 'imaging_setup_id',
                 'clock_offset_s', 'n_stims_logged', 'n_stims_matched', 'n_recordings']
    info = ''.join(f"<tr><th>{_e(k)}</th><td>{_e(meta.get(k, UNKNOWN) if meta.get(k) is not None else UNKNOWN)}"
                   f"</td></tr>" for k in info_keys)
    parts = [
        _HTML_HEAD.replace('__TITLE__', _e(title)),
        f"<h1>{_e(title)}</h1>",
        "<section><h2>Setup</h2><table class='kv'>", info, "</table></section>",
        "<section class='gaps'><h2>Gaps</h2><ul>", ''.join(f"<li>{_e(g)}</li>" for g in gaps), "</ul></section>",
        "<section><h2>Timeline</h2>", _timeline_svg(events), _LEGEND, "</section>",
    ]
    if prior is not None and len(prior) > 0:
        parts += ["<section><h2>Exposure before each recording</h2>",
                  "<p class='note'>Seconds of laser scanning and stimulation before the recording started. "
                  "Dose = mean LED input (0-255) x s. <code>*</code>: position taken from the next recording.</p>",
                  _html_table(to_display(prior.drop(columns=['t_start'], errors='ignore'))), "</section>"]
    shown = [c for c in EVENT_COLUMNS if c in events and c not in ('t_start', 't_end')]
    parts += ["<section><h2>Events</h2>", _html_table(to_display(events[shown])), "</section>", _HTML_TAIL]
    page = '\n'.join(parts)
    if path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(page)
    return page


def export_experiment(out_dir: str, events: pd.DataFrame, meta: Mapping, html: bool = True) -> dict:
    """Write CSV, JSON and (optionally) HTML of one experiment to `out_dir`. Returns the file paths."""
    meta = dict(meta)
    meta['gaps'] = experiment_gaps(events, meta)
    prior = summarize_prior_exposure(events)
    files = write_csv(out_dir, events, prior, meta)
    if html:
        files['html'] = os.path.join(out_dir, 'exposure_history.html')
        render_html(events, prior, meta, path=files['html'])
    return files


def experiment_folder_name(key: Mapping) -> str:
    """Folder name of an experiment, e.g. `2020-01-30_1_Name`."""
    return '_'.join(str(key[k]) for k in ('date', 'exp_num', 'experimenter') if k in key)


# ----------------------------------------------------------------------------------------------------------------------
# HTML helpers
# ----------------------------------------------------------------------------------------------------------------------
_HTML_HEAD = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root { --bg:#fff; --fg:#1d1d1f; --muted:#666; --line:#ddd; --laser:#c0392b; --led:#2e6fd8; --unk:#f6e7b0;
        --head:#f4f4f6; }
@media (prefers-color-scheme: dark) { :root { --bg:#16171a; --fg:#e8e8ea; --muted:#9a9aa0; --line:#33343a;
        --laser:#ff6b5b; --led:#6aa1ff; --unk:#5a4b14; --head:#202126; } }
body { background:var(--bg); color:var(--fg); font:14px/1.4 system-ui, sans-serif; margin:0; padding:16px; }
h1 { font-size:20px; } h2 { font-size:16px; margin-top:28px; }
table { border-collapse:collapse; font-size:12px; } th, td { border:1px solid var(--line); padding:2px 6px; }
th { background:var(--head); position:sticky; top:0; cursor:pointer; white-space:nowrap; }
td.unknown { background:var(--unk); } td.na { color:var(--muted); }
.kv th { text-align:left; cursor:default; position:static; }
.scroll { overflow:auto; max-height:70vh; max-width:100%; }
.gaps li { margin:2px 0; } .note { color:var(--muted); font-size:12px; }
svg text { fill:var(--fg); font-size:11px; } .legend span { margin-right:16px; font-size:12px; }
.sw { display:inline-block; width:14px; height:10px; margin-right:4px; vertical-align:middle; }
</style></head><body>"""

_HTML_TAIL = """<script>
document.querySelectorAll('table.sortable th').forEach((th, i) => th.addEventListener('click', () => {
  const tb = th.closest('table').tBodies[0], asc = th.dataset.asc !== '1';
  const rows = Array.from(tb.rows), key = r => r.cells[i].textContent;
  rows.sort((a, b) => { const x = key(a), y = key(b), nx = parseFloat(x), ny = parseFloat(y);
    const c = (!isNaN(nx) && !isNaN(ny)) ? nx - ny : x.localeCompare(y); return asc ? c : -c; });
  th.dataset.asc = asc ? '1' : '0'; rows.forEach(r => tb.appendChild(r));
}));
</script></body></html>"""

_LEGEND = ("<div class='legend'><span><i class='sw' style='background:var(--laser)'></i>laser scan</span>"
           "<span><i class='sw' style='background:var(--led)'></i>stimulus (simultaneous recording)</span>"
           "<span><i class='sw' style='background:none;border:1px dashed var(--led)'></i>stimulus, position from "
           "next recording</span><span><i class='sw' style='background:var(--led);opacity:.4'></i>unlisted"
           "</span></div>")


def _timeline_svg(events: pd.DataFrame) -> str:
    t0 = pd.to_numeric(events['t_start_rel_s'], errors='coerce')
    t1 = pd.to_numeric(events['t_end_rel_s'], errors='coerce')
    if t0.notna().sum() == 0:
        return "<p class='note'>No event with a known time.</p>"
    fields = _unique([str(f) for f in events['field'] if _notna(f)])
    lanes = fields + [UNKNOWN]
    t_max = float(np.nanmax(np.concatenate([t0.to_numpy(dtype=float), t1.to_numpy(dtype=float)])))
    t_max = max(t_max, 1.)
    lane_h, left, width = 26, 90, 1000
    height = lane_h * len(lanes) + 30
    out = [f"<div class='scroll'><svg viewBox='0 0 {left + width + 10} {height}' width='100%' "
           f"style='min-width:640px' role='img' aria-label='timeline'>"]
    for i, lane in enumerate(lanes):
        y = i * lane_h
        out.append(f"<text x='4' y='{y + 17}'>{_e(lane)}</text>"
                   f"<line x1='{left}' x2='{left + width}' y1='{y + lane_h}' y2='{y + lane_h}' "
                   f"stroke='var(--line)'/>")
    for e, a, b in zip(events.to_dict('records'), t0, t1):
        if pd.isna(a):
            continue
        b = a if pd.isna(b) else b
        lane = lanes.index(str(e['field'])) if _notna(e.get('field')) else len(lanes) - 1
        x = left + a / t_max * width
        w = max((b - a) / t_max * width, 1.5)
        laser = e['event'] == 'laser_scan'
        y = lane * lane_h + (3 if laser else 14)
        tip = _e(f"{e['event']} {e.get('rec_name') or ''} {e.get('stimulus') or ''} "
                 f"[{a / 60:.1f}-{b / 60:.1f} min] {e.get('incomplete') or ''}")
        if laser:
            style = "fill='var(--laser)'"
        elif e.get('position_source') == 'next_recording':
            style = "fill='none' stroke='var(--led)' stroke-dasharray='3 2'"
        else:
            style = "fill='var(--led)'" + (" fill-opacity='0.4'" if e.get('listed') == 'unlisted' else '')
        out.append(f"<rect x='{x:.1f}' y='{y}' width='{w:.1f}' height='9' {style}><title>{tip}</title></rect>")
    n_ticks = 6
    for k in range(n_ticks + 1):
        t = t_max * k / n_ticks
        x = left + t / t_max * width
        out.append(f"<text x='{x:.0f}' y='{height - 6}' text-anchor='middle'>{t / 60:.0f} min</text>")
    out.append("</svg></div>")
    return ''.join(out)


def _html_table(df: pd.DataFrame) -> str:
    head = ''.join(f"<th>{_e(c)}</th>" for c in df.columns)
    body = []
    for row in df.itertuples(index=False):
        cells = []
        for v in row:
            cls = " class='unknown'" if v == UNKNOWN else (" class='na'" if v == NOT_APPLICABLE else '')
            cells.append(f"<td{cls}>{_e(_fmt(v))}</td>")
        body.append(f"<tr>{''.join(cells)}</tr>")
    return (f"<div class='scroll'><table class='sortable'><thead><tr>{head}</tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table></div>")


# ----------------------------------------------------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------------------------------------------------
def _e(v) -> str:
    return html.escape(str(v), quote=True)


def _fmt(v) -> str:
    if isinstance(v, (float, np.floating)):
        return f"{v:.4g}"
    return str(v)


def _notna(v) -> bool:
    if v is None:
        return False
    try:
        return not bool(pd.isna(v))
    except (TypeError, ValueError):
        return True


def _float(v) -> float | None:
    return float(v) if _notna(v) else None


def _bool(v) -> bool | None:
    return bool(v) if _notna(v) else None


def _mul(a, b) -> float | None:
    return float(a) * float(b) if (_notna(a) and _notna(b)) else None


def _listed_label(listed, kind) -> str | None:
    if kind is not None and kind != 'response':
        return NOT_APPLICABLE
    if listed is None:
        return None
    return 'listed' if listed else 'unlisted'


def _seconds_of_day(value) -> float | None:
    if not _notna(value):
        return None
    if isinstance(value, (timedelta, pd.Timedelta)):
        return value.total_seconds()
    if isinstance(value, (datetime, time)):
        return value.hour * 3600 + value.minute * 60 + value.second + value.microsecond / 1e6
    raise TypeError(f"Unsupported time type {type(value)}")


def _rel_s(t, t0) -> float | None:
    if t is None or t0 is None or pd.isna(t):
        return None
    return (pd.Timestamp(t) - pd.Timestamp(t0)).total_seconds()


def _clipped_dur(e: dict, t_rec) -> float:
    if not _notna(e.get('dur_s')):
        return np.nan
    end = e['t_start'] + timedelta(seconds=float(e['dur_s']))
    return max(0., (min(end, t_rec) - e['t_start']).total_seconds())


def _sort_events(events: pd.DataFrame) -> pd.DataFrame:
    order = {'laser_scan': 0, 'led_stimulus': 1}
    keys = [(t is None or pd.isna(t), t if (t is not None and not pd.isna(t)) else datetime.max, order[ev])
            for t, ev in zip(events['t_start'], events['event'])]
    idx = sorted(range(len(keys)), key=lambda i: keys[i])
    return events.iloc[idx]


def _unique(values: Iterable) -> list:
    seen, out = set(), []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out
