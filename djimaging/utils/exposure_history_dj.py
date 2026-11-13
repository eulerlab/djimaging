"""Fetch the inputs of the light exposure history (`djimaging.utils.exposure_history`) for one experiment.

Only `exp_history_table` is required. Every other table is optional (None: the information is `unknown`).
Only keys, counts and summary values are fetched from Presentation, Traces and the motion tables, never traces or
other stimulus-specific data. Unlisted recordings never have a Presentation, so nothing about them leaks.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import PurePosixPath

import numpy as np
import pandas as pd

from djimaging.utils import exposure_history as eh

_REC_ATTRS = ['kind', 'stim_token', 'rec_start', 'rec_dur_s', 'stopped_early', 'absx', 'absy', 'absz', 'zoom',
              'pixel_size_um', 'nxpix', 'nypix', 'nzpix', 'z_step_um', 'real_pix_dur_us', 'scan_type', 'setupid',
              'header_issue', 'h5_file', 'smp_file']


def fetch_exposure_history(
        key: Mapping, exp_history_table, stimulus_table=None, exp_history_field_table=None, stim_log_table=None,
        presentation_table=None, traces_table=None, motion_detection_table=None, motion_trust_table=None,
        experiment_stimulator_table=None, experiment_imaging_setup_table=None, light_levels_table=None,
        from_raw_data: bool = False) -> tuple[pd.DataFrame, dict]:
    """Events (see `exposure_history.build_events`) and meta of one experiment.

    Parameters
    ----------
    key : Mapping
        Restriction to one experiment (extra keys are ignored).
    exp_history_table : ExpHistory
    stimulus_table : Stimulus, optional
        For the listed status of recordings and logged stimuli.
    exp_history_field_table : ExpHistoryField, optional
        Recordings of each field; without it, all recordings have field `unknown`.
    stim_log_table : ExpHistoryStimLog, optional
        Logged QDSpy stimuli and clock offset; without it, stimuli are taken from the file names.
    presentation_table, traces_table, motion_detection_table, motion_trust_table : optional
        Processing status of listed recordings (presence, counts and QC summaries only).
    experiment_stimulator_table, experiment_imaging_setup_table : optional
        Stimulator (geometry, color map, calibration, confidence) and imaging setup of the experiment.
    light_levels_table : optional
        Table with `match_md5`, `match_file_name`, `match_stim_name`, `mean_uv`, `mean_green`, `extent_x_um`,
        `extent_y_um` (see `exposure_history.lookup_light_levels`).
    from_raw_data : bool
        Use the .smp file names for the listed status (as Presentation with `from_raw_data=1`).
    """
    exp_key = (exp_history_table & key).fetch1('KEY')
    exp_row = (exp_history_table & exp_key).fetch1()

    recs = pd.DataFrame((exp_history_table.Recording & exp_key).proj(*_REC_ATTRS).to_dicts())
    if len(recs) == 0:
        recs = pd.DataFrame(columns=['rec_name', 'rec_start', 'rec_dur_s'] + _REC_ATTRS)
    recs['field'] = _field_labels(exp_key, recs, exp_history_field_table)

    if stimulus_table is not None:
        listed = exp_history_table().listed(stimulus_table, restriction=exp_key, from_raw_data=from_raw_data)
        info = {r: (bool(l), ';'.join(n)) for r, l, n in
                zip(listed.get('rec_name', []), listed.get('listed', []), listed.get('stim_names', []))}
        recs['listed'] = [info.get(r, (None, None))[0] for r in recs['rec_name']]
        recs['stim_name'] = [info.get(r, (None, None))[1] or None for r in recs['rec_name']]
    else:
        recs['listed'] = None
        recs['stim_name'] = None

    status = _processing_status(exp_key, presentation_table, traces_table, motion_detection_table,
                                motion_trust_table)
    for col in ['has_presentation', 'n_traces', 'motion_slow_range_px', 'motion_trust_median',
                'motion_n_rois_flagged']:
        recs[col] = None
    if presentation_table is not None:
        by_path = {}
        for row in status:
            by_path[row['path']] = row
        for i, r in recs.iterrows():
            row = next((by_path[p] for p in (_ref_path(r['h5_file']), _ref_path(r['smp_file'])) if p in by_path),
                       None)
            if r['listed']:
                recs.at[i, 'has_presentation'] = row is not None
            if row is not None:
                for col in ['n_traces', 'motion_slow_range_px', 'motion_trust_median', 'motion_n_rois_flagged']:
                    recs.at[i, col] = row.get(col)

    meta = dict(experiment=eh.experiment_folder_name(exp_key), key={k: str(v) for k, v in exp_key.items()},
                n_recordings=len(recs), data_dirs=list(exp_row.get('data_dirs') or []),
                unscanned_files=list(exp_row.get('unscanned_files') or []))
    setupids = sorted({str(s) for s in recs.get('setupid', []) if eh._notna(s)})
    meta['setupid'] = ';'.join(setupids) if setupids else None

    stims, clock_offset_s = None, None
    meta['has_stim_log'] = False
    if stim_log_table is not None and len(stim_log_table & exp_key) > 0:
        stim_row = (stim_log_table & exp_key).fetch1()
        clock_offset_s = stim_row['clock_offset_s']
        meta.update(has_stim_log=True, clock_offset_s=clock_offset_s, n_stims_logged=stim_row['n_stims'],
                    n_stims_matched=stim_row['n_matched'])
        stims = pd.DataFrame(((stim_log_table.Stim & exp_key) * stim_log_table.Stim.stim_log_table).proj(
            'rec_name', 'stim_file_name', 'stim_md5', 't_start', 't_end', 't_dur_s', 'aborted').to_dicts())
        if len(stims) > 0:
            stims['stim_name'] = [_logged_stim_name(m, f, stimulus_table)
                                  for m, f in zip(stims['stim_md5'], stims['stim_file_name'])]

    meta.update(_stimulator_meta(exp_key, experiment_stimulator_table))
    meta.update(_imaging_setup_meta(exp_key, experiment_imaging_setup_table))

    light_levels = light_levels_table().to_pandas().reset_index() if light_levels_table is not None else None
    exp_date = exp_key.get('date')
    events = eh.build_events(recs.drop(columns=['h5_file', 'smp_file', 'setupid'], errors='ignore'), stims,
                             clock_offset_s=clock_offset_s, exp_date=exp_date, light_levels=light_levels)
    return events, meta


def export_exposure_history(out_dir: str, keys, html: bool = True, **tables) -> list[dict]:
    """Export one sub-folder `<out_dir>/<date>_<exp_num>_<experimenter>/` per experiment key.

    `tables` are passed to `fetch_exposure_history`. Returns the written file paths per experiment.
    """
    import os
    files = []
    for key in keys:
        events, meta = fetch_exposure_history(key, **tables)
        files.append(eh.export_experiment(os.path.join(out_dir, meta['experiment']), events, meta, html=html))
    return files


def _ref_path(value) -> str | None:
    """Store-relative POSIX path of a fetched `<filepath>` (ObjectRef or str)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    path = getattr(value, 'path', value)
    return PurePosixPath(str(path).replace('\\', '/')).as_posix()


def _field_labels(exp_key: Mapping, recs: pd.DataFrame, exp_history_field_table) -> list:
    if exp_history_field_table is None or len(recs) == 0:
        return [None] * len(recs)
    field_table = exp_history_field_table.field_table
    label_keys = [k for k in field_table().primary_key if k not in exp_key and k != 'raw_id']
    labels = {}
    for row in (exp_history_field_table.Recording & exp_key).to_dicts():
        label = '/'.join(str(row[k]) for k in label_keys if str(row[k]) not in ('N/A', ''))
        labels.setdefault(row['rec_name'], set()).add(label)
    return [';'.join(sorted(labels[r])) if r in labels else None for r in recs['rec_name']]


def _processing_status(exp_key, presentation_table, traces_table, motion_detection_table,
                       motion_trust_table) -> list[dict]:
    if presentation_table is None:
        return []
    pres_key_names = presentation_table().primary_key
    rows = []
    for pres in (presentation_table & exp_key).proj('pres_data_file').to_dicts():
        pres_key = {k: pres[k] for k in pres_key_names}
        row = dict(path=_ref_path(pres['pres_data_file']))
        if traces_table is not None:
            row['n_traces'] = len(traces_table & pres_key)
        if motion_detection_table is not None:
            ranges = (motion_detection_table & pres_key).to_arrays('slow_range_px')
            row['motion_slow_range_px'] = float(np.max(ranges)) if len(ranges) > 0 else None
        if motion_trust_table is not None:
            medians, flagged = (motion_trust_table & pres_key).to_arrays('trust_median', 'n_rois_flagged')
            row['motion_trust_median'] = float(np.median(medians)) if len(medians) > 0 else None
            row['motion_n_rois_flagged'] = int(np.sum(flagged)) if len(flagged) > 0 else None
        rows.append(row)
    return rows


def _logged_stim_name(md5: str | None, file_name: str | None, stimulus_table) -> str | None:
    """Stimulus name of a logged QDSpy stimulus: md5 in `stim_hash_list` or file name starting with a name of
    `stim_name_list` of `stim_dict` (as PresentationLog). None if not in the Stimulus table."""
    if stimulus_table is None:
        return None
    names = []
    for stim_name, stim_hash, stim_dict in zip(*stimulus_table().to_arrays('stim_name', 'stim_hash', 'stim_dict')):
        stim_dict = stim_dict or {}
        hashes = ({stim_hash} | set(stim_dict.get('stim_hash_list', []))) - {''}
        if md5 is not None and md5 in hashes:
            return stim_name
        if file_name is not None and any(str(file_name).lower().startswith(str(n).lower())
                                         for n in stim_dict.get('stim_name_list', []) if n):
            names.append(stim_name)
    return names[0] if len(names) == 1 else None


def _stimulator_meta(exp_key, experiment_stimulator_table) -> dict:
    if experiment_stimulator_table is None or len(experiment_stimulator_table & exp_key) == 0:
        return dict(stimulator_id=None)
    assignment = (experiment_stimulator_table & exp_key).fetch1()
    stim_table = experiment_stimulator_table.stimulator_table
    row = (stim_table & {'stimulator_id': assignment['stimulator_id']}).fetch1()
    out = dict(stimulator_id=assignment['stimulator_id'], stimulator_confidence=assignment.get('confidence'),
               stimulator_reason=assignment.get('reason'), stimulator_source=row.get('source'))
    for k in ('projection', 'color_map', 'um_per_px', 'stimulator_delay_ms', 'screen_x_axis', 'screen_y_axis',
              'offset_relx_um', 'offset_rely_um', 'calibration'):
        out[f'stimulator_{k}'] = row.get(k)
    return out


def _imaging_setup_meta(exp_key, experiment_imaging_setup_table) -> dict:
    if experiment_imaging_setup_table is None or len(experiment_imaging_setup_table & exp_key) == 0:
        return dict(imaging_setup_id=None)
    assignment = (experiment_imaging_setup_table & exp_key).fetch1()
    setup_table = experiment_imaging_setup_table.imaging_setup_table
    row = (setup_table & {'imaging_setup_id': assignment['imaging_setup_id']}).fetch1()
    return dict(imaging_setup_id=assignment['imaging_setup_id'], imaging_setup_confidence=assignment.get('confidence'),
                imaging_setup_reason=assignment.get('reason'), scan_width_um=row.get('scan_width_um'))
