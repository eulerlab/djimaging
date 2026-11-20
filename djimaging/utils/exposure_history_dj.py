"""Fetch the inputs of the light exposure history (`djimaging.utils.exposure_history`) for one experiment.

Only `exp_history_table` is required. Every other table is optional (None: the information is `unknown`).
Only keys, counts and summary values are fetched from Presentation, Traces and the motion tables, never traces or
other stimulus-specific data. Unlisted recordings never have a Presentation, so nothing about them leaks.
"""
from __future__ import annotations

import os
import warnings
from collections.abc import Mapping
from pathlib import PurePosixPath

import numpy as np
import pandas as pd

from djimaging.utils import exposure_history as eh
from djimaging.utils.exp_history_utils import FIELD_KEY_DEFAULTS, logged_stimulus_name
from djimaging.utils.exp_history_utils import ref_path as _ref_path

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
    meta.update(_file_paths(exp_key, exp_history_table, recs))
    setupids = sorted({str(s) for s in recs.get('setupid', []) if eh._notna(s)})
    meta['setupid'] = ';'.join(setupids) if setupids else None

    stims, clock_offset_s = None, None
    meta['has_stim_log'] = False
    if stim_log_table is not None and len(stim_log_table & exp_key) > 0:
        stim_row = (stim_log_table & exp_key).fetch1()
        clock_offset_s = stim_row['clock_offset_s']
        meta.update(has_stim_log=True, clock_offset_s=clock_offset_s, n_stims_logged=stim_row['n_stims'],
                    n_stims_matched=stim_row['n_matched'],
                    n_stims_other_experiments=stim_row['n_outside'])
        stims = pd.DataFrame(((stim_log_table.Stim & exp_key) * stim_log_table.Stim.stim_log_table).proj(
            'rec_name', 'status', 'stim_file_name', 'stim_md5', 't_start', 't_end', 't_dur_s', 'aborted').to_dicts())
        if len(stims) > 0:
            stimuli = [] if stimulus_table is None else list(zip(*stimulus_table().to_arrays(
                'stim_name', 'stim_hash', 'stim_dict')))
            stims['stim_name'] = [logged_stimulus_name(m, f, stimuli)
                                  for m, f in zip(stims['stim_md5'], stims['stim_file_name'])]

    meta.update(_stimulator_meta(exp_key, experiment_stimulator_table))
    meta.update(_imaging_setup_meta(exp_key, experiment_imaging_setup_table))

    light_levels = light_levels_table().to_pandas().reset_index() if light_levels_table is not None else None
    exp_date = exp_key.get('date')
    events = eh.build_events(recs.drop(columns=['h5_file', 'smp_file', 'setupid'], errors='ignore'), stims,
                             clock_offset_s=clock_offset_s, exp_date=exp_date, light_levels=light_levels)
    return events, meta


INI_REPORT_DIR = 'exposure_history'  # Report folder in the experiment folder (next to the .ini file)


def report_dir_next_to_ini(key: Mapping, exp_history_table) -> str:
    """Report folder `<experiment folder>/exposure_history` next to the .ini file of the experiment (`header_path`)."""
    header_path = (exp_history_table.experiment_table & key).fetch1('header_path')
    return os.path.join(header_path, INI_REPORT_DIR)


def export_exposure_history(out_dir: str | None, keys, html: bool = True, exists: str = 'skip',
                            next_to_ini: bool = False, field_map: bool = True, field_map_kwargs: dict | None = None,
                            **tables) -> list[dict]:
    """Export the report of each experiment key to `<out_dir>/<date>_<exp_num>_<experimenter>/` and/or next to the
    .ini file of the experiment (`<experiment folder>/exposure_history/`).

    Parameters
    ----------
    out_dir : str | None
        One sub-folder per experiment; None: no export there.
    keys : iterable of experiment keys.
    html : bool
        Also write the HTML report.
    exists : str
        'error', 'skip' (default) or 'overwrite' if a report exists, see `exposure_history.check_report_dir`. Only
        report files are ever overwritten, never other files.
    next_to_ini : bool
        Also write the report to `<experiment folder>/exposure_history/`.
    field_map : bool
        Include the field map (`fetch_field_map`).
    field_map_kwargs : dict | None
        Passed to `fetch_field_map`, e.g. `retinal_location_table`, `images_from_files`.
    tables
        Passed to `fetch_exposure_history` (`exp_history_table`, ...).

    Returns
    -------
    list[dict]
        Per experiment: `{'out_dir': files or None, 'next_to_ini': files or None}` (None: not written or skipped).
    """
    results = []
    for key in keys:
        events, meta = fetch_exposure_history(key, **tables)
        fields = None
        if field_map:
            fields = fetch_field_map(
                key, tables['exp_history_table'], exp_history_field_table=tables.get('exp_history_field_table'),
                from_raw_data=tables.get('from_raw_data', False), **(field_map_kwargs or {}))
        result = dict(out_dir=None, next_to_ini=None)
        targets = []
        if out_dir is not None:
            targets.append(('out_dir', os.path.join(out_dir, meta['experiment'])))
        if next_to_ini:
            targets.append(('next_to_ini', report_dir_next_to_ini(key, tables['exp_history_table'])))
        for name, target in targets:
            result[name] = eh.export_experiment(target, events, meta, html=html, fields=fields, exists=exists)
        results.append(result)
    return results


_FIELD_REC_ATTRS = ['kind', 'name_info', 'rec_start', 'absx', 'absy', 'absz', 'pixel_size_um', 'nxpix', 'nypix',
                    'scan_type', 'angle_deg', 'h5_file', 'smp_file']


def fetch_field_map(key: Mapping, exp_history_table, exp_history_field_table=None, retinal_location_table=None,
                    images: bool = True, images_from_files: bool = True,
                    image_channels: tuple = ('wDataCh0', 'wDataCh1'), from_raw_data: bool = False,
                    verbose: bool = False) -> pd.DataFrame:
    """All recorded fields (also structure-only fields) and landmarks of one experiment, see
    `exposure_history.build_fields`.

    Parameters
    ----------
    key : Mapping
        Restriction to one experiment.
    exp_history_table : ExpHistory
    exp_history_field_table : ExpHistoryField, optional
        Fields of the Field table and their recordings: position, scan area and image (`StackAverages`) are taken from
        the Field table. Recordings that are not linked to a Field are grouped by the field parts of their file names
        (`field`, `region`, ...).
    retinal_location_table : RetinalFieldLocation, optional
        Retinal positions of the fields in the table. For other fields and landmarks, the position is computed from
        the optic disc (`OpticDisk` of its `relativefieldlocation_table`) and eye and orientation
        (`retina_orientation_table`),
        if they are known.
    images : bool
        Load field images.
    images_from_files : bool
        Load the images of fields that are not in the Field table from their recording (.h5, else .smp file).
    image_channels : tuple
        Channels to show.
    from_raw_data : bool
        Use the .smp file names to group recordings (as Presentation with `from_raw_data=1`).
    """
    exp_key = (exp_history_table & key).fetch1('KEY')
    recs = pd.DataFrame((exp_history_table.Recording & exp_key).proj(*_FIELD_REC_ATTRS).to_dicts())
    if len(recs) == 0:
        return eh.build_fields(pd.DataFrame(columns=['rec_name', 'kind', 'field']))

    field_table = exp_history_field_table.field_table if exp_history_field_table is not None else None
    key_names = list(field_table().new_primary_keys) if field_table is not None else ['field', 'region']

    linked, field_keys = {}, {}
    if exp_history_field_table is not None:
        for row in (exp_history_field_table.Recording & exp_key).to_dicts():
            label = _field_label(row, key_names)
            linked.setdefault(row['rec_name'], label)
            field_keys.setdefault(label, {k: row[k] for k in field_table().primary_key})

    from djimaging.tables.history.exp_history import _preferred_name_info
    labels = []
    for rec_name, name_info in zip(recs['rec_name'], recs['name_info']):
        if rec_name in linked:
            labels.append(linked[rec_name])
            continue
        info = _preferred_name_info(name_info or {}, from_raw_data)
        if info is None or info.get('field') is None:
            labels.append(None)
        else:
            labels.append(_field_label(
                {k: info.get(k) if info.get(k) is not None else FIELD_KEY_DEFAULTS.get(k) for k in key_names},
                key_names))
    recs['field'] = labels

    field_info = {label: _field_table_info(field_table, fkey, images, image_channels)
                  for label, fkey in field_keys.items()}
    retinal = _retinal_from_table(field_keys, retinal_location_table) if retinal_location_table is not None else {}
    fields = eh.build_fields(recs.drop(columns=['name_info']), field_info=field_info, retinal=retinal)
    if retinal_location_table is not None:
        _add_computed_retinal(fields, exp_key, retinal_location_table)

    if images and images_from_files:
        files = {r: (h5, smp) for r, h5, smp in zip(recs['rec_name'], recs['h5_file'], recs['smp_file'])}
        for i, row in fields.iterrows():
            if row['status'] not in ('response', 'structure') or isinstance(row['images'], Mapping) \
                    or not row['image_rec']:
                continue
            imgs, issue = _images_from_file(*files[row['image_rec']], scan_type=row['scan_type'],
                                            channels=image_channels, store=exp_history_table._filepath_store)
            fields.at[i, 'images'] = imgs
            fields.at[i, 'image_source'] = 'file' if imgs else None
            fields.at[i, 'image_issue'] = issue
            if verbose:
                print(f"{row['field']}: image from {row['image_rec']}{f' failed: {issue}' if issue else ''}")
    return fields


def _field_label(row: Mapping, key_names) -> str:
    return '/'.join(str(row[k]) for k in key_names if k in row and str(row[k]) not in ('N/A', '', 'None'))


def _field_table_info(field_table, field_key: Mapping, images: bool, image_channels) -> dict:
    """Position, scan area and stack averages of a field of the Field table."""
    row = (field_table & field_key).fetch1()
    pixel_size = row.get('pixel_size_um')
    pixel_size = float(pixel_size) if pixel_size is not None and pixel_size > 0 else None
    info = dict(x_um=row['absx'], y_um=row['absy'], z_um=row['absz'], scan_type=row['scan_type'],
                image_rec=PurePosixPath(str(_ref_path(row['field_data_file']))).name)
    if pixel_size is not None:
        info['width_um'] = pixel_size * row['nxpix']
        info['height_um'] = pixel_size * row['nypix'] if row['scan_type'] in ('xy', 'xyz') else None
    if images:
        from djimaging.utils.dj_storage import load_array
        ch_names, averages = (field_table.StackAverages & field_key).to_arrays('ch_name', 'ch_average')
        imgs = {ch: eh.field_image(load_array(avg)) for ch, avg in zip(ch_names, averages) if ch in image_channels}
        if imgs:
            info.update(images=imgs, image_source='Field.StackAverages')
    return info


def _images_from_file(h5_file, smp_file, scan_type, channels, store: str) -> tuple[dict | None, str | None]:
    """Images of one recording (.h5 file, else .smp file); returns (images, issue)."""
    from djimaging.utils.dj_storage import local_path
    from djimaging.utils.scanm.recording import ScanMRecording

    issues = []
    for value in (h5_file, smp_file):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        try:
            with local_path(value, store) as filepath:
                rec = ScanMRecording(str(filepath))
            imgs = {ch: eh.field_image(stack, scan_type) for ch, stack in rec.ch_stacks.items() if ch in channels}
            if imgs:
                return imgs, None
            issues.append(f"{PurePosixPath(str(_ref_path(value))).name}: no channel of {list(channels)}")
        except Exception as e:  # the report must not fail because of one unreadable file
            issues.append(f"{PurePosixPath(str(_ref_path(value))).name}: {type(e).__name__}: {e}"[:300])
    return None, '; '.join(issues) or 'no file'


def _retinal_from_table(field_keys: Mapping, retinal_location_table) -> dict:
    """Retinal positions of fields in RetinalFieldLocation: {label: (ventral_dorsal, temporal_nasal, source)}."""
    out = {}
    for label, fkey in field_keys.items():
        rows = (retinal_location_table & fkey).to_dicts()
        if len(rows) > 0:
            out[label] = (rows[0]['ventral_dorsal_pos_um'], rows[0]['temporal_nasal_pos_um'], 'RetinalFieldLocation')
    return out


def _add_computed_retinal(fields: pd.DataFrame, exp_key: Mapping, retinal_location_table) -> None:
    """Retinal positions of the other fields and landmarks, from their stage position, the optic disc and eye and
    orientation of the experiment (as RetinalFieldLocation). Only if the experiment has one optic disc position."""
    missing = fields['ventral_dorsal_pos_um'].isna() & fields['x_um'].notna() & fields['y_um'].notna()
    if not missing.any():
        return
    try:
        od_table = retinal_location_table.relativefieldlocation_table.opticdisk_table
        od_pos = {(float(x), float(y)) for x, y in zip(*(od_table & exp_key).to_arrays('odx', 'ody'))}
        orientations = {(row['eye'], row['orientation_deg'])
                        for row in (retinal_location_table.retina_orientation_table & exp_key).to_dicts()}
    except Exception as e:  # optional information, e.g. RetinaOrientation not bound
        warnings.warn(f"Retinal position of fields that are not in RetinalFieldLocation unknown for {exp_key}: {e}")
        return
    if len(od_pos) != 1 or len(orientations) != 1:
        return
    eye, rotation = orientations.pop()
    if rotation is None or eye not in ('left', 'right'):
        return
    from djimaging.utils.scanm.setup_utils import get_retinal_position
    odx, ody = od_pos.pop()
    for i in fields.index[missing]:
        vd, tn = get_retinal_position(rel_xcoord_um=fields.at[i, 'x_um'] - odx,
                                      rel_ycoord_um=fields.at[i, 'y_um'] - ody, rotation=rotation, eye=eye)
        fields.at[i, 'ventral_dorsal_pos_um'] = float(vd)
        fields.at[i, 'temporal_nasal_pos_um'] = float(tn)
        fields.at[i, 'retinal_source'] = 'computed from OpticDisk'


def _file_paths(exp_key: Mapping, exp_history_table, recs: pd.DataFrame) -> dict:
    """`header_path` (experiment folder) and `files`: per recording its field label and the absolute paths of its
    .h5, .smp and .smh files (store-relative if the store is not mounted locally)."""
    header_path = (exp_history_table.experiment_table & exp_key).fetch1('header_path')
    store = exp_history_table._filepath_store
    smh = {}
    if len(recs) > 0:
        smh = dict(zip(*(exp_history_table.Recording & exp_key).to_arrays('rec_name', 'smh_file')))
    files = []
    for r in recs.to_dict('records'):
        files.append(dict(field=r.get('field'), rec_name=r['rec_name'], kind=r.get('kind'),
                          h5_file=_abs_path(r.get('h5_file'), store), smp_file=_abs_path(r.get('smp_file'), store),
                          smh_file=_abs_path(smh.get(r['rec_name']), store)))
    return dict(header_path=header_path, files=files)


def _abs_path(value, store: str) -> str | None:
    """Absolute local path of a fetched `<filepath>`, else its store-relative path; None if missing."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    from djimaging.utils.dj_storage import local_file_path
    try:
        return str(local_file_path(value, store))
    except Exception:  # remote or unconfigured store
        return _ref_path(value)


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
