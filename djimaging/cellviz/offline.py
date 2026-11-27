"""Offline field source: one experiment folder with ScanM files (raw ``Raw/*.smp|smh`` or ``Pre/*.h5``) and saved ROI
mask files (``<roi_mask_dir>/*_ROIs.npy|pkl``), without DataJoint connection, schema import or AutoROIs models.

Every step reuses the function the pipeline uses for it, with the defaults of the example schema
(``djimaging.schemas.blocks.core`` / ``basic``):

=====================  ========================================================================================
Pipeline table         Offline equivalent
=====================  ========================================================================================
Field / Presentation   ``filesystem_utils.get_file_info_df`` (file name -> field, stimulus, condition, kind),
                       stimulus alias matching as in ``PresentationTemplate.load_field_stim_file_info_df``,
                       ``ScanMRecording`` + ``compute_triggers`` (automatic threshold, as stored by ``Presentation``)
RoiMask(Presentation)  ``mask_utils.to_roi_mask_file`` / ``load_roi_mask_file`` / ``to_igor_format``, field mask chosen
                       with ``sort_roi_mask_files`` (``mask_alias``), ``compare_roi_masks`` -> same/shifted/different
Roi                    ROI IDs = ``abs`` of the Igor labels of the field mask
Traces                 ``roi2trace_from_stack`` (``compute_from_stack``, ``trace_precision='line'``), ROIs touching the
                       light-artifact lines are skipped (``include_artifacts=0``), ``check_valid_triggers_rel_to_tracetime``
PreprocessTraces       ``process_trace`` with ``PreprocessParams.add_default`` values and ``_baseline_max_dt=2``
Snippets               ``split_trace_by_reps`` with ``_pad_trace=True``, no snippet baseline (``snippet_base_dt=NULL``)
ChirpQI / OsDsIndexes  same formulas (``RepeatQITemplate.make``, ``quality_index_ds``; DSI/OSI without shuffle test)
=====================  ========================================================================================

What the files cannot establish and is therefore *assumed* (and reported in the viewer): the stimulus identity and
its parameters (from the file name and the default ``Stimulus.add_chirp`` / ``add_movingbar`` / ``add_noise``
parameters), the preprocessing parameters, and the experimenter / ``raw_id`` / ``preprocess_id`` keys. Quality values
are therefore labelled "offline recompute" and are not the stored database values.
"""
from __future__ import annotations

import configparser
import os
import warnings
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import Mapping, Optional

import numpy as np

from djimaging.cellviz.data import (
    HighResImage, PresentationData, PresentationInfo, QualityValues, RoiLabels, Signal, StimulusInfo, Trace,
    to_display)
from djimaging.cellviz.quality import quality_family, unsupported_reason
from djimaging.cellviz.timing import sample_times

# Defaults of UserInfoTemplate and of the userinfo in the notebook blocks (raw files have no datatype prefix).
DEFAULT_USERINFO = dict(
    datatype_loc=0, animal_loc=1, region_loc=2, field_loc=3, stimulus_loc=4, cond1_loc=5, cond2_loc=6, cond3_loc=7,
    opticdisk_alias='od_opticdisk_opticdisc', outline_alias='outline_edge', highres_alias='hq_hr_highresolution_512',
    mask_alias='chirp_mb_movingbar', raw_data_dir='Raw', pre_data_dir='Pre')

MOVINGBAR_DIRECTIONS = (0, 180, 45, 225, 90, 270, 135, 315)

# Defaults of StimulusTemplate.add_chirp / add_movingbar / add_noise.
DEFAULT_STIMULI = {
    'chirp': dict(alias='chirp_gchirp_globalchirp_lchirp_localchirp', stim_family='chirp', isrepeated=True,
                  ntrigger_rep=2, trial_info=None),
    'movingbar': dict(alias='mb_mbar_bar_movingbar', stim_family='movingbar', isrepeated=True, ntrigger_rep=1,
                      trial_info=[dict(name=d, ntrigger=1) for d in MOVINGBAR_DIRECTIONS]),
    'noise': dict(alias='dn_noise', stim_family='noise', isrepeated=False, ntrigger_rep=1500, trial_info=None),
}

# PreprocessParams.add_default and PreprocessTraces._baseline_max_dt of the example schema.
DEFAULT_PREPROCESS = dict(window_len_seconds=60, poly_order=3, subtract_baseline=True, standardize=1,
                          non_negative=False, f_cutoff=None, fs_resample=None, baseline_max_dt=2.)

TRIGGER_CHANNEL = 'wDataCh2'
SNIPPETS_PAD_TRACE = True  # Snippets._pad_trace of the example schema (djimaging.schemas.blocks.core)


def read_experiment_ini(experiment_dir: str | Path) -> dict:
    """Setup id, date and eye from the ``.ini`` header of an experiment folder (read-only)."""
    inis = sorted(Path(experiment_dir).glob('*.ini'))
    if len(inis) == 0:
        return dict()
    parser = configparser.ConfigParser()
    parser.read(inis[0], encoding='utf-8')
    out = dict(ini_file=str(inis[0]))
    if parser.has_option('Project', 'string_setupID'):
        out['setup_id'] = int(parser.get('Project', 'string_setupID'))
    if parser.has_option('Project', 'string_date'):
        out['date'] = parser.get('Project', 'string_date')
    if parser.has_option('Animal', 'string_eye'):
        out['eye'] = parser.get('Animal', 'string_eye')
    return out


def trigger_labels_for(triggers: np.ndarray, stim: Optional[Mapping]) -> Optional[list]:
    """Short labels of the triggers: ``'rep k'`` at repetition starts, or the condition of every trigger for
    one-trigger trials (moving bar: the direction, ``'45°'``; ``trial_info`` order). Repeat boundaries of one-trigger
    trials are marked by the renderer, not in the label, so that labels stay short."""
    from djimaging.cellviz.trials import condition_label
    n = len(triggers)
    if stim is None or n == 0 or not stim.get('isrepeated'):
        return None
    trial_info = stim.get('trial_info')
    ntrigger_rep = int(stim.get('ntrigger_rep') or 1)
    if trial_info and ntrigger_rep == 1:  # one trigger per trial, e.g. moving bar directions
        names = [condition_label(t['name']) for t in trial_info]
        return [names[k % len(names)] for k in range(n)]
    return [f"rep {k // ntrigger_rep + 1}" if k % ntrigger_rep == 0 else '' for k in range(n)]


def stimulus_info_for(stim: Optional[Mapping], pad_trace: bool) -> Optional[StimulusInfo]:
    """``StimulusInfo`` from a ``Stimulus`` row (``isrepeated``, ``ntrigger_rep``, ``trial_info``, ``stim_dict``)."""
    if stim is None:
        return None
    stim_dict = stim.get('stim_dict') or {}
    return StimulusInfo(isrepeated=bool(stim.get('isrepeated')), ntrigger_rep=int(stim.get('ntrigger_rep') or 0),
                        trial_info=stim.get('trial_info'), trigger_delay=float(stim_dict.get('trigger_delay', 0.) or 0.),
                        pad_trace=bool(pad_trace))


def compute_quality(family: str, snippets: np.ndarray, stim: Mapping, dt: float) -> dict:
    """QI of one ROI from its snippets (time x repetitions), with the formulas of the QI tables."""
    if family == 'chirp':
        qidx = np.var(np.mean(snippets, axis=1)) / np.mean(np.var(snippets, axis=0))
        return dict(value=float(qidx), reference=1 / snippets.shape[1])
    if family == 'movingbar':
        from djimaging.tables.response.movingbar.orientation_utils import (
            get_si, get_time_dir_kernels, preprocess_mb_snippets, quality_index_ds)
        dir_order = np.asarray([t['name'] for t in stim['trial_info']])
        sorted_directions, sorted_responses, sorted_averages = preprocess_mb_snippets(snippets, dir_order)
        _, dir_component = get_time_dir_kernels(sorted_averages, dt=dt)
        dsi, _ = get_si(dir_component, sorted_directions, 1)
        osi, _ = get_si(dir_component, sorted_directions, 2)
        return dict(value=float(quality_index_ds(sorted_responses)), ds_index=float(dsi), os_index=float(osi))
    raise ValueError(family)


class LocalFieldSource:
    """All presentations of one field of a local experiment folder. Presentations are loaded lazily and cached.

    Parameters
    ----------
    experiment_dir : folder containing the ``.ini`` header, ``Raw/`` or ``Pre/`` and the ROI mask folder.
    from_raw_data : read ``Raw/*.smp`` (True) or ``Pre/*.h5`` (False); default: raw if ``Raw/`` has ``.smp`` files.
        The ROI masks always come from the mask files (``igor_roi_masks='no'``), not from ROIs stored in ``.h5``.
    field : field name as in the file names, e.g. ``'GCL0'``.
    roi_mask_dir : folder of the saved masks (``RoiMask().rescan_filesystem(roi_mask_dir=...)``).
    setup_id, date : default from the ``.ini`` file.
    stimuli : ``{stim_name: dict(alias, stim_family, isrepeated, ntrigger_rep, trial_info)}``, default
        ``DEFAULT_STIMULI``. Files whose stimulus token matches no alias are still listed (``stim_name=None``).
    """

    def __init__(self, experiment_dir: str | Path, field: str, roi_mask_dir: str = 'AutoROIs',
                 setup_id: Optional[int] = None, date: Optional[str] = None, stimuli: Optional[Mapping] = None,
                 userinfo: Optional[Mapping] = None, preprocess: Optional[Mapping] = None,
                 trace_precision: str = 'line', trigger_precision: str = 'line', include_artifacts: bool = False,
                 max_shift: int = 5, cache_size: int = 3, from_raw_data: Optional[bool] = None):
        from djimaging.utils.filesystem_utils import get_file_info_df

        self.experiment_dir = Path(experiment_dir).resolve()
        self.field = field
        self.roi_mask_dir = roi_mask_dir
        ini = read_experiment_ini(self.experiment_dir)
        self.setup_id = int(setup_id if setup_id is not None else ini.get('setup_id', 1))
        self.date = date if date is not None else ini.get('date')
        self.stimuli = dict(DEFAULT_STIMULI if stimuli is None else stimuli)
        self.userinfo = {**DEFAULT_USERINFO, **(userinfo or {})}
        self.preprocess = {**DEFAULT_PREPROCESS, **(preprocess or {})}
        self.trace_precision, self.trigger_precision = trace_precision, trigger_precision
        self.include_artifacts = include_artifacts
        self.max_shift = max_shift
        self._cache: OrderedDict = OrderedDict()
        self._cache_size = cache_size
        self.notes = [] if 'ini_file' in ini else [f"no .ini file in {self.experiment_dir}; setup_id={self.setup_id}"]
        self.notes.append("offline mode: traces, triggers and QI are recomputed from the raw files with djimaging's "
                          "functions and default parameters; they are not database values")

        raw_dir = self.experiment_dir / self.userinfo['raw_data_dir']
        if from_raw_data is None:
            from_raw_data = raw_dir.is_dir() and any(raw_dir.glob('*.smp'))
        self.from_raw_data = bool(from_raw_data)
        data_dir = raw_dir if self.from_raw_data else self.experiment_dir / self.userinfo['pre_data_dir']
        if not data_dir.is_dir():
            raise FileNotFoundError(f"data folder not found: {data_dir}")
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            df = get_file_info_df(str(data_dir), self.userinfo, from_raw_data=self.from_raw_data)
        if len(df) == 0 or 'field' not in df:
            raise FileNotFoundError(f"no ScanM files in {data_dir}")
        df = df[df['field'] == field]
        if len(df) == 0:
            raise ValueError(f"no files of field {field!r} in {data_dir}")
        self._files = df
        self._infos = self._build_infos(df[df['kind'] == 'response'])
        hr = df[df['kind'] == 'hr']
        self.highres_file = None if len(hr) == 0 else hr.iloc[0]['filepath']
        if len(hr) > 1:
            self.notes.append(f"{len(hr)} HighRes files, using {Path(self.highres_file).name}")
        self.label = f"{self.experiment_dir.parent.name}/{self.experiment_dir.name} field {field}"

    # ---------------------------------------------------------------- listing
    def _match_stimulus(self, token: str) -> Optional[str]:
        for stim_name, stim in self.stimuli.items():
            if token.lower() in stim['alias'].lower().split('_'):
                return stim_name
        return None

    def _mask_file(self, data_file: str) -> Optional[str]:
        from djimaging.utils.mask_utils import to_roi_mask_file
        for file_format in ('numpy', 'pickle'):
            path = to_roi_mask_file(data_file, roi_mask_dir=self.roi_mask_dir, file_format=file_format)
            if os.path.isfile(path):
                return path
        return None

    def _build_infos(self, df) -> dict:
        from djimaging.utils.mask_format_utils import to_igor_format
        from djimaging.utils.mask_utils import compare_roi_masks, load_roi_mask_file, sort_roi_mask_files

        rows = []
        for _, row in df.sort_values('filepath').iterrows():
            token = str(row['stimulus'])
            stim_name = self._match_stimulus(token)
            cond1 = row.get('cond1') if isinstance(row.get('cond1'), str) else 'control'
            region = row.get('region') if isinstance(row.get('region'), str) else 'N/A'
            stem = Path(row['filepath']).stem
            key = dict(date=self.date, exp_num=self.experiment_dir.name, field=self.field, region=region,
                       stim_name=stim_name if stim_name is not None else f'?{token}', cond1=cond1, file=stem)
            rows.append(dict(key=key, stim_name=stim_name, token=token, filepath=row['filepath'],
                             mask_file=self._mask_file(row['filepath'])))

        # Field mask = preferred mask (mask_alias order), as RoiMask._add_field_roi_masks
        with_mask = [r for r in rows if r['mask_file'] is not None]
        self.field_mask_file, self._field_mask = None, None
        if with_mask:
            order = sort_roi_mask_files([r['filepath'] for r in with_mask], mask_alias=self.userinfo['mask_alias'],
                                        highres_alias=self.userinfo['highres_alias'], as_index=True)
            main = with_mask[order[0]]
            self.field_mask_file = main['mask_file']
            self._field_mask = to_igor_format(load_roi_mask_file(main['mask_file']).astype(np.int32))

        infos = OrderedDict()
        for r in rows:
            stim = self.stimuli.get(r['stim_name']) if r['stim_name'] else None
            notes = []
            if stim is None:
                notes.append(f"'{r['token']}' matches no Stimulus alias: with these stimuli it would not be a "
                             f"Presentation in the pipeline")
            same_stim = [o for o in rows if o is not r and o['stim_name'] == r['stim_name'] and r['stim_name']
                         and o['key']['cond1'] == r['key']['cond1'] and o['key']['region'] == r['key']['region']]
            if same_stim:
                notes.append(f"same stimulus and condition as {[o['token'] for o in same_stim]}: Presentation would "
                             f"raise 'Found multiple files'; add a condition (cond1/cond2) to separate them")
            mask_info = {}
            if r['mask_file'] is None:
                notes.append(f"no ROI mask file in {self.roi_mask_dir}/: no ROIs or traces for this presentation "
                             f"(RoiMask.RoiMaskPresentation has no entry)")
            else:
                pres_mask = to_igor_format(load_roi_mask_file(r['mask_file']).astype(np.int32))
                as_field_mask, (dx, dy) = compare_roi_masks(pres_mask, self._field_mask, max_shift=self.max_shift)
                mask_info = dict(as_field_mask=as_field_mask, shift_dx=int(dx), shift_dy=int(dy),
                                 roi_mask_file=os.path.relpath(r['mask_file'], self.experiment_dir),
                                 field_mask_file=os.path.relpath(self.field_mask_file, self.experiment_dir))
                if as_field_mask == 'different':
                    notes.append("mask differs from the field mask (as_field_mask='different'): Traces would refuse "
                                 "to populate this presentation")
            name = r['key']['stim_name'] + f"  [{r['key']['file']}]"
            channels = self._channels(r['filepath'])
            infos[name] = dict(info=PresentationInfo(
                name=name, key=r['key'], stim_name=r['stim_name'],
                stim_family='' if stim is None else stim.get('stim_family', ''), channels=channels,
                has_mask=r['mask_file'] is not None, notes=tuple(notes)),
                filepath=r['filepath'], mask_file=r['mask_file'], mask_info=mask_info, stim=stim)
        return infos

    @staticmethod
    def _channels(filepath: str) -> tuple:
        """Recorded channels from the header only (``.smh``: ``InputChannelMask``, ``.h5``: datasets), without the
        trigger channel."""
        from scanmsupport.scanm.scanm_smp import SMP
        from djimaging.utils.misc_utils import NoPrints
        try:
            if str(filepath).lower().endswith('.h5'):
                import h5py
                with h5py.File(filepath, 'r') as f:
                    names = sorted(k for k in f.keys() if k.startswith('wDataCh'))
                return tuple(c for c in names if c != TRIGGER_CHANNEL)
            smp = SMP()
            with NoPrints():
                smp.loadSMH(filepath, verbose=False)
            mask = int(smp._kvPairDict['InputChannelMask'][2])
            names = [f'wDataCh{i}' for i in range(8) if mask >> i & 1]
        except Exception:  # noqa: BLE001 (channel list is only a convenience; loading checks the channel again)
            names = ['wDataCh0', 'wDataCh1']
        return tuple(c for c in names if c != TRIGGER_CHANNEL)

    def list_presentations(self) -> list:
        return [v['info'] for v in self._infos.values()]

    # ---------------------------------------------------------------- loading
    def load_presentation(self, name: str, channel: str = 'wDataCh0') -> PresentationData:
        cache_key = (name, channel)
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]
        data = self._load_presentation(name, channel)
        self._cache[cache_key] = data
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return data

    def _load_presentation(self, name: str, channel: str) -> PresentationData:
        with _read_each_smp_once():
            return self._load_presentation_uncached(name, channel)

    def _load_presentation_uncached(self, name: str, channel: str) -> PresentationData:
        from djimaging.utils.mask_format_utils import to_igor_format
        from djimaging.utils.mask_utils import load_roi_mask_file
        from djimaging.utils.scanm.recording import ScanMRecording

        entry = self._infos[name]
        info, stim, filepath = entry['info'], entry['stim'], entry['filepath']
        notes = list(info.notes)
        isrepeated = None if stim is None else bool(stim['isrepeated'])
        ntrigger_rep = None if stim is None else int(stim['ntrigger_rep'])

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            rec = ScanMRecording(filepath=filepath, setup_id=self.setup_id, date=self.date, repeated_stim=isrepeated,
                                 ntrigger_rep=ntrigger_rep, time_precision=self.trigger_precision)
            _compute_triggers_like_presentation(rec, isrepeated, ntrigger_rep)
        if channel not in rec.ch_stacks:
            raise KeyError(f"channel {channel!r} not in {sorted(rec.ch_stacks)}")
        triggers = np.asarray(rec.trigger_times, dtype=float)
        if rec.trigger_valid is not None and not rec.trigger_valid:
            notes.append(f"trigger_valid=0: {triggers.size} triggers, expected {ntrigger_rep}"
                         f"{' per repetition' if isrepeated else ''}")

        meta = dict(absx=rec.pos_x_um, absy=rec.pos_y_um, absz=rec.pos_z_um, nxpix=rec.pix_nx, nypix=rec.pix_ny,
                    zoom=rec.obj_zoom, angle_deg=rec.obj_angle_deg, scan_type=rec.scan_type,
                    setup_id=self.setup_id, stimulator_delay_s=rec.stimulator_delay, file=os.path.basename(filepath))

        rois, signals, quality, roi_notes = None, {}, None, {}
        quality_unavailable = ''
        if entry['mask_file'] is not None:
            pres_mask_xy = to_igor_format(load_roi_mask_file(entry['mask_file']).astype(np.int32))
            rois = RoiLabels.from_igor_xy(pres_mask_xy)
            if entry['mask_info'].get('as_field_mask') != 'different':
                signals, quality, roi_notes, quality_unavailable = self._traces_and_quality(
                    filepath, channel, pres_mask_xy, rec, triggers, stim, info)
            else:
                quality_unavailable = "no traces (mask differs from the field mask)"
        else:
            quality_unavailable = "no ROI mask for this presentation"

        return PresentationData(
            info=info, channel=channel, frames=to_display(rec.ch_stacks[channel]).astype(np.float32),
            frame_times=rec.frame_times, frame_dt=rec.frame_dt, rois=rois, signals=signals, triggers=triggers,
            trigger_labels=trigger_labels_for(triggers, stim), stimulus=stimulus_info_for(stim, SNIPPETS_PAD_TRACE),
            quality=quality, quality_unavailable=quality_unavailable, roi_notes=roi_notes, mask_info=entry['mask_info'],
            pixel_size_um=float(rec.pix_dx_um), npixartifact=int(rec.pix_n_artifact or 0), meta=meta,
            notes=tuple(notes), source=f"local files ({self.experiment_dir})")

    def _traces_and_quality(self, filepath, channel, pres_mask_xy, rec, triggers, stim, info):
        from djimaging.tables.core.preprocesstraces import process_trace
        from djimaging.utils.scanm.traces_and_triggers_utils import (
            check_valid_triggers_rel_to_tracetime, roi2trace_from_stack)
        from djimaging.utils.snippet_utils import split_trace_by_reps

        field_roi_ids = np.unique(np.abs(self._field_mask[self._field_mask < 0]))  # Roi table: IDs of the field mask
        as_field_mask = self._infos[info.name]['mask_info'].get('as_field_mask')
        roi2trace, frame_dt = roi2trace_from_stack(
            filepath=filepath, roi_ids=field_roi_ids, roi_mask=pres_mask_xy, data_stack_name=channel,
            precision=self.trace_precision, from_raw_data=self.from_raw_data,
            accept_missing_rois=as_field_mask != 'same')

        n_art = int(rec.pix_n_artifact or 0)
        family = quality_family(info.stim_name, info.stim_family)
        do_quality = family is not None and stim is not None and stim.get('isrepeated') and triggers.size > 0
        raw, pp, roi_notes, qvals, qextra, qref = {}, {}, {}, {}, {'ds_index': {}, 'os_index': {}}, None
        for roi_id, roi_data in roi2trace.items():
            roi_id = int(roi_id)
            if not self.include_artifacts and np.any(-pres_mask_xy[:n_art, :] == roi_id):
                roi_notes[roi_id] = "no trace: ROI touches the light-artifact lines (include_artifacts=0)"
                continue
            if not roi_data['trace_valid']:
                roi_notes[roi_id] = "no trace: trace_valid=0"
                continue
            trace = np.asarray(roi_data['trace'], dtype=float)
            t0 = float(roi_data['trace_times'][0])
            raw[roi_id] = Trace(sample_times(t0, frame_dt, trace.size), trace)
            trigger_valid = check_valid_triggers_rel_to_tracetime(1, roi_data['trace_times'], triggers)
            if not trigger_valid or triggers.size == 0:
                roi_notes[roi_id] = "no preprocessed trace: trigger_valid=0 (PreprocessTraces skips it)"
                continue
            p = self.preprocess
            pp_trace, _, pp_dt = process_trace(
                trace=trace.copy(), trace_t0=t0, trace_dt=frame_dt, stim_start=triggers[0],
                poly_order=p['poly_order'], window_len_seconds=p['window_len_seconds'],
                subtract_baseline=p['subtract_baseline'], standardize=p['standardize'], non_negative=p['non_negative'],
                f_cutoff=p['f_cutoff'], fs_resample=p['fs_resample'], baseline_max_dt=p['baseline_max_dt'])
            pp_times = sample_times(t0, pp_dt, pp_trace.size)
            pp[roi_id] = Trace(pp_times, pp_trace)
            if not do_quality:
                continue
            try:
                snippets, _, _, _ = split_trace_by_reps(pp_trace, pp_times, triggers, int(stim['ntrigger_rep']),
                                                        delay=0., allow_drop_last=True, pad_trace=SNIPPETS_PAD_TRACE)
                q = compute_quality(family, snippets, stim, pp_dt)
            except Exception as e:  # noqa: BLE001 (a failing ROI is reported as missing value)
                roi_notes[roi_id] = f"QI missing: {type(e).__name__}: {e}"
                continue
            qvals[roi_id] = q['value']
            qref = q.get('reference', qref)
            for k in qextra:
                if k in q:
                    qextra[k][roi_id] = q[k]

        signals = {}
        if raw:
            signals['raw'] = Signal('raw', f'a.u. (ROI mean of {channel})', raw,
                                    'Traces.trace: mean over ROI pixels per frame, recomputed offline')
        if pp:
            signals['preprocessed'] = Signal(
                'preprocessed', 'SD of pre-stimulus baseline', pp,
                'PreprocessTraces.pp_trace: detrended (Savitzky-Golay 60 s), baseline subtracted, standardized; '
                'recomputed offline with default PreprocessParams')
        quality, unavailable = None, ''
        if family is None:
            unavailable = unsupported_reason(info.stim_name, info.stim_family)
        elif not do_quality:
            unavailable = "no triggers or stimulus is not repeated: QI cannot be computed"
        else:
            from djimaging.cellviz.quality import QUALITY_METRICS
            table, metric = QUALITY_METRICS[family]
            quality = QualityValues(
                table=table, metric=metric, values=qvals, reference=qref,
                extra={k: v for k, v in qextra.items() if v},
                source=f"offline recompute of {table}.{metric} (default Stimulus/PreprocessParams, assumed from file name)")
        return signals, quality, roi_notes, unavailable

    def load_highres(self, channel: str = 'wDataCh0') -> tuple[Optional[HighResImage], str]:
        """HighRes median image of ``channel`` (as ``HighRes.StackAverages``), or (None, reason)."""
        from djimaging.utils.scanm.recording import ScanMRecording
        if self.highres_file is None:
            return None, f"no HighRes file (alias {self.userinfo['highres_alias']!r}) for field {self.field}"
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            rec = ScanMRecording(filepath=self.highres_file, setup_id=self.setup_id, date=self.date)
        if channel not in rec.ch_stacks:
            return None, f"HighRes file has no channel {channel}"
        image = to_display(np.median(rec.ch_stacks[channel], axis=2))
        meta = dict(absx=rec.pos_x_um, absy=rec.pos_y_um, absz=rec.pos_z_um, nxpix=rec.pix_nx, nypix=rec.pix_ny,
                    zoom=rec.obj_zoom, angle_deg=rec.obj_angle_deg, scan_type=rec.scan_type,
                    file=os.path.basename(self.highres_file))
        return HighResImage(image, channel, float(rec.pix_dx_um), meta, source=self.highres_file), ''


@contextmanager
def _read_each_smp_once():
    """Within the context, ``read_smp_utils.load_smp_file`` returns the already loaded SMP object for a file.

    ``ScanMRecording`` (movie, triggers, scan geometry) and ``roi2trace_from_stack`` (traces) both read the same
    ``.smp`` file through this function; reading a functional recording takes several seconds. The pipeline functions
    are used unchanged, they just share one read. Not thread-safe; only used around one presentation load.
    """
    from djimaging.utils.scanm import read_smp_utils
    original = read_smp_utils.load_smp_file
    cache = {}

    def load_once(raw_file_path):
        key = os.path.normcase(os.path.abspath(str(raw_file_path)))
        if key not in cache:
            cache[key] = original(raw_file_path)
        return cache[key]

    read_smp_utils.load_smp_file = load_once
    try:
        yield
    finally:
        read_smp_utils.load_smp_file = original


def _compute_triggers_like_presentation(rec, isrepeated: Optional[bool], ntrigger_rep: Optional[int]) -> None:
    """Triggers as stored by ``PresentationTemplate._add_entry`` (``from_raw_data``): automatic threshold.

    ``_add_entry`` also has a loop that lowers the threshold if too few triggers are found, but it never updates its
    trigger count inside the loop, so it always restores the original threshold afterwards. The stored triggers are
    therefore those of the automatic threshold, which is what is reproduced here.
    """
    rec.compute_triggers(trigger_threshold='auto')
