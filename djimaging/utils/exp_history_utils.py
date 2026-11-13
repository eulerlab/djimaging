"""Find, pair and describe all ScanM recordings of an experiment, and link them to fields and stimulus logs.

Used by the ExpHistory tables. All functions are independent of DataJoint.

A *recording* is one ScanM acquisition: the raw `.smp` (+ `.smh` header) file and/or the `.h5` file that Igor
exported from it (`SMP_<name>.h5`). Both are linked by the recording name `<name>`.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta, time
from typing import Iterable, Optional

import numpy as np

from djimaging.utils.filesystem_utils import get_file_info_row
from djimaging.utils.scanm import read_header_utils as rhu

H5_PREFIX = 'SMP_'
FIELD_KEY_DEFAULTS = dict(region='N/A', cond1='control', cond2='control', cond3='control')


LINK_METHODS = ('orig_file_name', 'file_name', 'time_stamp', 'none')


@dataclass
class ScanMFile:
    """One .smp (with its .smh) or .h5 file and its header (None until `load_header` was called)."""
    path: str
    smh: Optional[str] = None  # .smh file of an .smp file
    header: Optional[dict] = None  # normalize_header(...) of the .smh header or .h5 wParams
    smh_raw: Optional[dict] = None  # .smh: all key-value pairs
    h5_info: Optional[dict] = None  # .h5: read_header_utils.read_h5_header(...)
    error: Optional[str] = None  # error while reading the header

    @property
    def is_smp(self) -> bool:
        return self.path.lower().endswith('.smp')

    @property
    def filename(self) -> str:
        return os.path.basename(self.path)

    @property
    def stem(self) -> str:
        return recording_stem(self.path)

    @property
    def is_scanm(self) -> bool:
        """False for .h5 files without wParams (e.g. morphology or Igor files)."""
        return self.is_smp or self.h5_info is None or self.h5_info.get('is_scanm', True)

    def load_header(self) -> None:
        try:
            if self.is_smp:
                if self.smh is None:
                    raise FileNotFoundError('no .smh header file')
                self.smh_raw = {k: rhu.to_plain(v) for k, v in rhu.read_smh_raw(self.path).items()}
                self.header = rhu.normalize_header(self.smh_raw, 'smh')
            else:
                self.h5_info = rhu.read_h5_header(self.path)
                if self.h5_info['is_scanm']:
                    self.header = self.h5_info['header']
        except Exception as e:
            self.error = f"{os.path.splitext(self.path)[1]}: {e}"


@dataclass
class RecordingFiles:
    """Files of one recording, None if the file does not exist. `link_method`: how .smp and .h5 were linked."""
    rec_name: str
    smp: Optional[ScanMFile] = None
    h5: Optional[ScanMFile] = None
    link_method: str = 'none'

    @property
    def source(self) -> str:
        if self.smp is not None and self.h5 is not None:
            return 'smp+h5'
        return 'smp' if self.smp is not None else 'h5'


def recording_stem(filename: str) -> str:
    """Recording name of a file: without extension and, for .h5 files, without the `SMP_` prefix."""
    stem, ext = os.path.splitext(os.path.basename(filename))
    if ext.lower() == '.h5' and stem.upper().startswith(H5_PREFIX):
        stem = stem[len(H5_PREFIX):]
    return stem


def resolve_data_dirs(header_path: str, data_dirs: Iterable[str]) -> list[str]:
    """Absolute, existing, unique folders to scan: `data_dirs` are relative to the experiment folder `header_path`.
    Missing folders are skipped with a warning."""
    folders = []
    for data_dir in data_dirs:
        if data_dir is None or data_dir == '':
            continue
        folder = os.path.normpath(os.path.join(header_path, data_dir))
        if folder in folders:
            continue
        if not os.path.isdir(folder):
            warnings.warn(f"Folder `{data_dir}` not found in {header_path}")
            continue
        folders.append(folder)
    return folders


def list_scanm_files(folders: Iterable[str]) -> tuple[list[ScanMFile], list[ScanMFile]]:
    """All .smp files (with the .smh file of the same name in the same folder) and all .h5 files in `folders`.
    Not recursive, hidden files (starting with '.') are ignored. Headers are not read."""
    smps, h5s = [], []
    for folder in folders:
        filenames = sorted(f for f in os.listdir(folder)
                           if not f.startswith('.') and os.path.isfile(os.path.join(folder, f)))
        smhs = {os.path.splitext(f)[0].lower(): f for f in filenames if f.lower().endswith('.smh')}
        for filename in filenames:
            path = os.path.join(folder, filename)
            if filename.lower().endswith('.smp'):
                smh = smhs.get(os.path.splitext(filename)[0].lower())
                smps.append(ScanMFile(path=path, smh=os.path.join(folder, smh) if smh else None))
            elif filename.lower().endswith('.h5'):
                h5s.append(ScanMFile(path=path))
    return smps, h5s


def _unique_index(items: list, key) -> dict:
    """{key(item): item} for keys that occur exactly once and are not None."""
    groups = dict()
    for item in items:
        k = key(item)
        if k is not None:
            groups.setdefault(k, []).append(item)
    return {k: v[0] for k, v in groups.items() if len(v) == 1}


def _orig_smp_name(h5: ScanMFile) -> str | None:
    orig = (h5.header or {}).get('orig_file_name')
    if not orig:
        return None
    return orig.replace('\\', '/').split('/')[-1].lower()


def _time_stamp(f: ScanMFile) -> datetime | None:
    return rhu.parse_scanm_datetime((f.header or {}).get('date_stamp'), (f.header or {}).get('time_stamp'))


def link_smp_h5(smps: list[ScanMFile], h5s: list[ScanMFile]) -> list[RecordingFiles]:
    """Link .smp and .h5 files of the same recording. Each file is used once. In this order:

    1. 'orig_file_name': the .h5 wParam `OrigPixDataFileName` (path of the .smp at acquisition) ends with the name
       of the .smp file. Works also if the files were renamed differently.
    2. 'file_name': `SMP_<name>.h5` and `<name>.smp` (case-insensitive).
    3. 'time_stamp': same ScanM date and time stamp (millisecond resolution).

    A name or time stamp is only used if it is unique among the files not linked yet. .h5 files that are not ScanM
    recordings (no wParams, `header` loaded and `is_scanm` False) are dropped. Unlinked files get `link_method='none'`.
    The recording name is the .smp name (else the .h5 name without `SMP_`) without extension. If it is not unique,
    the name of the folder is prepended (`<folder>/<name>`).
    """
    h5s = [h for h in h5s if h.is_scanm]
    smps_left, h5s_left = list(smps), list(h5s)
    pairs = []

    for method, smp_key, h5_key in [
        ('orig_file_name', lambda s: s.filename.lower(), _orig_smp_name),
        ('file_name', lambda s: s.stem.lower(), lambda h: h.stem.lower()),
        ('time_stamp', _time_stamp, _time_stamp),
    ]:
        smp_index = _unique_index(smps_left, smp_key)
        h5_index = _unique_index(h5s_left, h5_key)
        for k in sorted(set(smp_index) & set(h5_index), key=str):
            smp, h5 = smp_index[k], h5_index[k]
            pairs.append((smp, h5, method))
            smps_left.remove(smp)
            h5s_left.remove(h5)

    pairs += [(s, None, 'none') for s in smps_left] + [(None, h, 'none') for h in h5s_left]

    recordings = [RecordingFiles(rec_name=(smp or h5).stem, smp=smp, h5=h5, link_method=method)
                  for smp, h5, method in pairs]
    names = [r.rec_name for r in recordings]
    for rec in recordings:
        if names.count(rec.rec_name) > 1:
            rec.rec_name = f"{os.path.basename(os.path.dirname((rec.smp or rec.h5).path))}/{rec.rec_name}"
    names = [r.rec_name for r in recordings]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"Recording names are not unique, even with folder names: {duplicates}")
    return sorted(recordings, key=lambda r: r.rec_name)


def find_recordings(folders: Iterable[str]) -> list[RecordingFiles]:
    """All recordings in `folders`: list the files, read their headers and link .smp and .h5 files."""
    smps, h5s = list_scanm_files(folders)
    for f in smps + h5s:
        f.load_header()
    return link_smp_h5(smps, h5s)


def find_unscanned_recording_files(header_path: str, folders: Iterable[str]) -> list[str]:
    """ScanM files (.smp, `SMP_*.h5`) anywhere in the experiment folder, but not in the scanned `folders`."""
    folders = {os.path.normpath(f) for f in folders}
    unscanned = []
    for folder, _, files in os.walk(header_path, followlinks=True):
        if os.path.normpath(folder) in folders:
            continue
        for filename in files:
            if filename.startswith('.'):
                continue
            if filename.lower().endswith('.smp') or (
                    filename.lower().endswith('.h5') and filename.upper().startswith(H5_PREFIX)):
                unscanned.append(os.path.join(folder, filename))
    return sorted(unscanned)


def parse_recording_name(filename: str, user_dict: dict) -> dict:
    """Parts of a file name as in Field discovery (field, stimulus, region, cond1, ...), plus `kind`.

    Uses the `*_loc` positions and aliases of `user_dict` (a UserInfo row). The positions refer to .h5 names
    (`SMP_` prefix); for .smp names they are shifted by one, as in `get_file_info_df`.
    `kind` is 'response', 'hr', 'od', 'outline', or 'unknown' if the name has no stimulus part. Never raises.
    """
    is_raw = filename.lower().endswith('.smp')
    loc_mapper = {k[:-4]: v - int(is_raw) for k, v in user_dict.items() if k.endswith('_loc') and v is not None}
    aliases = {k: (user_dict.get(f'{k}_alias') or '').lower().split('_')
               for k in ('opticdisk', 'highres', 'outline', 'mask')}
    try:
        info = get_file_info_row('', os.path.basename(filename), loc_mapper, aliases['opticdisk'],
                                 aliases['highres'], aliases['outline'], aliases['mask'])
    except Exception:
        from djimaging.utils.filesystem_utils import get_file_info_at_loc
        parts = os.path.splitext(os.path.basename(filename))[0].split('_')
        info = {name: get_file_info_at_loc(parts, loc) for name, loc in loc_mapper.items()}
        info['kind'] = 'unknown'
    return {k: v for k, v in info.items() if k not in ('filepath', 'mask_order')}


def read_recording(rec: RecordingFiles, count_smp_triggers: bool = False, scan_width_um: float | None = None) -> dict:
    """Describe one recording from its headers, without loading the stacks.

    The .smh header is used if available (it knows how many frames were set and recorded), else the .h5 wParams.
    Errors are not raised but collected in `header_issue`, so that every recording can be registered.

    Parameters
    ----------
    rec : RecordingFiles
        The recording.
    count_smp_triggers : bool, optional
        Count the triggers of .smp files without .h5 file (slow, loads the whole file).
    scan_width_um : float | None, optional
        Full-field scan width at zoom 1 for the pixel size, e.g. from `ExperimentImagingSetup`. If None, the setupid
        constant of the recording header is used.

    Returns
    -------
    dict
        Values for the ExpHistory columns (without key, files and name parts), see `ExpHistoryTemplate`.
    """
    for f in (rec.smp, rec.h5):
        if f is not None and f.header is None and f.error is None:
            f.load_header()

    errors = [f.error for f in (rec.smp, rec.h5) if f is not None and f.error is not None]
    smh_header = rec.smp.header if rec.smp is not None else None
    smh_raw = rec.smp.smh_raw if rec.smp is not None else None
    h5_info = rec.h5.h5_info if (rec.h5 is not None and rec.h5.header is not None) else None

    header = smh_header if smh_header is not None else (h5_info['header'] if h5_info is not None else None)
    if header is None:
        header = {k: None for k in rhu.HEADER_KEYS}

    n_frames = rhu.n_frames_from_smh(smh_header) if smh_header is not None else None
    n_frames_h5 = h5_info['n_frames'] if h5_info is not None else None
    if n_frames is None:
        n_frames = n_frames_h5
    elif n_frames_h5 is not None and n_frames_h5 != n_frames:
        errors.append(f'{n_frames} frames recorded according to .smh, but {n_frames_h5} frames in .h5')

    n_triggers = h5_info['n_triggers'] if h5_info is not None else None
    if n_triggers is None and count_smp_triggers and rec.smp is not None and smh_header is not None:
        n_triggers = rhu.count_triggers_smp(rec.smp.path)

    scan_type = rhu.scan_type_from_dims(header['nypix'], header['nzpix'])
    frame_dt_s = rhu.frame_duration_s(header, scan_type)
    nxpix = None
    if header['nxpix_full'] is not None:
        nxpix = max(0, header['nxpix_full'] - (header['nxpix_retrace'] or 0) - (header['nxpix_offset'] or 0))

    n_set = header.get('n_frames_set')
    entry = dict(
        rec_start=rhu.parse_scanm_datetime(header['date_stamp'], header['time_stamp']),
        n_frames=n_frames,
        n_frames_set=n_set,
        stopped_early=None if (n_set is None or n_frames is None) else bool(n_frames < n_set),
        frame_dt_s=frame_dt_s,
        rec_dur_s=None if (frame_dt_s is None or n_frames is None) else float(frame_dt_s * n_frames),
        scan_type=scan_type,
        scan_type_id=header['scan_type_id'],
        nxpix=nxpix,
        nypix=header['nypix'],
        nzpix=header['nzpix'],
        nxpix_full=header['nxpix_full'],
        n_fr_per_step=header['n_fr_per_step'],
        real_pix_dur_us=header['real_pix_dur_us'],
        zoom=header['zoom'],
        angle_deg=header['angle_deg'],
        scan_width_um=scan_width_um if scan_width_um is not None else rhu.setup_scan_width_um(header['setupid']),
        pixel_size_um=rhu.pixel_size_um(header, scan_width_um=scan_width_um),
        absx=header['absx'],
        absy=header['absy'],
        absz=header['absz'],
        z_step_um=header['z_step_um'],
        setupid=header['setupid'],
        laser_wavelength_nm=header['laser_wavelength_nm'],
        objective=(header['objective'] or '')[:191],
        user_comment=(header['user_comment'] or '')[:1000],
        computer_name=(header['computer_name'] or '')[:191],
        scanm_version=(header['scanm_version'] or '')[:191],
        input_ch_mask=header['input_ch_mask'],
        stim_ch_mask=header['stim_ch_mask'],
        n_triggers=n_triggers,
        header=dict(smh=smh_raw, h5=h5_info['raw'] if h5_info is not None else None,
                    h5_ch_names=h5_info['ch_names'] if h5_info is not None else None),
        header_issue='; '.join(errors)[:1000] if errors else None,
    )
    return entry


def match_field_recordings(name_infos: dict, field_key: dict, field_key_names: Iterable[str]) -> list[str]:
    """Recordings that belong to a field: all parts of the field key (field, region, cond1, ...) equal the parts of
    the file name. Missing region / conditions get the same defaults as in Field ('N/A', 'control').

    Parameters
    ----------
    name_infos : dict
        `{rec_name: parse_recording_name(...)}`.
    field_key : dict
        Primary key of the field.
    field_key_names : Iterable[str]
        Primary key names added by the Field table, e.g. `Field().new_primary_keys`.
    """
    matches = []
    for rec_name, info in name_infos.items():
        if info is None or info.get('field') is None:
            continue
        for name in field_key_names:
            value = info.get(name)
            if value is None:
                value = FIELD_KEY_DEFAULTS.get(name)
            if value != field_key[name]:
                break
        else:
            matches.append(rec_name)
    return sorted(matches)


def seconds_of_day(value) -> float | None:
    """Seconds since midnight of a datetime, time or timedelta (MySQL TIME is fetched as timedelta)."""
    if value is None:
        return None
    if isinstance(value, timedelta):
        return value.total_seconds()
    if isinstance(value, (datetime, time)):
        return value.hour * 3600 + value.minute * 60 + value.second + value.microsecond / 1e6
    raise TypeError(f"Unsupported time type {type(value)}")


def match_stimuli_to_recordings(
        rec_starts_s: Iterable[float], rec_ends_s: Iterable[float], stim_starts_s: Iterable[float],
        max_offset_s: float = 900., tolerance_s: float = 2.) -> tuple[float | None, list]:
    """Find the recording during which each logged stimulus was shown.

    QDSpy (stimulus PC) and ScanM (recording PC) have separate clocks. The clock offset is estimated as the shift of
    the stimulus start times (within +-`max_offset_s`) that puts most stimulus starts into a recording
    (from `tolerance_s` before its start until its end). Among equally good shifts, the one with most stimuli that
    start after (not before) their recording start, and then with the smallest total time between recording start
    and stimulus start is used. So it is a lower estimate of the true offset (stimuli usually start a few seconds
    after the recording), off by at most the smallest of these lags.

    Parameters
    ----------
    rec_starts_s, rec_ends_s : Iterable[float]
        Start and end of the recordings in seconds of the day (ScanM clock).
    stim_starts_s : Iterable[float]
        Start of the stimuli in seconds of the day (QDSpy clock).

    Returns
    -------
    clock_offset_s : float | None
        Add to QDSpy times to get ScanM times. None if no stimulus could be matched.
    matches : list
        Per stimulus: `(rec_idx, lag_s)`, the index of the recording and the time from recording start to stimulus
        start (ScanM clock), or `(None, None)` if the stimulus was shown while no recording was running.
    """
    rec_starts = np.asarray(list(rec_starts_s), dtype=float)
    rec_ends = np.asarray(list(rec_ends_s), dtype=float)
    stim_starts = np.asarray(list(stim_starts_s), dtype=float)
    no_match = [(None, None)] * stim_starts.size

    if rec_starts.size == 0 or stim_starts.size == 0:
        return None, no_match

    candidates = (rec_starts[:, None] - stim_starts[None, :]).ravel()
    candidates = np.unique(np.round(candidates[np.abs(candidates) <= max_offset_s], 3))
    if candidates.size == 0:
        return None, no_match

    best = None
    for offset in candidates:
        shifted = stim_starts + offset
        inside = (shifted[:, None] >= rec_starts[None, :] - tolerance_s) & (shifted[:, None] <= rec_ends[None, :])
        n_matched = int(np.sum(np.any(inside, axis=1)))
        if n_matched == 0:
            continue
        lags = np.where(inside, shifted[:, None] - rec_starts[None, :], np.inf)
        lags = lags[np.arange(lags.shape[0]), np.argmin(np.abs(lags), axis=1)]
        lags = lags[np.isfinite(lags)]
        # Most matches, then most stimuli that start after (not before) the recording start, then smallest lags
        score = (n_matched, int(np.sum(lags >= 0)), -float(np.sum(np.abs(lags))))
        if best is None or score > best[0]:
            best = (score, offset)

    if best is None:
        return None, no_match

    offset = float(best[1])
    matches = []
    for start in stim_starts + offset:
        inside = (start >= rec_starts - tolerance_s) & (start <= rec_ends)
        if not np.any(inside):
            matches.append((None, None))
            continue
        lags = np.where(inside, start - rec_starts, np.inf)
        idx = int(np.argmin(np.abs(lags)))
        matches.append((idx, float(lags[idx])))
    return offset, matches
