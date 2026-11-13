"""Read the scan parameters of a ScanM recording without loading the stacks, and never fail on unusual recordings.

Used to register *every* recording of an experiment (ExpHistory), including z-stacks, line scans, optic disk and
high resolution recordings, recordings of unknown setups and recordings with odd zooms, for which `ScanMRecording`
raises errors.

The two file types use different names for the same parameters: `.smh` headers (ScanM, read via `scanmsupport`)
and the `wParamsNum` / `wParamsStr` tables in `.h5` files (exported by Igor). `normalize_header` maps both to the
same keys (`HEADER_KEYS`).
"""
from __future__ import annotations

import os
import warnings
from datetime import datetime

import h5py
import numpy as np

from djimaging.utils.scanm import read_h5_utils, setup_utils, wparams_utils

# Normalized key: (key in .smh header, key in .h5 wParams)
HEADER_KEYS = {
    'nxpix_full': ('FrameWidth', 'User_dxPix'),
    'nypix': ('FrameHeight', 'User_dyPix'),
    'nzpix': ('dZPixels', 'User_dzPix'),
    'nxpix_retrace': ('PixRetraceLen', 'User_nPixRetrace'),
    'nxpix_offset': ('XPixLineOffs', 'User_nXPixLineOffs'),
    'scan_type_id': ('ScanType', 'User_ScanType'),
    'scan_mode': ('ScanMode', 'User_ScanMode'),
    'n_frames_set': ('NumberOfFrames', None),
    'frame_counter': ('FrameCounter', None),
    'real_pix_dur_us': ('RealPixelDuration_us', 'RealPixDur'),
    'zoom': ('Zoom', 'Zoom'),
    'angle_deg': ('Angle_deg', 'Angle_deg'),
    'absx': ('XCoord_um', 'XCoord_um'),
    'absy': ('YCoord_um', 'YCoord_um'),
    'absz': ('ZCoord_um', 'ZCoord_um'),
    'z_step_um': ('ZStep_um', 'ZStep_um'),
    'n_fr_per_step': ('NFrPerStep', 'User_NFrPerStep'),
    'setupid': ('SetupID', 'User_SetupID'),
    'laser_wavelength_nm': ('LaserWavelength_nm', 'User_LaserWaveLen_nm'),
    'objective': ('Objective', 'User_Objective'),
    'user_comment': ('Comment', 'User_Comment'),
    'computer_name': ('ComputerName', 'ComputerName'),
    'scanm_version': ('ScanMproductVersionAndTargetOS', 'ScanM_PVer_TargetOS'),
    'igor_gui_version': ('IgorGUIVer', 'IgorGUIVer'),
    'input_ch_mask': ('InputChannelMask', 'InputChanMask'),
    'stim_ch_mask': ('StimulusChannelMask', 'StimChanMask'),
    'date_stamp': ('DateStamp', 'DateStamp_d_m_y'),
    'time_stamp': ('TimeStamp', 'TimeStamp_h_m_s_ms'),
    'orig_file_name': ('OriginalPixelDataFileName', 'OrigPixDataFileName'),  # path of the .smp at acquisition
}

_INT_KEYS = {'nxpix_full', 'nypix', 'nzpix', 'nxpix_retrace', 'nxpix_offset', 'scan_type_id', 'scan_mode',
             'n_frames_set', 'frame_counter', 'n_fr_per_step', 'setupid', 'input_ch_mask', 'stim_ch_mask'}
_FLOAT_KEYS = {'real_pix_dur_us', 'zoom', 'angle_deg', 'absx', 'absy', 'absz', 'z_step_um', 'laser_wavelength_nm'}
_STR_KEYS = {'objective', 'user_comment', 'computer_name', 'scanm_version', 'igor_gui_version',
             'date_stamp', 'time_stamp', 'orig_file_name'}


def _decode(value) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode('utf-8', errors='replace')
    return str(value)


def _clean_key(key: str) -> str:
    """Header keys with a micro sign are read with varying encodings (µ, �, ...). Use `us` instead."""
    if key.startswith('RealPixelDuration_') or key.startswith('TargetedPixelDuration_'):
        return key.split('_')[0] + '_us'
    return key


def _to_number(value, kind):
    try:
        if value is None or (isinstance(value, str) and value.strip() in ('', 'n/a')):
            return None
        value = float(value)
        if not np.isfinite(value):
            return None
        return int(round(value)) if kind is int else value
    except (TypeError, ValueError):
        return None


def to_plain(value):
    """Header value as plain Python type (str, int, float, list), e.g. to store it as blob."""
    if isinstance(value, (bytes, np.bytes_)):
        return _decode(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def normalize_header(raw: dict, source: str) -> dict:
    """Map a raw header (`source='smh'`: .smh key-value pairs, `source='h5'`: wParams) to `HEADER_KEYS`.

    Missing or unreadable values are None.
    """
    if source not in ('smh', 'h5'):
        raise ValueError(f"source must be 'smh' or 'h5', got {source!r}")
    raw = {_clean_key(k): v for k, v in raw.items()}
    idx = 0 if source == 'smh' else 1
    header = dict()
    for key, names in HEADER_KEYS.items():
        value = raw.get(names[idx]) if names[idx] is not None else None
        if key in _INT_KEYS:
            header[key] = _to_number(value, int)
        elif key in _FLOAT_KEYS:
            header[key] = _to_number(value, float)
        else:
            header[key] = None if value is None else _decode(value)
    return header


def parse_scanm_datetime(date_stamp: str | None, time_stamp: str | None) -> datetime | None:
    """Recording start from ScanM date and time stamps, e.g. '2019-1-30' and '19-44-4-6' (h-m-s-ms).

    Returns None if a stamp is missing or cannot be parsed.
    """
    if not date_stamp or not time_stamp:
        return None
    try:
        date = datetime.strptime(date_stamp.strip(), '%Y-%m-%d').date()
        parts = [int(p) for p in time_stamp.strip().replace(':', '-').replace('.', '-').split('-')]
        h, m, s = parts[:3]
        ms = parts[3] if len(parts) > 3 else 0
        return datetime(date.year, date.month, date.day, h, m, s, min(ms, 999) * 1000)
    except (ValueError, IndexError):
        return None


def scan_type_from_dims(nypix: int | None, nzpix: int | None) -> str:
    """'xy', 'xz', 'xyz', 'line' (neither y nor z) or 'unknown' (dimensions missing). Never raises."""
    if nypix is None and nzpix is None:
        return 'unknown'
    ny, nz = (nypix or 0), (nzpix or 0)
    if ny > 1 and nz <= 1:
        return 'xy'
    if ny <= 1 and nz > 1:
        return 'xz'
    if ny > 1 and nz > 1:
        return 'xyz'
    return 'line'


def frame_duration_s(header: dict, scan_type: str) -> float | None:
    """Duration of one frame (laser scanning time) in seconds, from the full line length and the pixel duration."""
    nx, pix_dur_us = header.get('nxpix_full'), header.get('real_pix_dur_us')
    if not nx or not pix_dur_us:
        return None
    ny, nz = (header.get('nypix') or 1), (header.get('nzpix') or 1)
    n_lines = {'xy': ny, 'xz': nz, 'xyz': ny * nz, 'line': 1}.get(scan_type)
    if n_lines is None:
        return None
    return float(nx * n_lines * pix_dur_us * 1e-6)


def n_frames_from_smh(header: dict) -> int | None:
    """Number of recorded frames from an .smh header: `NumberOfFrames - FrameCounter`.

    `NumberOfFrames` is the number of frames set in ScanM, `FrameCounter` the number of frames that were left when
    the recording was stopped (0 if it ran to the end). Checked against the stack length of the exported .h5 files.
    """
    n_set = header.get('n_frames_set')
    if n_set is None:
        return None
    counter = header.get('frame_counter') or 0
    return int(n_set - counter) if 0 <= counter <= n_set else int(n_set)


def setup_scan_width_um(setupid: int | None) -> float | None:
    """Scan width at zoom 1 from the setupid constant of `setup_utils`, None if the setup is not supported.
    Fallback only: if the schema has an imaging setup table (ExperimentImagingSetup), use its dated value instead."""
    if setupid is None:
        return None
    try:
        return float(setup_utils.get_setup_xscale(setupid))
    except (AssertionError, ValueError, TypeError):
        return None


def pixel_size_um(header: dict, scan_width_um: float | None = None) -> float | None:
    """Pixel size in um: `scan_width_um / npix / zoom` (as for Field).

    Parameters
    ----------
    header : dict
        `normalize_header(...)` of the recording.
    scan_width_um : float | None, optional
        Full-field scan width at zoom 1, e.g. from `ExperimentImagingSetup`. If None, the setupid constant of the
        recording header is used (`setup_scan_width_um`).

    Returns
    -------
    float | None
        None if the scan width (unknown setup), the zoom or the number of pixels is missing or not supported.
    """
    nx = header.get('nxpix_full')
    if scan_width_um is None:
        scan_width_um = setup_scan_width_um(header.get('setupid'))
    if nx is None or scan_width_um is None or header.get('zoom') is None:
        return None
    nxpix = nx - (header.get('nxpix_retrace') or 0) - (header.get('nxpix_offset') or 0)
    try:
        return float(setup_utils.pixel_size_from_scan_width(scan_width_um=scan_width_um, npix=nxpix,
                                                            zoom=header['zoom']))
    except (AssertionError, ValueError, TypeError):
        return None


def read_smh_raw(smp_path: str) -> dict:
    """All key-value pairs of the .smh header that belongs to `smp_path` (.smp or .smh). Does not load the data."""
    from scanmsupport.scanm.scanm_smp import SMP
    from djimaging.utils.misc_utils import NoPrints

    raw_file = SMP()
    with NoPrints():
        raw_file.loadSMH(str(smp_path), verbose=False)
    return {k: v[2] for k, v in raw_file._kvPairDict.items()}


def count_triggers_in_stack(stack: np.ndarray, wparams: dict) -> int | None:
    """Number of triggers in a trigger channel stack (x, y, frames), as `ScanMRecording.compute_triggers` would find
    them with an automatic threshold. 0 for a flat channel. None if they cannot be computed (e.g. z-stacks)."""
    if stack is None or stack.size == 0:
        return None
    vmin, vmax = float(np.min(stack)), float(np.max(stack))
    if vmax - vmin < 5_000:  # Trigger pulses go from ~0 to >30_000; no pulses
        return 0
    threshold = 30_000 if (vmin < 25_000 and vmax > 35_000) else 0.5 * (vmin + vmax)
    try:
        triggertimes, _ = wparams_utils.compute_triggers_from_wparams(
            stack=stack, wparams=wparams, stimulator_delay=0., threshold=threshold, precision='line')
    except Exception:
        return None
    return int(triggertimes.size)


def read_h5_header(h5_path: str, count_triggers: bool = True) -> dict:
    """Header of an .h5 recording without loading the stacks (except the trigger channel, see `count_triggers`).

    Returns
    -------
    dict
        `is_scanm`: False if the file has no wParams (not a ScanM recording, e.g. a morphology or Igor root file).
        `raw`: wParams as plain Python values, `header`: `normalize_header(raw, 'h5')`,
        `n_frames`: length of the first data channel, `ch_names`: data channels,
        `n_triggers`: number of entries in 'Triggertimes' if present, else counted in the trigger channel
        `wDataCh2` (if `count_triggers`), else None.
    """
    with h5py.File(h5_path, 'r') as h5_file:
        keys = list(h5_file.keys())
        if not any(k.lower() == 'wparamsnum' for k in keys):
            return dict(is_scanm=False)

        raw = read_h5_utils.extract_wparams(h5_file, lower_keys=False)
        ch_names = sorted(k for k in keys if k.startswith('wDataCh') and k[7:].isdigit())
        n_frames = int(h5_file[ch_names[0]].shape[-1]) if len(ch_names) > 0 else None

        n_triggers = None
        if any(k.lower() == 'triggertimes' for k in keys):
            triggertimes, _ = read_h5_utils.extract_triggers(h5_file)
            n_triggers = int(np.sum(np.isfinite(np.asarray(triggertimes, dtype=float))))
        elif count_triggers and 'wDataCh2' in keys:
            n_triggers = count_triggers_in_stack(h5_file['wDataCh2'][()], {k.lower(): v for k, v in raw.items()})

    return dict(is_scanm=True, raw={k: to_plain(v) for k, v in raw.items()}, header=normalize_header(raw, 'h5'),
                n_frames=n_frames, ch_names=ch_names, n_triggers=n_triggers)


def count_triggers_smp(smp_path: str) -> int | None:
    """Count the triggers in the trigger channel (2) of an .smp file. Loads the whole file, so this is slow."""
    from djimaging.utils.scanm import read_smp_utils

    try:
        wparams, raw_file = read_smp_utils.load_wparams(smp_path, return_file=True, lower_keys=True)
        stack = raw_file.getData(ch=2, crop=True).T
    except Exception as e:
        warnings.warn(f"Could not load the trigger channel of {smp_path}: {e}")
        return None
    return count_triggers_in_stack(stack, wparams)


def file_size_and_mtime(path: str) -> tuple[int, datetime]:
    stat = os.stat(path)
    return int(stat.st_size), datetime.fromtimestamp(stat.st_mtime).replace(microsecond=0)
