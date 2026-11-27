"""Stimulus geometry: from stimulus arrays and QDSpy screens to the recording field and the retina.

Coordinate frames
-----------------
- **Stimulus array** (e.g. dense noise `stim_trace`, a spatial RF, a QDSpy export frame): `x` is the column index
  (last axis), `y` is the row index (second to last axis). How these axes relate to the QDSpy screen is a property of
  the stimulus and is declared in `stim_dict['array_to_screen']`, e.g. `{'x': '+y', 'y': '+x'}`.
- **QDSpy screen**: `+x` is right and `+y` is up, as on the screen of the computer that designs the stimulus.
  Moving bar directions are defined in this frame (see `MB_CONVENTIONS`).
- **Field** (Igor/ScanM): `+x` is `XCoord_um` (from the back of the setup towards the curtain) and `+y` is `YCoord_um`
  (from left to right, standing at the curtain). ROI and field locations (`relx`, `rely`) use this frame.
  How the screen is projected into this frame is a property of the stimulator (`Stimulator` table).
- **Retina**: (dorsal, nasal) components, computed from the field frame with the retina orientation in the chamber
  (`prepwmorient`) and the eye, using the same function as the location tables (`get_retinal_position`).
  Retinal angles are in degrees, counterclockwise, 0° = nasal, 90° = dorsal, 180° = temporal, 270° = ventral.
  The angle relative to the optic disc is 0° for motion away from the optic disc (centrifugal) and 180° for motion
  towards it (centripetal), positive towards counterclockwise in the (nasal, dorsal) plane.

All maps between these frames are orthogonal (rotations by multiples of 90° and flips) and are represented as 2x2
integer matrices `M` with `M @ (a, b) = (c, d)`; column k of `M` is the image of the k-th source unit vector.

Confidence levels are 'high', 'medium' and 'low'; combined values are the worst level, and every downgrade comes with
a reason string.
"""
from __future__ import annotations

import datetime
from collections.abc import Iterable, Mapping

import numpy as np

from djimaging.utils.scanm.setup_utils import get_retinal_position

CONFIDENCE_LEVELS = ('high', 'medium', 'low')
AXIS_TOKENS = ('+x', '-x', '+y', '-y')

_AXIS_VECTORS = {'+x': (1, 0), '-x': (-1, 0), '+y': (0, 1), '-y': (0, -1)}

# Default stimulators. The orientation is documented in QDSpy-stimuli-Documentation (README, "Orientation") for
# 2022 to 05/2026: on setup 3 the QDSpy screen appears rotated by 90° counterclockwise in the chamber top view
# (dorsal/back up, curtain down), on setup 1 rotated by 90° counterclockwise and flipped upside-down. In the field
# frame (XCoord towards the curtain, YCoord to the right) this is:
#   setup 1: screen +x -> +XCoord, screen +y -> -YCoord
#   setup 3: screen +x -> -XCoord, screen +y -> -YCoord
# The same transforms are what the setup 1 and setup 3 branches of the former RfRoiOffset implemented.
# No change of these transforms is known before 2022, but it is not documented: experiments outside the documented
# period get 'medium' confidence. Setup 2 is assumed to be like setup 1 (not documented).
_DOC_SOURCE = 'QDSpy-stimuli-Documentation/README.md (Orientation, Color mapping)'
_DELAY_SOURCE = 'delay: djimaging/utils/scanm/stimulator_delay.csv'
_STIMULATOR_GEOMETRIES = {
    '1': dict(projection='QDSpy stimulator of setup 1, projected through the objective/condenser into the chamber',
              documented_from='2022-01-01', documented_to='2026-05-31', screen_x_axis='+x', screen_y_axis='-y',
              color_map='R:none, G:green, B:UV', confidence='high', source=_DOC_SOURCE),
    '2': dict(projection='QDSpy stimulator of setup 2 (rarely used; under construction as of 05/2026)',
              documented_from=None, documented_to=None, screen_x_axis='+x', screen_y_axis='-y',
              color_map='unknown', confidence='medium',
              source='Not documented; assumed to be like setup 1 (lab knowledge)'),
    '3': dict(projection='QDSpy stimulator of setup 3, projected through the objective/condenser into the chamber',
              documented_from='2022-01-01', documented_to='2026-05-31', screen_x_axis='-x', screen_y_axis='-y',
              color_map='R:UV, G:green, B:none', confidence='high', source=_DOC_SOURCE),
}


def stimulator_delay_periods(file: str | None = None) -> dict[str, list[tuple[str, str | None, float]]]:
    """Periods of constant stimulator delay per setup from stimulator_delay.csv (columns date;setup1;setup2;...).

    Returns {setupid: [(valid_from, valid_to or None, delay_ms), ...]}. A period starts at a row with a value for
    the setup and ends the day before the next such row (as `setup_utils.get_stimulator_delay`).
    """
    import pandas as pd
    if file is None:
        import os
        file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scanm', 'stimulator_delay.csv')
    df = pd.read_csv(file, sep=';', parse_dates=['date']).sort_values('date')
    periods = {}
    for col in df.columns:
        if not col.startswith('setup'):
            continue
        rows = df[['date', col]].dropna()
        starts = [d.date() for d in rows['date']]
        setup_periods = []
        for i, (start, delay_ms) in enumerate(zip(starts, rows[col])):
            end = (starts[i + 1] - datetime.timedelta(days=1)).isoformat() if i + 1 < len(starts) else None
            setup_periods.append((start.isoformat(), end, float(delay_ms)))
        periods[col.removeprefix('setup')] = setup_periods
    return periods


def _default_stimulators() -> tuple[dict, ...]:
    """One stimulator per setup and period of constant stimulator delay, with the documented geometry."""
    rows = []
    for setupid, geometry in _STIMULATOR_GEOMETRIES.items():
        for valid_from, valid_to, delay_ms in stimulator_delay_periods()[setupid]:
            rows.append(dict(
                stimulator_id=f'setup{setupid}_{valid_from}', setupid=setupid, valid_from=valid_from,
                valid_to=valid_to, offset_relx_um=0., offset_rely_um=0., um_per_px=None,
                stimulator_delay_ms=delay_ms, **{**geometry, 'source': f"{geometry['source']}; {_DELAY_SOURCE}"}))
    return tuple(rows)


DEFAULT_STIMULATORS = _default_stimulators()

# Imaging setups: scan width at zoom 1 and light artifact rows, formerly the setupid constants of
# djimaging.utils.scanm.setup_utils (get_setup_xscale, get_npixartifact). No date dependence is documented.
_IMAGING_SOURCE = 'djimaging.utils.scanm.setup_utils constants (no documented date dependence)'
DEFAULT_IMAGING_SETUPS = (
    dict(imaging_setup_id='setup1', setupid='1', valid_from='2000-01-01', valid_to=None, scan_width_um=112.,
         npixartifact=1, confidence='high', source=_IMAGING_SOURCE),
    dict(imaging_setup_id='setup2', setupid='2', valid_from='2000-01-01', valid_to=None, scan_width_um=71.5,
         npixartifact=4, confidence='medium', source=_IMAGING_SOURCE + '; scan width assumed as setup 3'),
    dict(imaging_setup_id='setup3', setupid='3', valid_from='2000-01-01', valid_to=None, scan_width_um=71.5,
         npixartifact=3, confidence='high',
         source=_IMAGING_SOURCE + '; some 2018 recordings have 4-6 artifact rows'),
)

# Moving bar direction conventions: velocity of the bar on the QDSpy screen (x right, y up) for script angle theta.
#   'rgc_movingbar' (RGC_MovingBar.py, rot_rad = theta - 90°): (-sin(theta), -cos(theta)), e.g. 0° moves down
#   'ds' (DS.py, rot_rad = theta):                              (-cos(theta), +sin(theta)), e.g. 0° moves left
# Both are linear in (cos(theta), sin(theta)), so a preferred direction (vector sum) maps with the same matrix.
MB_CONVENTIONS = {
    'rgc_movingbar': np.array([[0, -1], [-1, 0]]),
    'ds': np.array([[-1, 0], [0, 1]]),
}

# Moving bar versions (QDSpy stimulus md5) documented in QDSpy-stimuli-Documentation/stimuli/MovingBar.
DEFAULT_MB_VERSIONS = (
    dict(stim_md5='a574d90497d97ff9b93381f6b2eb207d', stim_file_name='RGC_MovingBar', dir_convention='rgc_movingbar',
         confidence='high', notes='Setup-1/Dominic/RGC_MovingBar.py and many others; documented directions'),
    dict(stim_md5='00bdaae87a8a50500c2063a68bfb3c1e', stim_file_name='DS', dir_convention='ds',
         confidence='high', notes='Setup-1/Katrin/RGCs/DS.py and Setup-3/Katrin/RGCs/DS.py; documented directions'),
)

# Array convention of QDSpy export frames and of images shown with matplotlib imshow (row 0 at the top)
QDSPY_EXPORT_ARRAY_TO_SCREEN = {'x': '+x', 'y': '-y'}
# Array convention of the standard dense noise (noise.h5 transposed, as loaded in all_rgcs/all_gcl_v2) and of the
# shifty noise (flipped on loading to match): array x is screen y and array y is screen x. This is what reproduces the
# former setup-specific branches of RfRoiOffset with the documented setup transforms.
NOISE_ARRAY_TO_SCREEN = {'x': '+y', 'y': '+x'}

CARDINALS = ('nasal', 'dorsal', 'temporal', 'ventral')  # at 0, 90, 180, 270 degrees


# ----------------------------------------------------------------------------------------------------------------------
# Confidence
# ----------------------------------------------------------------------------------------------------------------------
def combine_confidence(*levels: str) -> str:
    """Worst of the given confidence levels ('high' < 'medium' < 'low'); 'high' if none are given."""
    worst = 0
    for level in levels:
        if level not in CONFIDENCE_LEVELS:
            raise ValueError(f"Unknown confidence `{level}`, expected one of {CONFIDENCE_LEVELS}")
        worst = max(worst, CONFIDENCE_LEVELS.index(level))
    return CONFIDENCE_LEVELS[worst]


def join_reasons(*reasons: str | None) -> str:
    """Join non-empty reason strings with '; ', without duplicates."""
    out = []
    for reason in reasons:
        if reason and reason not in out:
            out.append(reason)
    return '; '.join(out)


# ----------------------------------------------------------------------------------------------------------------------
# Axis maps
# ----------------------------------------------------------------------------------------------------------------------
def axes_to_matrix(x_to: str, y_to: str) -> np.ndarray:
    """Matrix that maps source +x to `x_to` and source +y to `y_to` (each one of '+x', '-x', '+y', '-y')."""
    for token in (x_to, y_to):
        if token not in AXIS_TOKENS:
            raise ValueError(f"Unknown axis `{token}`, expected one of {AXIS_TOKENS}")
    if x_to[1] == y_to[1]:
        raise ValueError(f"Axes must be orthogonal, got x -> {x_to} and y -> {y_to}")
    return np.array([_AXIS_VECTORS[x_to], _AXIS_VECTORS[y_to]], dtype=int).T


def screen_to_field_matrix(screen_x_axis: str, screen_y_axis: str) -> np.ndarray:
    """Matrix from QDSpy screen (x right, y up) to the field frame (XCoord, YCoord) of a stimulator."""
    return axes_to_matrix(screen_x_axis, screen_y_axis)


def array_to_field_matrix(array_to_screen: Mapping[str, str], screen_x_axis: str, screen_y_axis: str) -> np.ndarray:
    """Matrix from stimulus array coordinates (x = column, y = row) to the field frame (XCoord, YCoord)."""
    if array_to_screen is None or 'x' not in array_to_screen or 'y' not in array_to_screen:
        raise ValueError(f"`array_to_screen` must define 'x' and 'y', got {array_to_screen}")
    m_arr = axes_to_matrix(array_to_screen['x'], array_to_screen['y'])
    return screen_to_field_matrix(screen_x_axis, screen_y_axis) @ m_arr


def array_to_field(arr: np.ndarray, array_to_screen: Mapping[str, str],
                   screen_x_axis: str, screen_y_axis: str) -> np.ndarray:
    """Reorient a stimulus frame or movie (..., rows, cols) to the field frame.

    The result has rows along +XCoord (back to curtain, i.e. dorsal at the top for prepwmorient=0) and columns along
    +YCoord (left to right), i.e. the setup-adjusted view of QDSpy-stimuli-Documentation. Only the last two axes are
    changed. Pixel sizes and cropping are not handled.
    """
    m = array_to_field_matrix(array_to_screen, screen_x_axis, screen_y_axis)
    out = np.asarray(arr)
    if m[0, 0] != 0:  # array x -> +-XCoord and array y -> +-YCoord: swap rows and columns
        out = np.swapaxes(out, -1, -2)
        flip_rows, flip_cols = m[0, 0] < 0, m[1, 1] < 0
    else:  # array y -> +-XCoord and array x -> +-YCoord
        flip_rows, flip_cols = m[0, 1] < 0, m[1, 0] < 0
    if flip_rows:
        out = np.flip(out, axis=-2)
    if flip_cols:
        out = np.flip(out, axis=-1)
    return out


def ellipse_to_frame(width: float, height: float, angle_deg: float, m: np.ndarray) -> tuple[float, float, float]:
    """Map an ellipse (width along its angle, matplotlib convention) through the orthogonal matrix `m`."""
    a = np.deg2rad(angle_deg)
    major = np.asarray(m) @ np.array([np.cos(a), np.sin(a)])
    return width, height, float(np.rad2deg(np.arctan2(major[1], major[0])))


# ----------------------------------------------------------------------------------------------------------------------
# Stimulator assignment
# ----------------------------------------------------------------------------------------------------------------------
def normalize_setupid(setupid) -> str:
    """'1', 1, '1.0' and ' 1 ' -> '1'. Other strings are stripped and lower-cased."""
    s = str(setupid).strip()
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except ValueError:
        pass
    return s.lower()


def _to_date(value) -> datetime.date | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, datetime.datetime):
        return value.date()
    if isinstance(value, datetime.date):
        return value
    return datetime.date.fromisoformat(str(value)[:10])


def assign_stimulator(setupid, date, stimulators: Iterable[Mapping]) -> tuple[str | None, str, str]:
    """Pick the stimulator of an experiment from its setup and date, see `assign_by_setup_and_date`."""
    return assign_by_setup_and_date(setupid, date, stimulators, id_key='stimulator_id', label='stimulator')


def assign_imaging_setup(setupid, date, imaging_setups: Iterable[Mapping]) -> tuple[str | None, str, str]:
    """Pick the imaging setup of an experiment from its setup and date, see `assign_by_setup_and_date`."""
    return assign_by_setup_and_date(setupid, date, imaging_setups, id_key='imaging_setup_id', label='imaging setup')


def assign_by_setup_and_date(setupid, date, rows: Iterable[Mapping], id_key: str,
                             label: str) -> tuple[str | None, str, str]:
    """Pick the row (stimulator, imaging setup) of an experiment from its setup and date.

    Parameters
    ----------
    setupid : str or int
        `setupid` of Experiment.ExpInfo.
    date : date-like
        Experiment date.
    rows : iterable of dict
        Rows with `id_key`, setupid, valid_from, valid_to, confidence and optionally documented_from, documented_to
        (both missing: no documented period is checked; both None: not documented).
    id_key : str
        Name of the identifier, e.g. 'stimulator_id'.
    label : str
        Used in reasons.

    Returns
    -------
    row_id : str or None
        None if no row or more than one row matches.
    confidence : str
    reason : str
        Empty for a row with 'high' confidence inside its documented period.
    """
    date = _to_date(date)
    setup = normalize_setupid(setupid)
    if setup in ('', 'none', 'nan', 'unknown'):
        return None, 'low', f"setupid `{setupid}` is unknown"

    matches = []
    for row in rows:
        if normalize_setupid(row['setupid']) != setup:
            continue
        valid_from, valid_to = _to_date(row.get('valid_from')), _to_date(row.get('valid_to'))
        if valid_from is not None and date < valid_from:
            continue
        if valid_to is not None and date > valid_to:
            continue
        matches.append(row)

    if len(matches) == 0:
        return None, 'low', f"No {label} for setup `{setupid}` on {date}"
    if len(matches) > 1:
        ids = sorted(str(r[id_key]) for r in matches)
        return None, 'low', f"Several {label}s for setup `{setupid}` on {date}: {ids}"

    row = matches[0]
    row_id = row[id_key]
    confidence = row['confidence']
    reasons = []
    if confidence != 'high':
        reasons.append(f"{label} `{row_id}` has {confidence} confidence ({row.get('source', '')})")

    if 'documented_from' in row or 'documented_to' in row:
        doc_from, doc_to = _to_date(row.get('documented_from')), _to_date(row.get('documented_to'))
        if doc_from is None and doc_to is None:
            if confidence == 'high':
                confidence = 'medium'
                reasons.append(f"geometry of {label} `{row_id}` is not documented")
        elif (doc_from is not None and date < doc_from) or (doc_to is not None and date > doc_to):
            confidence = combine_confidence(confidence, 'medium')
            reasons.append(f"{date} is outside the documented period ({doc_from} to {doc_to}) of {label} "
                           f"`{row_id}`; assumed unchanged")
    return row_id, confidence, join_reasons(*reasons)


# ----------------------------------------------------------------------------------------------------------------------
# Moving bar
# ----------------------------------------------------------------------------------------------------------------------
def infer_mb_convention(stim_md5: str | None, stim_file_name: str | None,
                        versions: Iterable[Mapping]) -> tuple[str | None, str, str]:
    """Direction convention of a moving bar recording from its QDSpy md5 (and file name for unknown versions).

    Returns
    -------
    convention : str or None
        Key of `MB_CONVENTIONS`, None if unknown.
    confidence : str
    reason : str
    """
    if stim_md5:
        for row in versions:
            if row['stim_md5'] == stim_md5:
                reason = '' if row['confidence'] == 'high' else \
                    f"MB version {stim_md5} has {row['confidence']} confidence ({row.get('notes', '')})"
                return row['dir_convention'], row['confidence'], reason

    if not stim_md5:
        prefix = "No QDSpy log (md5) for this presentation"
    else:
        prefix = f"Unknown MB version {stim_md5}"

    name = (stim_file_name or '').strip()
    lname = name.lower()
    if 'movingbar' in lname.replace('_', '').replace(' ', ''):
        return 'rgc_movingbar', 'medium', f"{prefix}; convention of RGC_MovingBar assumed from file name `{name}`"
    if lname == 'ds' or lname.startswith('ds_') or lname.startswith('ds.'):
        return 'ds', 'medium', f"{prefix}; convention of DS assumed from file name `{name}`"
    if name:
        return None, 'low', f"{prefix}; convention cannot be inferred from file name `{name}`"
    return None, 'low', f"{prefix}; no file name"


def script_to_screen_matrix(convention: str) -> np.ndarray:
    """Matrix from (cos(theta), sin(theta)) of the script angle theta to the bar velocity on the QDSpy screen."""
    if convention not in MB_CONVENTIONS:
        raise ValueError(f"Unknown moving bar convention `{convention}`, expected one of {list(MB_CONVENTIONS)}")
    return MB_CONVENTIONS[convention]


def angles_to_vectors(angles_deg) -> np.ndarray:
    """(n,) angles in degrees -> (n, 2) unit vectors (cos, sin)."""
    a = np.deg2rad(np.asarray(angles_deg, dtype=float))
    return np.stack([np.cos(a), np.sin(a)], axis=-1)


def vectors_to_angles(vectors) -> np.ndarray:
    """(n, 2) vectors -> angles in degrees in [0, 360)."""
    v = np.asarray(vectors, dtype=float)
    return np.mod(np.rad2deg(np.arctan2(v[..., 1], v[..., 0])), 360.)


def field_to_retina(field_vectors, prepwmorient: float, eye: str) -> np.ndarray:
    """(n, 2) vectors in the field frame (XCoord, YCoord) -> (n, 2) retinal (nasal, dorsal) components.

    Uses `get_retinal_position`, so directions and locations share one convention. Nasal components are NaN if the
    eye is not 'left' or 'right'.
    """
    v = np.atleast_2d(np.asarray(field_vectors, dtype=float))
    out = np.full(v.shape, np.nan)
    for i, (dx, dy) in enumerate(v):
        # Returns (ventral_dorsal, temporal_nasal) with positive = dorsal and nasal; linear, so valid for vectors
        dorsal, nasal = get_retinal_position(rel_xcoord_um=dx, rel_ycoord_um=dy, rotation=prepwmorient, eye=eye)
        out[i] = nasal, dorsal
    return out


def retinal_angle_deg(retinal_vectors) -> np.ndarray:
    """(n, 2) (nasal, dorsal) vectors -> angles in degrees, 0 = nasal, 90 = dorsal; NaN if the nasal part is NaN."""
    return vectors_to_angles(retinal_vectors)


def cardinal(angle_deg: float) -> str | None:
    """Nearest of nasal (0°), dorsal (90°), temporal (180°), ventral (270°); None for NaN."""
    if angle_deg is None or not np.isfinite(angle_deg):
        return None
    return CARDINALS[int(np.round(np.mod(angle_deg, 360.) / 90.)) % 4]


def wrap_deg(angle_deg):
    """Wrap angles to (-180, 180]."""
    a = np.mod(np.asarray(angle_deg, dtype=float) + 180., 360.) - 180.
    return np.where(a == -180., 180., a)


def angle_rel_optic_disc(retinal_angle: float, nasal_pos_um: float, dorsal_pos_um: float,
                         min_dist_um: float = 100.) -> float:
    """Angle of a retinal direction relative to the radial direction from the optic disc to the field.

    0 = away from the optic disc (centrifugal), +-180 = towards it. NaN if the field is closer than `min_dist_um` to
    the optic disc or any input is NaN.
    """
    if not (np.isfinite(retinal_angle) and np.isfinite(nasal_pos_um) and np.isfinite(dorsal_pos_um)):
        return np.nan
    if np.hypot(nasal_pos_um, dorsal_pos_um) < min_dist_um:
        return np.nan
    radial = np.rad2deg(np.arctan2(dorsal_pos_um, nasal_pos_um))
    return float(wrap_deg(retinal_angle - radial))


def retina_status(eye: str, prepwmorient) -> tuple[bool, bool, str, str]:
    """Which retinal components can be computed.

    Returns
    -------
    orientation_known, eye_known : bool
    confidence, reason : str
    """
    eye_known = eye in ('left', 'right')
    orientation_known = prepwmorient is not None and np.isfinite(prepwmorient) and int(prepwmorient) != -1
    reasons = []
    if not orientation_known:
        reasons.append("retina orientation (prepwmorient) unknown: no retinal direction")
    if not eye_known:
        reasons.append(f"eye `{eye}` unknown: only the dorsal-ventral component is known")
    confidence = 'high' if (orientation_known and eye_known) else 'low'
    return orientation_known, eye_known, confidence, join_reasons(*reasons)


def map_mb_directions(script_dirs_deg, convention: str | None, screen_x_axis: str | None, screen_y_axis: str | None,
                      prepwmorient, eye: str) -> dict:
    """Map moving bar script angles to field and retinal directions.

    Returns a dict with (n,) arrays `field_dirs_deg` (angle in the (XCoord, YCoord) plane, 0 = +XCoord = towards the
    curtain, 90 = +YCoord = to the right), `retinal_dirs_deg` (0 = nasal, 90 = dorsal) and `dorsal_components`
    (cosine of the angle to dorsal), each NaN where unknown, plus the 2x2 `script_to_field` and `script_to_retina`
    matrices (None where unknown). `script_to_retina` has NaN in the nasal row if the eye is unknown.
    """
    script_dirs_deg = np.asarray(script_dirs_deg, dtype=float)
    n = script_dirs_deg.size
    out = dict(field_dirs_deg=np.full(n, np.nan), retinal_dirs_deg=np.full(n, np.nan),
               dorsal_components=np.full(n, np.nan), script_to_field=None, script_to_retina=None)
    if convention is None or screen_x_axis is None or screen_y_axis is None:
        return out

    m_field = screen_to_field_matrix(screen_x_axis, screen_y_axis) @ script_to_screen_matrix(convention)
    out['script_to_field'] = m_field
    field_vectors = angles_to_vectors(script_dirs_deg) @ m_field.T
    out['field_dirs_deg'] = vectors_to_angles(field_vectors)

    orientation_known, eye_known, _, _ = retina_status(eye, prepwmorient)
    if not orientation_known:
        return out

    m_retina = field_to_retina(np.eye(2), prepwmorient=prepwmorient, eye=eye).T @ m_field
    out['script_to_retina'] = m_retina
    retinal_vectors = angles_to_vectors(script_dirs_deg) @ m_retina.T
    out['dorsal_components'] = retinal_vectors[:, 1]
    if eye_known:
        out['retinal_dirs_deg'] = retinal_angle_deg(retinal_vectors)
    return out


def map_pref_dir(pref_dir_rad: float, script_to_retina: np.ndarray | None) -> tuple[float, float]:
    """Map a preferred direction (radians, script angle convention as in OsDsIndexes) to the retina.

    Returns (retinal angle in degrees or NaN, dorsal component or NaN).
    """
    if script_to_retina is None or pref_dir_rad is None or not np.isfinite(pref_dir_rad):
        return np.nan, np.nan
    v = np.asarray(script_to_retina, dtype=float) @ np.array([np.cos(pref_dir_rad), np.sin(pref_dir_rad)])
    dorsal = float(v[1])
    angle = float(vectors_to_angles(v)) if np.isfinite(v[0]) else np.nan
    return angle, dorsal


# ----------------------------------------------------------------------------------------------------------------------
# Mouse cam
# ----------------------------------------------------------------------------------------------------------------------
def mc_version_from_params(params: Mapping | None) -> str | None:
    """MouseCam version from the logged movie names (QDSpy-stimuli-Documentation/stimuli/MouseCam).

    'v1': test_images_rand_* / train_images_*, 'v2': MC2_* without '-new', 'v2new': MC2_*-new, 'v3': MC3_*.
    None if no movie name is logged or the name is not recognised.
    """
    params = params or {}
    names = [str(params.get(k) or '') for k in ('movName_Test', 'movName_Train')]
    names = [n.replace('\\', '/').rsplit('/', 1)[-1].lower() for n in names if n]
    if len(names) == 0:
        return None
    versions = set()
    for name in names:
        if name.startswith('mc3'):
            versions.add('v3')
        elif name.startswith('mc2'):
            versions.add('v2new' if '-new' in name or '- new' in name else 'v2')
        elif name.startswith('test_images') or name.startswith('train_images'):
            versions.add('v1')
        else:
            versions.add(None)
    if len(versions) != 1:
        return None
    return versions.pop()


def mc_confidence(version: str | None, setupid) -> tuple[str, str]:
    """Confidence that a MouseCam presentation was shown as intended on the given setup."""
    setup = normalize_setupid(setupid)
    if version is None:
        return 'low', 'MouseCam version unknown (no or unrecognised movie name in the QDSpy log)'
    if version == 'v1':
        if setup == '3':
            return 'high', ''
        return 'low', f'MouseCam v1 assumes the color mapping of setup 3, but was shown on setup {setupid}'
    if setup != '1':
        return 'low', f'MouseCam {version} assumes the color mapping of setup 1, but was shown on setup {setupid}'
    if version == 'v2':
        return 'medium', ('MouseCam v2 is flipped on setup 1 (documented bug): the geometry of what was shown is '
                          'known, but the scene is not in its intended orientation')
    return 'high', ''
