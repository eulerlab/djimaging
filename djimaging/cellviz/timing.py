"""Timestamp lookups shared by the data model, the renderer and the video export.

All times are seconds on the clock of one presentation, as used by djimaging: frame ``i`` starts at
``i * frame_dt`` (``compute_frame_times``), a trace sample ``k`` of a ROI is at ``trace_t0 + k * trace_dt`` where
``trace_t0`` includes the ROI's line offset within the frame (``compute_traces``), and trigger times come from the
same pixel clock plus the stimulator delay (``compute_triggers``).

Synchronisation is done on timestamps only. Index ``i`` of one array is never assumed to correspond to index ``i``
of another. Frame ``i`` covers ``[frame_times[i], frame_times[i] + frame_dt)``; a trace sample belongs to the frame
whose interval contains its timestamp.
"""
from __future__ import annotations

import numpy as np


def check_increasing(times: np.ndarray, name: str) -> np.ndarray:
    """Return ``times`` as a 1-D float array, raising if it is empty, not finite or not strictly increasing."""
    times = np.asarray(times, dtype=float)
    if times.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {times.shape}")
    if times.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(times)):
        raise ValueError(f"{name} contains non-finite values")
    if np.any(np.diff(times) <= 0):
        raise ValueError(f"{name} must be strictly increasing")
    return times


def sample_times(t0: float, dt: float, n: int) -> np.ndarray:
    """Sample times of a trace stored as (``trace_t0``, ``trace_dt``, ``n`` samples), as in djimaging tables."""
    return float(t0) + np.arange(int(n)) * float(dt)


def frame_index_at(frame_times: np.ndarray, t: float) -> int:
    """Index of the latest frame starting at or before ``t``, clamped to the valid frame range."""
    idx = int(np.searchsorted(frame_times, t, side='right')) - 1
    return int(np.clip(idx, 0, len(frame_times) - 1))


def frame_range(frame_times: np.ndarray, frame_dt: float, t_start: float | None, t_stop: float | None) -> range:
    """Frames whose interval overlaps ``[t_start, t_stop)``. ``None`` means the start / end of the presentation.

    Used by both the GUI time range and the video export, so both show the same frames for the same range.
    """
    n = len(frame_times)
    i0 = 0 if t_start is None else frame_index_at(frame_times, t_start)
    if t_stop is None:
        i1 = n
    else:
        i1 = int(np.searchsorted(frame_times, t_stop, side='left'))  # frames starting before t_stop
        i1 = int(np.clip(i1, i0 + 1, n))
    return range(i0, i1)


def samples_in_interval(times: np.ndarray, t_start: float, t_stop: float) -> np.ndarray:
    """Indices of samples with ``t_start <= times < t_stop``."""
    lo = np.searchsorted(times, t_start, side='left')
    hi = np.searchsorted(times, t_stop, side='left')
    return np.arange(lo, hi)


def n_samples_before(times: np.ndarray, t_stop: float) -> int:
    """Number of samples with ``times < t_stop``; the visible part of a trace that builds up to ``t_stop``."""
    return int(np.searchsorted(times, t_stop, side='left'))
