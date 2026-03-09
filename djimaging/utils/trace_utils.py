import warnings

import numpy as np

from djimaging.utils import filter_utils


def get_mean_dt(tracetime, rtol_error=2.0, rtol_warning=1.0) -> (float, float):
    """Compute the mean time-step of a time array and check for consistency.

    Parameters
    ----------
    tracetime : array-like
        1-D array of time stamps (in seconds), assumed to be monotonically
        increasing.
    rtol_error : float, optional
        Relative tolerance threshold for raising a ``ValueError``. When the
        maximum relative deviation of any individual step from the mean step
        exceeds this value the function raises an error. Default is 2.0.
    rtol_warning : float, optional
        Relative tolerance threshold for issuing a ``UserWarning``. Checked
        before `rtol_error`. Default is 1.0.

    Returns
    -------
    dt : float
        Mean time-step, i.e. ``mean(diff(tracetime))``.
    dt_rel_error : float
        Relative deviation of the most extreme time-step from the mean,
        i.e. ``max(dt_max - dt, dt - dt_min) / dt``.

    Raises
    ------
    ValueError
        If `dt_rel_error` is greater than or equal to `rtol_error`.
    """
    dts = np.diff(tracetime)
    dt = np.mean(dts)
    dt_rel_error = np.maximum(np.max(dts) - dt, dt - np.min(dts)) / dt

    if dt_rel_error >= rtol_error:
        raise ValueError(f"Inconsistent dts. dt_mean={dt:.3g}, but " +
                         f"dt_max={np.max(dts):.3g}, dt_min={np.min(dts):.3g}, dt_std={np.std(dts):.3g}")
    elif dt_rel_error >= rtol_warning:
        warnings.warn(f"Inconsistent dts. dt_mean={dt:.3g}, but " +
                      f"dt_max={np.max(dts):.3g}, dt_min={np.min(dts):.3g}, dt_std={np.std(dts):.3g}")

    return dt, dt_rel_error


def align_stim_to_trace(stim: np.ndarray, stimtime: np.ndarray, trace: np.ndarray, tracetime: np.ndarray,
                        stim_lag_atol: float = 0.003):
    """Align stimulus and trace.

    This function was modified from RFEst, which is licensed under
    the GNU General Public License version 3.0 (GNU GPL v3.0).

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <https://www.gnu.org/licenses/>.

    For details see: https://github.com/berenslab/RFEst

    ---

    Parameters
    ----------
    stim : np.ndarray
        Stimulus indexes or stimulus values, one entry per stimulus frame.
    stimtime : np.ndarray
        Stimulus frame onset times in seconds.
    trace : np.ndarray
        Recorded trace values.
    tracetime : np.ndarray
        Time stamps corresponding to `trace`, in seconds.
    stim_lag_atol : float, optional
        Absolute tolerance (in seconds) for a stimulus-onset lag. The
        stimulus time axis is shifted by this amount so that a small
        mismatch between trigger and trace clocks is absorbed. Must be
        non-negative and smaller than the stimulus frame duration.
        Default is 0.003.

    Returns
    -------
    aligned_stim : np.ndarray
        Stimulus array resampled to match the trace time grid.
    aligned_trace : np.ndarray
        Trace resampled to the mean trace time-step and trimmed to the
        stimulus window.
    dt_trace : float
        Mean time-step of the (resampled) trace.
    t0 : float
        Time of the first sample in the aligned trace.
    dt_rel_error : float
        Relative deviation of the most extreme trace time-step from the
        mean (see :func:`get_mean_dt`).

    Raises
    ------
    ValueError
        If `stim_lag_atol` is negative or not smaller than the stimulus
        frame duration.
    """
    dt_trace, dt_rel_error = get_mean_dt(tracetime, rtol_error=np.inf, rtol_warning=0.5)
    dt_stim, _ = get_mean_dt(stimtime, rtol_error=np.inf, rtol_warning=0.5)

    if stim_lag_atol < 0:
        raise ValueError(stim_lag_atol)
    if stim_lag_atol >= dt_stim:
        raise ValueError(stim_lag_atol)

    stimtime = np.append(stimtime, stimtime[-1] + dt_stim) - stim_lag_atol

    valid_idxs = (tracetime >= stimtime[0]) & (tracetime < stimtime[-1])
    r_tracetime, aligned_trace = filter_utils.resample_trace(
        tracetime=tracetime[valid_idxs], trace=trace[valid_idxs], dt=dt_trace)

    num_repeats = np.array([np.sum((r_tracetime >= t_a) & (r_tracetime < t_b))
                            for t_a, t_b in zip(stimtime[:-1], stimtime[1:])])
    aligned_stim = np.repeat(stim, num_repeats, axis=0)

    assert aligned_stim.shape[0] == aligned_trace.shape[0], (aligned_stim.shape[0], aligned_trace.shape[0])

    t0 = r_tracetime[0]

    return aligned_stim, aligned_trace, dt_trace, t0, dt_rel_error


def align_trace_to_stim(stimtime, trace, tracetime):
    """Align a trace to the stimulus time grid by averaging within each stimulus bin.

    For each stimulus frame interval ``[t_a, t_b)`` the mean of all trace
    samples whose time stamps fall inside that interval is assigned to the
    output sample at position ``t_a``.

    Parameters
    ----------
    stimtime : np.ndarray
        Stimulus frame onset times in seconds.
    trace : np.ndarray
        Recorded trace values.
    tracetime : np.ndarray
        Time stamps corresponding to `trace`, in seconds.

    Returns
    -------
    aligned_trace : np.ndarray
        Trace values averaged onto the stimulus time grid. Shape is
        ``(len(stimtime),)``.
    dt : float
        Mean time-step of the stimulus time grid.
    t0 : float
        Time of the first stimulus frame (``stimtime[0]``).
    dt_rel_error : float
        Relative deviation of the most extreme stimulus time-step from
        the mean (see :func:`get_mean_dt`).
    """
    dt, dt_rel_error = get_mean_dt(stimtime, rtol_error=np.inf, rtol_warning=0.5)
    t0 = stimtime[0]

    aligned_trace = np.zeros(stimtime.size)
    for i, (t_a, t_b) in enumerate(zip(stimtime, np.append(stimtime[1:], stimtime[-1] + dt))):
        # Take mean to not put more weight on lagged frames
        aligned_trace[i] = np.mean(trace[(tracetime >= t_a) & (tracetime < t_b)])

    return aligned_trace, dt, t0, dt_rel_error


def find_closest(target: float, data: np.ndarray, atol=np.inf, as_index=False):
    """Find the element in `data` that is closest to `target`.

    Parameters
    ----------
    target : float
        The reference value to search for.
    data : np.ndarray
        1-D array of values to search in.
    atol : float, optional
        Maximum allowed absolute distance between `target` and the closest
        element. Default is ``np.inf`` (no constraint).
    as_index : bool, optional
        If True, return the integer index of the closest element instead
        of its value. Default is False.

    Returns
    -------
    closest : float or int
        Value of the closest element in `data`, or its index when
        `as_index` is True.

    Raises
    ------
    ValueError
        If the closest element is farther from `target` than `atol`.
    """
    closest_index = np.argmin(np.abs(target - data))
    closest = data[closest_index]

    eps = np.abs(target - closest)
    if eps > atol:
        raise ValueError(
            f'Closest point={closest:.3g} is too far from target={target:.3g} by delta={eps:.3g} vs. atol={atol:.3g}. '
            f'Data range is [{data[0]:.3g}, {data[-1]:.3g}].'
        )
    if as_index:
        return closest_index
    else:
        return closest


def find_closest_after(target: float, data: np.ndarray, atol=np.inf, as_index=False):
    """Find the closest element in `data` that is greater than or equal to `target`.

    Parameters
    ----------
    target : float
        The reference value.
    data : np.ndarray
        1-D sorted array of values to search in.
    atol : float, optional
        Maximum allowed absolute distance between `target` and the
        returned element. Default is ``np.inf`` (no constraint).
    as_index : bool, optional
        If True, return the integer index of the element in `data`
        instead of its value. Default is False.

    Returns
    -------
    closest : float or int
        Value of the closest element >= `target` in `data`, or its index
        when `as_index` is True.

    Raises
    ------
    ValueError
        If no element within `atol` of `target` is found.
    """
    data_larger = data[data >= target]
    closest_index_larger = np.argmin(data_larger - target)
    closest = data_larger[closest_index_larger]
    closest_index = np.argmax(data == closest)

    eps = np.abs(target - closest)
    if eps > atol:
        raise ValueError(f'Did not find any point being close enough {eps} vs. atol={atol}')
    if as_index:
        return closest_index
    else:
        return closest


def find_closest_before(target: float, data: np.ndarray, atol=np.inf, as_index=False):
    """Find the closest element in `data` that is less than or equal to `target`.

    Parameters
    ----------
    target : float
        The reference value.
    data : np.ndarray
        1-D sorted array of values to search in.
    atol : float, optional
        Maximum allowed absolute distance between `target` and the
        returned element. Default is ``np.inf`` (no constraint).
    as_index : bool, optional
        If True, return the integer index of the element in `data`
        instead of its value. Default is False.

    Returns
    -------
    closest : float or int
        Value of the closest element <= `target` in `data`, or its index
        when `as_index` is True.

    Raises
    ------
    ValueError
        If no element within `atol` of `target` is found.
    """
    data_lower = data[data <= target]
    closest_index_larger = np.argmin(target - data_lower)
    closest = data_lower[closest_index_larger]
    closest_index = np.argmax(data == closest)

    eps = np.abs(target - closest)
    if eps > atol:
        raise ValueError(f'Did not find any point being close enough {eps} vs. atol={atol}')
    if as_index:
        return closest_index
    else:
        return closest


def argsort_traces(traces, ignore_nan=False):
    """Return indices that sort traces by similarity to the most correlated trace.

    A reference trace is chosen as the row whose sum of pairwise
    correlations with all other rows is maximal.  The remaining rows are
    then sorted by their correlation to that reference.

    Parameters
    ----------
    traces : np.ndarray
        2-D array of shape ``(n_samples, n_time)`` where each row is one
        trace.
    ignore_nan : bool, optional
        If True, columns that contain any non-finite value are removed
        before computing correlations. Default is False.

    Returns
    -------
    sort_idxs : np.ndarray
        Integer array of shape ``(n_samples,)`` with the row indices
        sorted from least to most correlated with the reference trace.
    """
    assert traces.ndim == 2, traces.ndim

    if ignore_nan:
        traces = traces[:, np.all(np.isfinite(traces), axis=0)]

    if traces.shape[0] <= 2:
        return np.arange(traces.shape[0])

    ccs = np.corrcoef(traces)
    ref_idx = np.argmax(np.sum(ccs, axis=0))
    sort_idxs = np.argsort(ccs[ref_idx])
    return sort_idxs


def sort_traces(traces, ignore_nan=False):
    """Sort traces by similarity to the most correlated trace.

    Applies :func:`argsort_traces` and returns the reordered array.

    Parameters
    ----------
    traces : np.ndarray
        2-D array of shape ``(n_samples, n_time)`` where each row is one
        trace.
    ignore_nan : bool, optional
        If True, columns that contain any non-finite value are removed
        before computing correlations. Default is False.

    Returns
    -------
    np.ndarray
        2-D array with the same shape as `traces` but with rows reordered
        from least to most correlated with the reference trace.
    """
    assert traces.ndim == 2
    sort_idxs = argsort_traces(traces, ignore_nan=ignore_nan)
    return traces[sort_idxs, :]
