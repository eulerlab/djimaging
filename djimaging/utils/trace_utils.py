import warnings

import numpy as np

from djimaging.utils import filter_utils


def get_mean_dt(tracetime, rtol_error=2.0, rtol_warning=1.0) -> (float, float):
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

    :param stim: Stimulus indexes or stimulus values
    :param stimtime: Stimulus times in seconds
    :param trace: Trace values
    :param tracetime: Trace times in seconds
    :param stim_lag_atol: Absolute tolerance for stimulus lag in seconds, if the trigger is not exactly aligned

    :return: aligned_stim, aligned_trace, dt_trace, t0, dt_rel_error
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
    """Align stimulus and trace."""
    dt, dt_rel_error = get_mean_dt(stimtime, rtol_error=np.inf, rtol_warning=0.5)
    t0 = stimtime[0]

    aligned_trace = np.zeros(stimtime.size)
    for i, (t_a, t_b) in enumerate(zip(stimtime, np.append(stimtime[1:], stimtime[-1] + dt))):
        # Take mean to not put more weight on lagged frames
        aligned_trace[i] = np.mean(trace[(tracetime >= t_a) & (tracetime < t_b)])

    return aligned_trace, dt, t0, dt_rel_error


def find_closest(target: float, data: np.ndarray, atol=np.inf, as_index=False):
    """Find the closest point in data"""
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
    """Find the closest point in sorted data that is larger"""
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
    """Find the closest point in sorted data that is lower"""
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
    """Traces (n_samples x n_time)"""
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
    """Traces (n_samples x n_time)"""
    assert traces.ndim == 2
    sort_idxs = argsort_traces(traces, ignore_nan=ignore_nan)
    return traces[sort_idxs, :]
