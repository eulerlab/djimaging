import numpy as np

from djimaging.utils.filter_utils import resample_trace


def get_mean_dt(tracetime, rtol_error=1.0) -> (float, float):
    dts = np.diff(tracetime)
    dt = np.mean(dts)
    dt_rel_error = np.maximum(np.max(dts) - dt, dt - np.min(dts)) / dt

    if dt_rel_error >= rtol_error:
        raise ValueError(f"Inconsistent dts. dt_mean={dt:.3g}, but " +
                         f"dt_max={np.max(dts):.3g}, dt_min={np.min(dts):.3g}, dt_std={np.std(dts):.3g}")
    return dt, dt_rel_error


def align_stim_to_trace(stim, stimtime, trace, tracetime):
    """Align stimulus and trace.
     Modified from RFEst"""
    dt, dt_rel_error = get_mean_dt(tracetime)

    valid_idxs = (tracetime >= stimtime[0]) & (tracetime <= (stimtime[-1] + np.max(np.diff(stimtime))))
    r_tracetime, aligned_trace = resample_trace(tracetime=tracetime[valid_idxs], trace=trace[valid_idxs], dt=dt)
    t0 = r_tracetime[0]

    num_repeats = np.array([np.sum((r_tracetime > t_a) & (r_tracetime <= t_b))
                            for t_a, t_b in
                            zip(stimtime, np.append(stimtime[1:], stimtime[-1] + np.max(np.diff(stimtime))))])
    aligned_stim = np.repeat(stim, num_repeats, axis=0)

    return aligned_stim, aligned_trace, dt, t0, dt_rel_error


def align_trace_to_stim(stim, stimtime, trace, tracetime):
    """Align stimulus and trace."""
    dt, dt_rel_error = get_mean_dt(stimtime)

    t0 = stimtime[0]
    aligned_stim = stim

    aligned_trace = np.zeros(stimtime.size)
    for i, (t_a, t_b) in enumerate(zip(stimtime, np.append(stimtime[1:], stimtime[-1] + dt))):
        aligned_trace[i] = np.sum(trace[(tracetime >= t_a) & (tracetime < t_b)])

    return aligned_stim, aligned_trace, dt, t0, dt_rel_error


def find_closest(target: float, data: np.ndarray, atol=np.inf, as_index=False):
    """Find the closest point in data"""
    closest_index = np.argmin(np.abs(target - data))
    closest = data[closest_index]
    eps = np.abs(target - closest)
    if eps > atol:
        raise ValueError(f'Did not find any point being close enough {eps} vs. atol={atol}')
    if as_index:
        return closest_index
    else:
        return closest
