import numpy as np

from djimaging.utils.filter_utils import resample_trace


def align_stim_to_trace(stim, stimtime, trace, tracetime):
    """Align stimulus and trace.
     Modified from RFEst"""
    dt = np.mean(np.diff(tracetime))

    valid_idxs = (tracetime >= stimtime[0]) & (tracetime <= (stimtime[-1] + np.max(np.diff(stimtime))))
    r_tracetime, aligned_trace = resample_trace(tracetime=tracetime[valid_idxs], trace=trace[valid_idxs], dt=dt)
    t0 = r_tracetime[0]

    num_repeats = np.array([np.sum((r_tracetime > t_a) & (r_tracetime <= t_b))
                            for t_a, t_b in
                            zip(stimtime, np.append(stimtime[1:], stimtime[-1] + np.max(np.diff(stimtime))))])
    aligned_stim = np.repeat(stim, num_repeats, axis=0)

    return aligned_stim, aligned_trace, dt, t0


def get_mean_dt(tracetime, rtol=0.1, raise_error=False) -> (float, bool):
    dts = np.diff(tracetime)
    dt = np.mean(dts)
    dt_max_diff = np.max(dts) - np.min(dts)
    is_consistent = (dt_max_diff / dt) <= rtol

    if raise_error and not is_consistent:
        raise ValueError(f"Inconsistent dts. dt_mean={dt:.3g}, dt_max={np.max(dts):.3g}, dt_min={np.min(dts):.3g}")

    return dt, is_consistent


def align_trace_to_stim(stim, stimtime, trace, tracetime):
    """Align stimulus and trace."""
    dt, is_consistent = get_mean_dt(stimtime, rtol=0.1, raise_error=True)

    t0 = stimtime[0]
    aligned_stim = stim

    aligned_trace = np.zeros(stimtime.size)
    for i, (t_a, t_b) in enumerate(zip(stimtime, np.append(stimtime[1:], stimtime[-1] + dt))):
        aligned_trace[i] = np.sum(trace[(tracetime >= t_a) & (tracetime < t_b)])

    return aligned_stim, aligned_trace, dt, t0
