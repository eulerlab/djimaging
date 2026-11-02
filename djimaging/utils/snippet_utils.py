import warnings

import numpy as np

from djimaging.utils.math_utils import truncated_vstack, padded_vstack
from djimaging.utils.trace_utils import find_closest, find_closest_before


def compute_t_idxs(trace, times, triggertimes, ntrigger_rep, delay=0., atol=0.128,
                   allow_drop_last=True, pad_trace=False, all_reps_same_length=True):
    """Compute repetition start indices, frames per repetition, and a drop-flag.

    Parameters
    ----------
    trace : np.ndarray
        1-D input trace data.
    times : np.ndarray
        1-D time stamps corresponding to `trace`.
    triggertimes : np.ndarray
        1-D array of trigger onset times.
    ntrigger_rep : int
        Number of triggers per stimulus repetition.
    delay : float, optional
        Delay in seconds added to each trigger time before searching for
        the closest trace index. Default is 0.0.
    atol : float, optional
        Absolute tolerance (in seconds) passed to the index-finding
        helper. Default is 0.128.
    allow_drop_last : bool, optional
        If True and the last repetition has insufficient data, it is
        silently dropped. If False a ``ValueError`` is raised instead.
        Default is True.
    pad_trace : bool, optional
        If True, use :func:`find_closest_before` so that trigger indices
        always fall within the trace (useful when the trace is slightly
        shorter than expected). Default is False.
    all_reps_same_length : bool, optional
        If True, enforce equal-length repetitions by truncating to the
        median repetition length. Default is True.

    Returns
    -------
    t_idxs : list of int
        Start indices into `trace` / `times` for each repetition.
    n_frames_per_rep : int
        Number of frames per repetition.
    dropped_last_rep_flag : bool
        True if the last repetition was dropped because it had
        insufficient data.

    Raises
    ------
    ValueError
        If `allow_drop_last` is False and the last repetition is
        incomplete.
    """
    if not pad_trace:
        t_idxs = [find_closest(target=tt + delay, data=times, atol=atol, as_index=True)
                  for tt in triggertimes[::ntrigger_rep]]
    else:
        t_idxs = [find_closest_before(target=tt + delay, data=times, atol=np.inf, as_index=True)
                  for tt in triggertimes[::ntrigger_rep]]

    if len(t_idxs) > 1 and all_reps_same_length:
        n_frames_per_rep = int(np.median(np.diff(t_idxs)))  # Use median rep length to ignore outliers
        if pad_trace:
            n_frames_per_rep += 1  # Heuristic to include frame from next repetition

        assert trace.shape == times.shape, 'Shapes do not match'

        if times[t_idxs[-1]:].size < n_frames_per_rep:
            if not allow_drop_last:
                raise ValueError('Data incomplete, allow to drop last repetition or fix data')
            # if there are not enough data points after the last trigger,
            # remove the last trigger (e.g. if a chirp was cancelled)
            dropped_last_rep_flag = True
            t_idxs.pop(-1)
        else:
            dropped_last_rep_flag = False
    else:
        n_frames_per_rep = times.size - t_idxs[0]
        dropped_last_rep_flag = False

    return t_idxs, n_frames_per_rep, dropped_last_rep_flag


def split_trace_by_reps(trace, times, triggertimes, ntrigger_rep, delay=0., atol=0.128, allow_drop_last=True,
                        pad_trace=False):
    """Split a trace into equal-length snippets aligned to trigger times.

    Parameters
    ----------
    trace : np.ndarray
        1-D recorded trace.
    times : np.ndarray
        1-D time stamps corresponding to `trace`.
    triggertimes : np.ndarray
        1-D array of trigger onset times.
    ntrigger_rep : int
        Number of triggers per stimulus repetition.
    delay : float, optional
        Delay in seconds added to each trigger time. Default is 0.0.
    atol : float, optional
        Absolute tolerance (in seconds) for matching trigger times to
        trace time stamps. Default is 0.128.
    allow_drop_last : bool, optional
        If True and the last repetition is incomplete, it is silently
        dropped. Default is True.
    pad_trace : bool, optional
        If True, use ``find_closest_before`` so that trigger indices
        always fall within the trace. Default is False.

    Returns
    -------
    snippets : np.ndarray
        2-D array of shape ``(n_frames_per_rep, n_reps)`` containing
        the trace values for each repetition.
    snippets_times : np.ndarray
        2-D array of shape ``(n_frames_per_rep, n_reps)`` containing
        the time stamps for each repetition.
    snippets_triggertimes : np.ndarray
        2-D array of shape ``(ntrigger_rep, n_reps)`` containing the
        trigger times for each repetition.
    droppedlastrep_flag : bool
        True if the last repetition was dropped.

    Raises
    ------
    ValueError
        If no trigger times are given or if `n_frames_per_rep` is
        invalid.
    """

    if len(triggertimes) <= 0:
        raise ValueError("No trigger times given")

    t_idxs, n_frames_per_rep, droppedlastrep_flag = compute_t_idxs(
        trace, times, triggertimes, ntrigger_rep,
        delay=delay, atol=atol, allow_drop_last=allow_drop_last, pad_trace=pad_trace)

    if n_frames_per_rep < 1:
        raise ValueError(f"n_frames_per_rep={n_frames_per_rep} is invalid for trigger times {triggertimes}")

    snippets = np.zeros((n_frames_per_rep, len(t_idxs)))
    snippets_times = np.zeros((n_frames_per_rep, len(t_idxs)))
    snippets_triggertimes = np.zeros((ntrigger_rep, len(t_idxs)))

    # Frames may be reused, this is not a standard reshaping
    for rep_idx, t_idx in enumerate(t_idxs):
        snippets[:, rep_idx] = trace[t_idx:t_idx + n_frames_per_rep]
        snippets_times[:, rep_idx] = times[t_idx:t_idx + n_frames_per_rep]
        snippets_triggertimes[:, rep_idx] = triggertimes[rep_idx * ntrigger_rep:(rep_idx + 1) * ntrigger_rep]

    return snippets, snippets_times, snippets_triggertimes, droppedlastrep_flag


def group_reps_to_list(data, truncate_last_rep=True):
    """Flatten a nested list of repetition arrays into a single list.

    Parameters
    ----------
    data : list of list of list of np.ndarray
        Nested structure ``data[rep][sub_rep]`` where each innermost
        element is a 2-D array whose columns are individual repetitions.
    truncate_last_rep : bool, optional
        If True, the last element of the flattened list is truncated to
        the length of the longest preceding element. Default is True.

    Returns
    -------
    rep_list : list of np.ndarray
        Flat list of 1-D arrays, one per column of each innermost array.
    """
    rep_list = [sub_sub_rep for rep in data for sub_rep in rep for sub_sub_rep in sub_rep.T]
    if len(rep_list) > 1 and truncate_last_rep:
        rep_lens = [len(rep) for rep in rep_list]
        max_rep_len = max(rep_lens[:-1])
        if rep_lens[-1] > max_rep_len:
            rep_list[-1] = rep_list[-1][:max_rep_len]
    return rep_list


def split_trace_by_group_reps(
        trace, times, triggertimes, trial_info, delay=0., atol=0.128, rtol_trunc=0.1,
        squeeze_single_reps=False, allow_incomplete=True, stack_kind=None, pad_trace=False):
    """Split a trace into per-group snippets allowing different stimulus groups within a repetition.

    Each entry in `trial_info` describes one stimulus group. Snippets are
    organised by group name and repetition index.  Optionally the per-rep
    lists are stacked into 2-D arrays.

    Parameters
    ----------
    trace : np.ndarray
        1-D recorded trace.
    times : np.ndarray
        1-D time stamps corresponding to `trace`.
    triggertimes : np.ndarray
        1-D array of all trigger onset times across all groups and
        repetitions.
    trial_info : list of dict
        Ordered list of dictionaries, one per stimulus group within a
        single repetition. Each dict must contain at least:

        - ``'name'`` (str): group identifier used as dictionary key.
        - ``'ntrigger'`` (int): total number of triggers for this group
          per repetition.

        An optional key ``'ntrigger_split'`` (int) can override the
        number of triggers used to determine snippet boundaries.
    delay : float, optional
        Delay in seconds added to trigger times. Default is 0.0.
    atol : float, optional
        Absolute tolerance (in seconds) for matching trigger times to
        trace time stamps. Default is 0.128.
    rtol_trunc : float, optional
        Relative tolerance passed to :func:`truncated_vstack` when
        ``stack_kind='truncate'``. Default is 0.1.
    squeeze_single_reps : bool, optional
        If True and there is only one repetition, the outer repetition
        list is removed. Default is False.
    allow_incomplete : bool, optional
        If True, incomplete repetitions at the end of the recording are
        silently skipped. If False an ``IndexError`` is raised.
        Default is True.
    stack_kind : {None, 'pad', 'truncate'}, optional
        How to stack per-repetition snippets. ``None`` returns raw
        nested lists. ``'pad'`` pads shorter snippets with ``NaN``.
        ``'truncate'`` truncates all snippets to the shortest one.
        Default is None.
    pad_trace : bool, optional
        Passed to :func:`compute_t_idxs`. Default is False.

    Returns
    -------
    group_snippets : dict of {str: list or np.ndarray}
        Snippet values keyed by group name.  When `stack_kind` is not
        None each value is a 2-D array of shape
        ``(n_frames_per_rep, n_reps)``.
    group_snippets_times : dict of {str: list or np.ndarray}
        Corresponding time stamps, same structure as `group_snippets`.
    group_snippets_triggertimes : dict of {str: list or np.ndarray}
        Corresponding trigger times, same structure as `group_snippets`.
    droppedlastrep_flag : bool
        True if the last repetition was dropped.

    Raises
    ------
    IndexError
        If `allow_incomplete` is False and data are incomplete.
    NotImplementedError
        If `stack_kind` is not one of the supported options.
    """

    ntrigger_rep = sum([trial_info_i['ntrigger'] for trial_info_i in trial_info])

    t_idxs, n_frames_per_rep, droppedlastrep_flag = compute_t_idxs(
        trace, times, triggertimes, ntrigger_rep, delay=delay, atol=atol, pad_trace=pad_trace,
        all_reps_same_length=False)

    n_reps = len(t_idxs)

    names = np.unique([trial_info_i['name'] for trial_info_i in trial_info])

    # Create data structure
    group_snippets = {name: [[] for _ in range(n_reps)] for name in names}
    group_snippets_times = {name: [[] for _ in range(n_reps)] for name in names}
    group_snippets_triggertimes = {name: [[] for _ in range(n_reps)] for name in names}

    for rep in range(n_reps):
        trg_count = rep * ntrigger_rep
        for trial_info_i in trial_info:
            ntrigger_i = trial_info_i['ntrigger']
            ntrigger_split_i = trial_info_i.get('ntrigger_split', ntrigger_i)

            if allow_incomplete and trg_count >= len(triggertimes):
                break

            try:
                idx_start = find_closest(
                    target=triggertimes[trg_count] + delay, data=times, atol=atol, as_index=True)
            except ValueError:
                if allow_incomplete:
                    warnings.warn('Data incomplete')
                    break
                else:
                    raise IndexError('Data incomplete')

            try:
                idx_end = find_closest(
                    target=triggertimes[trg_count + ntrigger_split_i], data=times, as_index=True)
            except IndexError:
                idx_end = -1

            snippets_i, snippets_t_i, snippets_tt_i, _ = split_trace_by_reps(
                trace[idx_start:idx_end], times[idx_start:idx_end],
                triggertimes[trg_count:trg_count + ntrigger_i], ntrigger_split_i,
                delay=delay, atol=atol, allow_drop_last=False)

            group_snippets[trial_info_i['name']][rep].append(snippets_i)
            group_snippets_times[trial_info_i['name']][rep].append(snippets_t_i)
            group_snippets_triggertimes[trial_info_i['name']][rep].append(snippets_tt_i)

            trg_count += ntrigger_i

    # Stack repetitions
    if stack_kind is not None:
        for name in names:
            snippets_i = group_reps_to_list(group_snippets[name], truncate_last_rep=True)
            snippets_t_i = group_reps_to_list(group_snippets_times[name], truncate_last_rep=True)
            snippets_tt_i = group_reps_to_list(group_snippets_triggertimes[name], truncate_last_rep=False)

            stacked_snippets_triggertimes = truncated_vstack(snippets_tt_i, rtol=0.).T

            if stack_kind == 'pad':
                stacked_snippets = padded_vstack(snippets_i, cval=np.nan).T
                stacked_snippets_times = padded_vstack(snippets_t_i, cval=np.nan).T
            elif stack_kind == 'truncate':
                stacked_snippets = truncated_vstack(snippets_i, rtol=rtol_trunc).T
                stacked_snippets_times = truncated_vstack(snippets_t_i, rtol=rtol_trunc).T
            else:
                raise NotImplementedError(f'Unknown stack_kind: {stack_kind}')

            group_snippets[name] = stacked_snippets
            group_snippets_times[name] = stacked_snippets_times
            group_snippets_triggertimes[name] = stacked_snippets_triggertimes

    elif n_reps == 1 and squeeze_single_reps:
        # Squeeze single reps
        for name in names:
            group_snippets[name] = group_snippets[name][0]
            group_snippets_times[name] = group_snippets_times[name][0]
            group_snippets_triggertimes[name] = group_snippets_triggertimes[name][0]

    return group_snippets, group_snippets_times, group_snippets_triggertimes, droppedlastrep_flag


def align_reps_with_trailing_nans(reps):
    """Crop trailing NaN-padded frames from a repetition array.

    When repetitions have different lengths they are stored in a
    common array padded with ``NaN`` at the end of shorter repetitions.
    This function removes the trailing rows that contain NaN values in
    any repetition, and also drops repetitions that consist almost
    entirely of NaNs (fewer than 2 finite values).

    Parameters
    ----------
    reps : np.ndarray
        2-D array of shape ``(n_frames, n_reps)`` potentially containing
        trailing ``NaN`` values.

    Returns
    -------
    np.ndarray
        Cropped array with trailing NaN rows removed and all-NaN
        repetition columns dropped.
    """
    if np.all(np.isfinite(reps)):  # Snippets may not have equal length and filled with NaNs
        return reps

    n_nan = np.array([np.argmax(np.isfinite(rep[::-1])) if np.any(np.isfinite(rep)) else len(rep)
                      for rep in reps.T])

    all_nan_reps = n_nan >= reps.shape[0] - 1
    if np.any(all_nan_reps):  # Remove reps if not at least 2 frames are present
        reps = reps[:, ~all_nan_reps]
        n_nan = n_nan[~all_nan_reps]

    reps = reps[:-n_nan.max(), :]
    return reps


def compute_repeat_correlation(reps):
    """Compute pairwise Pearson correlations between all repetitions.

    Trailing NaN-padded frames are removed via
    :func:`align_reps_with_trailing_nans` before computing correlations.

    Parameters
    ----------
    reps : np.ndarray
        2-D array of shape ``(n_frames, n_reps)`` containing the
        repetition traces, possibly with trailing ``NaN`` padding.

    Returns
    -------
    corrs : np.ndarray
        1-D array of upper-triangular pairwise correlation coefficients.
        Contains a single ``NaN`` element when `n_reps` is 1 or less.
    """
    reps = align_reps_with_trailing_nans(reps)
    n_reps = reps.shape[1]

    if n_reps <= 1:
        corrs = np.array([np.nan])
    else:
        corrs = np.corrcoef(reps, rowvar=False)[np.triu_indices(n_reps, 1)]
    return corrs
