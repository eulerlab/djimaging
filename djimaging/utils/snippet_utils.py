import numpy as np

from djimaging.utils.math_utils import truncated_vstack, padded_vstack
from djimaging.utils.trace_utils import find_closest, find_closest_before


def compute_t_idxs(trace, times, triggertimes, ntrigger_rep, delay=0., atol=0.128, allow_drop_last=True,
                   pad_trace=False):
    """

    Compute t_idxs, n_frames_per_rep, and dropped_last_rep_flag based on the given parameters.

    :param trace: ndarray, input trace data
    :param times: ndarray, time data corresponding to the trace
    :param triggertimes: ndarray, time data corresponding to the triggers
    :param ntrigger_rep: int, number of triggers per repetition
    :param delay: float, delay in seconds to apply to the trigger times
    :param atol: float, absolute tolerance used for finding closest index
    :param allow_drop_last: bool, flag to allow dropping last repetition if data is incomplete
    :param pad_trace: bool, if True crop trace such that the triggers fall within the trace
    :return: tuple (t_idxs, n_frames_per_rep, dropped_last_rep_flag)

    """
    if not pad_trace:
        t_idxs = [find_closest(target=tt + delay, data=times, atol=atol, as_index=True)
                  for tt in triggertimes[::ntrigger_rep]]
    else:
        t_idxs = [find_closest_before(target=tt + delay, data=times, atol=np.inf, as_index=True)
                  for tt in triggertimes[::ntrigger_rep]]

    if len(t_idxs) > 1:
        n_frames_per_rep = int(np.median(np.diff(t_idxs)))  # Use median rep length to ignore outliers
        if pad_trace:
            n_frames_per_rep += 1  # Heuristic to include frame from next repetition

        assert trace.shape == times.shape, 'Shapes do not match'

        if times[t_idxs[-1]:].size < n_frames_per_rep:
            assert allow_drop_last, 'Data incomplete, allow to drop last repetition or fix data'
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
    """Split trace in snippets, using triggertimes."""

    t_idxs, n_frames_per_rep, droppedlastrep_flag = compute_t_idxs(
        trace, times, triggertimes, ntrigger_rep,
        delay=delay, atol=atol, allow_drop_last=allow_drop_last, pad_trace=pad_trace)

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
    """
    Group Reps to List

    This method takes in a nested list of data and converts it into a flat list of repetitions.
    Each repetition is represented by a sub-sub-representation.

    Parameters:
        - data (list of list of ndarray): The nested list of data containing repetitions.
        - truncate_last_rep (bool): Specifies whether to truncate the last repetition to match
        the length of the longest repetition. Default is True.

    Returns:
        - rep_list (list): The flat list of sub-sub-representations.

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
        allow_drop_last=True, squeeze_single_reps=False, stack_kind=None):
    """Split trace in snippets, using triggertimes. allow to have different stimulus groups"""

    ntrigger_rep = sum([trial_info_i['ntrigger'] for trial_info_i in trial_info])

    t_idxs, n_frames_per_rep, droppedlastrep_flag = compute_t_idxs(
        trace, times, triggertimes, ntrigger_rep, delay=delay, atol=atol, allow_drop_last=allow_drop_last)

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

            idx_start = find_closest(
                target=triggertimes[trg_count] + delay, data=times, atol=atol, as_index=True)

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
