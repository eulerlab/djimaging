"""Tests for djimaging.utils.snippet_utils.

Most tests use a ramp trace (``trace[i] == i``) sampled on a regular grid, so the
values in a snippet directly identify the trace indices it was cut from.
"""
import warnings

import numpy as np
import pytest

from djimaging.utils import snippet_utils


def _ramp(duration=6., dt=0.1, t0=0.):
    """Ramp trace with trace[i] == i on times t0 + i * dt (rounded to avoid float noise)."""
    n = int(round(duration / dt))
    times = np.round(t0 + np.arange(n) * dt, 9)
    trace = np.arange(n, dtype=float)
    return trace, times


# ---------------------------------------------------------------------------
# compute_t_idxs
# ---------------------------------------------------------------------------

def test_compute_t_idxs_regular_triggers():
    trace, times = _ramp(duration=6.)
    t_idxs, n_frames, dropped = snippet_utils.compute_t_idxs(trace, times, np.array([0., 2., 4.]), ntrigger_rep=1)
    assert list(t_idxs) == [0, 20, 40]
    assert n_frames == 20
    assert dropped is False


def test_compute_t_idxs_uses_every_ntrigger_rep_th_trigger():
    trace, times = _ramp(duration=6.)
    triggertimes = np.array([0., 1., 2., 3., 4., 5.])
    t_idxs, n_frames, dropped = snippet_utils.compute_t_idxs(trace, times, triggertimes, ntrigger_rep=2)
    assert list(t_idxs) == [0, 20, 40]
    assert n_frames == 20
    assert not dropped


def test_compute_t_idxs_positive_delay_shifts_start_and_drops_incomplete_last_rep():
    trace, times = _ramp(duration=6.)
    t_idxs, n_frames, dropped = snippet_utils.compute_t_idxs(
        trace, times, np.array([0., 2., 4.]), ntrigger_rep=1, delay=0.3)
    # Rep starts at 0.3, 2.3, 4.3 s -> idx 3, 23, 43; the last one has only 17 < 20 samples left
    assert list(t_idxs) == [3, 23]
    assert n_frames == 20
    assert dropped


def test_compute_t_idxs_negative_delay():
    trace, times = _ramp(duration=6.)
    t_idxs, n_frames, dropped = snippet_utils.compute_t_idxs(
        trace, times, np.array([1., 3., 5.]), ntrigger_rep=1, delay=-0.5)
    # Rep starts at 0.5, 2.5, 4.5 s -> idx 5, 25, 45; the last one has only 15 < 20 samples left
    assert list(t_idxs) == [5, 25]
    assert n_frames == 20
    assert dropped


def test_compute_t_idxs_disallow_drop_last_raises():
    trace, times = _ramp(duration=5.)
    with pytest.raises(ValueError, match='incomplete'):
        snippet_utils.compute_t_idxs(trace, times, np.array([0., 2., 4.]), ntrigger_rep=1, allow_drop_last=False)


def test_compute_t_idxs_last_rep_exactly_complete_is_kept():
    trace, times = _ramp(duration=6.)  # 60 samples, last rep idx 40..59 is complete
    t_idxs, _, dropped = snippet_utils.compute_t_idxs(
        trace, times, np.array([0., 2., 4.]), ntrigger_rep=1, allow_drop_last=False)
    assert list(t_idxs) == [0, 20, 40]
    assert not dropped


def test_compute_t_idxs_off_grid_trigger_within_atol_snaps_to_closest_sample():
    trace, times = _ramp(duration=6.)
    t_idxs, _, _ = snippet_utils.compute_t_idxs(trace, times, np.array([0.04, 2.06, 3.97]), ntrigger_rep=1)
    assert list(t_idxs) == [0, 21, 40]


def test_compute_t_idxs_trigger_outside_atol_raises():
    trace, times = _ramp(duration=6.)
    with pytest.raises(ValueError):
        snippet_utils.compute_t_idxs(trace, times, np.array([0., 2., 6.5]), ntrigger_rep=1, atol=0.128)


def test_compute_t_idxs_atol_is_absolute_in_seconds():
    trace, times = _ramp(duration=6., dt=1.)  # coarse sampling: 1 s
    with pytest.raises(ValueError):
        snippet_utils.compute_t_idxs(trace, times, np.array([0.4]), ntrigger_rep=1, atol=0.3)
    t_idxs, _, _ = snippet_utils.compute_t_idxs(trace, times, np.array([0.4]), ntrigger_rep=1, atol=0.5)
    assert list(t_idxs) == [0]


def test_compute_t_idxs_single_rep_uses_rest_of_trace():
    trace, times = _ramp(duration=6.)
    t_idxs, n_frames, dropped = snippet_utils.compute_t_idxs(trace, times, np.array([1., 2., 3.]), ntrigger_rep=3)
    assert list(t_idxs) == [10]
    assert n_frames == 50
    assert not dropped


def test_compute_t_idxs_not_same_length_returns_all_starts():
    trace, times = _ramp(duration=6.)
    t_idxs, n_frames, dropped = snippet_utils.compute_t_idxs(
        trace, times, np.array([0., 2., 4.5]), ntrigger_rep=1, all_reps_same_length=False)
    assert list(t_idxs) == [0, 20, 45]
    assert n_frames == 60
    assert not dropped


def test_compute_t_idxs_uneven_spacing_uses_median():
    trace, times = _ramp(duration=12.)
    # Rep lengths 20, 20, 20, 30 samples -> median 20, outlier ignored
    t_idxs, n_frames, dropped = snippet_utils.compute_t_idxs(
        trace, times, np.array([0., 2., 4., 6., 9.]), ntrigger_rep=1)
    assert n_frames == 20
    assert list(t_idxs) == [0, 20, 40, 60, 90]
    assert not dropped


def test_compute_t_idxs_pad_trace_uses_sample_before_trigger():
    trace, times = _ramp(duration=8.)
    # 0.07 is closest to sample 1 (0.1) but pad_trace picks the sample at/before the trigger
    t_idxs, n_frames, dropped = snippet_utils.compute_t_idxs(
        trace, times, np.array([0.07, 2.07, 4.07]), ntrigger_rep=1, pad_trace=True)
    assert list(t_idxs) == [0, 20, 40]
    assert n_frames == 21  # one extra frame from the next repetition
    assert not dropped


def test_compute_t_idxs_non_integer_sampling_rate():
    dt = 1. / 7.8125  # 0.128 s, typical 64x64 scan
    n = 100
    times = np.arange(n) * dt + 0.37
    trace = np.arange(n, dtype=float)
    triggertimes = times[[3, 33, 63]] + 0.01
    t_idxs, n_frames, dropped = snippet_utils.compute_t_idxs(trace, times, triggertimes, ntrigger_rep=1)
    assert list(t_idxs) == [3, 33, 63]
    assert n_frames == 30
    assert not dropped


# ---------------------------------------------------------------------------
# split_trace_by_reps
# ---------------------------------------------------------------------------

def test_split_trace_by_reps_values_identify_indices():
    trace, times = _ramp(duration=6.)
    triggertimes = np.array([0., 1., 2., 3., 4., 5.])
    snippets, snippets_times, snippets_tt, dropped = snippet_utils.split_trace_by_reps(
        trace, times, triggertimes, ntrigger_rep=2)
    assert snippets.shape == (20, 3)
    assert snippets_times.shape == (20, 3)
    assert snippets_tt.shape == (2, 3)
    for rep in range(3):
        np.testing.assert_array_equal(snippets[:, rep], np.arange(20 * rep, 20 * (rep + 1)))
        np.testing.assert_allclose(snippets_times[:, rep], times[20 * rep:20 * (rep + 1)])
        np.testing.assert_array_equal(snippets_tt[:, rep], triggertimes[2 * rep:2 * rep + 2])
    assert not dropped


def test_split_trace_by_reps_delay_shifts_window_not_triggertimes():
    trace, times = _ramp(duration=7.)
    triggertimes = np.array([0., 2., 4.])
    snippets, snippets_times, snippets_tt, dropped = snippet_utils.split_trace_by_reps(
        trace, times, triggertimes, ntrigger_rep=1, delay=0.5)
    np.testing.assert_array_equal(snippets[0, :], [5., 25., 45.])
    np.testing.assert_allclose(snippets_times[0, :], [0.5, 2.5, 4.5])
    np.testing.assert_array_equal(snippets_tt[0, :], triggertimes)
    assert not dropped


def test_split_trace_by_reps_drops_incomplete_last_rep():
    trace, times = _ramp(duration=5.)
    snippets, _, snippets_tt, dropped = snippet_utils.split_trace_by_reps(
        trace, times, np.array([0., 2., 4.]), ntrigger_rep=1)
    assert snippets.shape == (20, 2)
    np.testing.assert_array_equal(snippets_tt, [[0., 2.]])
    assert dropped


def test_split_trace_by_reps_pad_trace_reuses_first_frame_of_next_rep():
    trace, times = _ramp(duration=8.)
    snippets, _, _, _ = snippet_utils.split_trace_by_reps(
        trace, times, np.array([0., 2., 4.]), ntrigger_rep=1, pad_trace=True)
    assert snippets.shape == (21, 3)
    assert snippets[-1, 0] == snippets[0, 1] == 20.


def test_split_trace_by_reps_single_rep():
    trace, times = _ramp(duration=3.)
    snippets, snippets_times, snippets_tt, dropped = snippet_utils.split_trace_by_reps(
        trace, times, np.array([1.]), ntrigger_rep=1)
    assert snippets.shape == (20, 1)
    np.testing.assert_array_equal(snippets[:, 0], np.arange(10, 30))
    assert not dropped


def test_split_trace_by_reps_int_trace_returns_float():
    _, times = _ramp(duration=4.)
    trace = np.arange(times.size)  # int
    snippets, _, _, _ = snippet_utils.split_trace_by_reps(trace, times, np.array([0., 2.]), ntrigger_rep=1)
    assert snippets.dtype == np.float64
    np.testing.assert_array_equal(snippets[:, 1], np.arange(20, 40))


def test_split_trace_by_reps_no_triggers_raises():
    trace, times = _ramp()
    with pytest.raises(ValueError, match='No trigger'):
        snippet_utils.split_trace_by_reps(trace, times, np.array([]), ntrigger_rep=1)


def test_split_trace_by_reps_zero_length_reps_raises():
    trace, times = _ramp()
    with pytest.raises(ValueError, match='n_frames_per_rep'):
        snippet_utils.split_trace_by_reps(trace, times, np.array([1., 1., 1.]), ntrigger_rep=1)


def test_split_trace_by_reps_last_rep_with_missing_triggers_is_dropped():
    # Stimulus cancelled after 5 of 6 triggers while recording continued
    trace, times = _ramp(duration=7.)
    triggertimes = np.array([0., 1., 2., 3., 4.])
    snippets, _, snippets_tt, dropped = snippet_utils.split_trace_by_reps(
        trace, times, triggertimes, ntrigger_rep=2)
    assert snippets.shape == (20, 2)
    np.testing.assert_array_equal(snippets_tt, [[0., 2.], [1., 3.]])
    assert dropped


# ---------------------------------------------------------------------------
# group_reps_to_list
# ---------------------------------------------------------------------------

def test_group_reps_to_list_flattens_columns_in_order():
    data = [
        [np.array([[1., 2.], [3., 4.]])],  # rep 0: one sub-rep with two columns
        [np.array([[5.], [6.]]), np.array([[7.], [8.]])],  # rep 1: two sub-reps
    ]
    out = snippet_utils.group_reps_to_list(data)
    assert len(out) == 4
    for obs, exp in zip(out, [[1., 3.], [2., 4.], [5., 6.], [7., 8.]]):
        np.testing.assert_array_equal(obs, exp)


def test_group_reps_to_list_truncates_only_longer_last_rep():
    data = [[np.arange(3.)[:, None]], [np.arange(5.)[:, None]]]
    out = snippet_utils.group_reps_to_list(data, truncate_last_rep=True)
    np.testing.assert_array_equal(out[-1], [0., 1., 2.])

    out = snippet_utils.group_reps_to_list(data, truncate_last_rep=False)
    assert len(out[-1]) == 5

    data_short_last = [[np.arange(5.)[:, None]], [np.arange(3.)[:, None]]]
    out = snippet_utils.group_reps_to_list(data_short_last, truncate_last_rep=True)
    assert [len(o) for o in out] == [5, 3]


def test_group_reps_to_list_empty_and_single():
    assert snippet_utils.group_reps_to_list([[], []]) == []
    out = snippet_utils.group_reps_to_list([[np.arange(4.)[:, None]]])
    assert len(out) == 1 and len(out[0]) == 4


# ---------------------------------------------------------------------------
# split_trace_by_group_reps
# ---------------------------------------------------------------------------

_TRIAL_INFO_AB = [dict(name='b', ntrigger=1), dict(name='a', ntrigger=1)]


def test_split_trace_by_group_reps_nested_lists():
    trace, times = _ramp(duration=10.)
    # Two reps of [b, a]; each group lasts 2 s; trace continues after the last group
    triggertimes = np.array([0., 2., 4., 6.])
    snips, snips_t, snips_tt, dropped = snippet_utils.split_trace_by_group_reps(
        trace, times, triggertimes, _TRIAL_INFO_AB)
    assert sorted(snips.keys()) == ['a', 'b']
    assert len(snips['a']) == len(snips['b']) == 2
    np.testing.assert_array_equal(snips['b'][0][0][:, 0], np.arange(0, 20))
    np.testing.assert_array_equal(snips['a'][0][0][:, 0], np.arange(20, 40))
    np.testing.assert_array_equal(snips['b'][1][0][:, 0], np.arange(40, 60))
    # Last group of last rep runs until the end of the trace
    assert snips['a'][1][0][0, 0] == 60.
    np.testing.assert_allclose(snips_t['a'][0][0][:, 0], times[20:40])
    np.testing.assert_array_equal(snips_tt['a'][1][0], [[6.]])
    assert dropped is False


def test_split_trace_by_group_reps_stack_pad_and_truncate():
    trace, times = _ramp(duration=10.)
    triggertimes = np.array([0., 2., 4., 6.])
    for stack_kind in ['pad', 'truncate']:
        snips, snips_t, snips_tt, _ = snippet_utils.split_trace_by_group_reps(
            trace, times, triggertimes, _TRIAL_INFO_AB, stack_kind=stack_kind)
        assert snips['a'].shape == (20, 2)
        assert snips_t['a'].shape == (20, 2)
        np.testing.assert_array_equal(snips['a'][0], [20., 60.])
        np.testing.assert_array_equal(snips['b'][:, 1], np.arange(40, 60))
        np.testing.assert_array_equal(snips_tt['a'], [[2., 6.]])
        np.testing.assert_array_equal(snips_tt['b'], [[0., 4.]])
        assert np.all(np.isfinite(snips['a']))


def test_split_trace_by_group_reps_unknown_stack_kind_raises():
    trace, times = _ramp(duration=10.)
    with pytest.raises(NotImplementedError):
        snippet_utils.split_trace_by_group_reps(
            trace, times, np.array([0., 2., 4., 6.]), _TRIAL_INFO_AB, stack_kind='nope')


def test_split_trace_by_group_reps_squeeze_single_rep():
    trace, times = _ramp(duration=6.)
    snips, snips_t, snips_tt, _ = snippet_utils.split_trace_by_group_reps(
        trace, times, np.array([0., 2.]), _TRIAL_INFO_AB, squeeze_single_reps=True)
    assert isinstance(snips['b'], list) and len(snips['b']) == 1
    np.testing.assert_array_equal(snips['b'][0][:, 0], np.arange(0, 20))
    np.testing.assert_array_equal(snips_tt['a'][0], [[2.]])


def test_split_trace_by_group_reps_incomplete_last_rep_allowed():
    trace, times = _ramp(duration=8.)
    # Second rep only has the trigger for 'b'
    snips, _, _, _ = snippet_utils.split_trace_by_group_reps(
        trace, times, np.array([0., 2., 4.]), _TRIAL_INFO_AB)
    assert [len(r) for r in snips['b']] == [1, 1]
    assert [len(r) for r in snips['a']] == [1, 0]

    snips, _, _, _ = snippet_utils.split_trace_by_group_reps(
        trace, times, np.array([0., 2., 4.]), _TRIAL_INFO_AB, stack_kind='pad')
    assert snips['b'].shape[1] == 2
    assert snips['a'].shape == (20, 1)


def test_split_trace_by_group_reps_incomplete_last_rep_not_allowed_raises():
    trace, times = _ramp(duration=8.)
    with pytest.raises(IndexError):
        snippet_utils.split_trace_by_group_reps(
            trace, times, np.array([0., 2., 4.]), _TRIAL_INFO_AB, allow_incomplete=False)


def test_split_trace_by_group_reps_int_names():
    trace, times = _ramp(duration=10.)
    trial_info = [dict(name=d, ntrigger=1) for d in (180, 0)]
    snips, _, _, _ = snippet_utils.split_trace_by_group_reps(
        trace, times, np.array([0., 2., 4., 6.]), trial_info, stack_kind='pad')
    assert sorted(int(k) for k in snips.keys()) == [0, 180]
    np.testing.assert_array_equal(snips[180][0], [0., 40.])


def test_split_trace_by_group_reps_final_group_includes_last_sample():
    trace, times = _ramp(duration=8.)  # 80 samples, last group [60, 80) ends exactly with the trace
    snips, _, _, _ = snippet_utils.split_trace_by_group_reps(
        trace, times, np.array([0., 2., 4., 6.]), _TRIAL_INFO_AB, stack_kind='pad')
    np.testing.assert_array_equal(snips['a'][:, 1], np.arange(60, 80))


def test_split_trace_by_group_reps_delay_shifts_window_without_shortening():
    trace, times = _ramp(duration=10.)
    snips, _, _, _ = snippet_utils.split_trace_by_group_reps(
        trace, times, np.array([0., 2., 4., 6.]), _TRIAL_INFO_AB, delay=0.5)
    np.testing.assert_array_equal(snips['b'][0][0][:, 0], np.arange(5, 25))
    np.testing.assert_array_equal(snips['a'][0][0][:, 0], np.arange(25, 45))
    np.testing.assert_array_equal(snips['b'][1][0][:, 0], np.arange(45, 65))


@pytest.mark.xfail(strict=True, reason="BUG: group end index uses ntrigger_split instead of ntrigger, "
                                       "so ntrigger_split < ntrigger crashes")
def test_split_trace_by_group_reps_ntrigger_split():
    trace, times = _ramp(duration=10.)
    # Per rep: group 'a' with 2 triggers split into 2 snippets of 1 trigger, then group 'b' with 2 triggers
    trial_info = [dict(name='a', ntrigger=2, ntrigger_split=1), dict(name='b', ntrigger=2)]
    triggertimes = np.arange(8.)
    snips, _, snips_tt, _ = snippet_utils.split_trace_by_group_reps(trace, times, triggertimes, trial_info)
    assert snips['a'][0][0].shape == (10, 2)
    np.testing.assert_array_equal(snips['a'][0][0][0], [0., 10.])
    np.testing.assert_array_equal(snips['a'][1][0][0], [40., 50.])
    np.testing.assert_array_equal(snips_tt['a'][0][0], [[0., 1.]])
    np.testing.assert_array_equal(snips['b'][0][0][:, 0], np.arange(20, 40))


# ---------------------------------------------------------------------------
# align_reps_with_trailing_nans / compute_repeat_correlation
# ---------------------------------------------------------------------------

def test_align_reps_all_finite_returned_unchanged():
    reps = np.arange(12.).reshape(4, 3)
    assert snippet_utils.align_reps_with_trailing_nans(reps) is reps


def test_align_reps_crops_to_shortest_rep():
    reps = np.array([[1., 1.], [2., 2.], [3., np.nan], [4., np.nan], [5., np.nan]])
    out = snippet_utils.align_reps_with_trailing_nans(reps)
    np.testing.assert_array_equal(out, [[1., 1.], [2., 2.]])


def test_align_reps_drops_reps_with_less_than_two_frames():
    reps = np.array([[1., 1., 9.], [2., 2., np.nan], [3., np.nan, np.nan], [4., np.nan, np.nan]])
    out = snippet_utils.align_reps_with_trailing_nans(reps)
    np.testing.assert_array_equal(out, [[1., 1.], [2., 2.]])


@pytest.mark.xfail(strict=True, reason="BUG: reps[:-n_nan.max()] with n_nan.max()==0 returns an empty array")
@pytest.mark.parametrize('reps', [
    # one all-NaN rep, others complete
    np.array([[1., 2., np.nan], [2., 3., np.nan], [3., 5., np.nan], [4., 4., np.nan]]),
    # NaN in the middle of a rep, but no trailing NaNs
    np.array([[1., 2.], [np.nan, 3.], [3., 5.], [4., 4.]]),
    # one rep with a single finite frame (dropped), others complete
    np.array([[0., 0., 7.], [1., -1., np.nan], [3., -3., np.nan], [2., -2., np.nan]]),
], ids=['all_nan_rep', 'non_trailing_nan', 'single_frame_rep'])
def test_align_reps_without_trailing_nans_keeps_all_frames(reps):
    out = snippet_utils.align_reps_with_trailing_nans(reps)
    assert out.shape[0] == reps.shape[0]


@pytest.mark.xfail(strict=True, reason="BUG: all-NaN reps crash with max() of empty array instead of NaN result")
def test_compute_repeat_correlation_all_nan_returns_nan():
    corrs = snippet_utils.compute_repeat_correlation(np.full((5, 3), np.nan))
    assert corrs.shape == (1,) and np.isnan(corrs[0])


def test_compute_repeat_correlation_pair_order():
    x = np.array([0., 1., 3., 2., 5.])
    reps = np.stack([x, 2 * x + 1, -x], axis=1)
    corrs = snippet_utils.compute_repeat_correlation(reps)
    # upper-triangular order: (0,1), (0,2), (1,2)
    np.testing.assert_allclose(corrs, [1., -1., -1.])


def test_compute_repeat_correlation_single_rep_is_nan():
    corrs = snippet_utils.compute_repeat_correlation(np.arange(5.)[:, None])
    assert corrs.shape == (1,) and np.isnan(corrs[0])


def test_compute_repeat_correlation_with_trailing_nans():
    reps = np.array([[0., 0.], [1., 1.], [3., 3.], [2., np.nan]])
    corrs = snippet_utils.compute_repeat_correlation(reps)
    np.testing.assert_allclose(corrs, [1.])


def test_compute_repeat_correlation_drops_almost_empty_rep():
    reps = np.array([[0., 0., 7.], [1., -1., np.nan], [3., -3., np.nan], [2., np.nan, np.nan]])
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        corrs = snippet_utils.compute_repeat_correlation(reps)
    np.testing.assert_allclose(corrs, [-1.])


def test_split_trace_by_group_reps_group_trigger_after_trace_end():
    trace, times = _ramp(duration=8.)
    triggertimes = np.array([0., 2., 4., 9.])  # 'a' of rep 2 starts after the recording ended
    with pytest.warns(UserWarning, match='incomplete'):
        snips, _, _, _ = snippet_utils.split_trace_by_group_reps(trace, times, triggertimes, _TRIAL_INFO_AB)
    assert [len(r) for r in snips['b']] == [1, 1]
    assert [len(r) for r in snips['a']] == [1, 0]

    with pytest.raises(IndexError):
        snippet_utils.split_trace_by_group_reps(trace, times, triggertimes, _TRIAL_INFO_AB, allow_incomplete=False)
