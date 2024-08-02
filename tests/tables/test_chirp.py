import numpy as np

from djimaging.tables.response.chirp.chirp_features_rgc import compute_on_off_index


def test_compute_on_off_index_on_response():
    t_start = 5.0
    n_reps = 3
    sf = 7.81
    stim_dur = sf * 4.2
    sin_f = 0.2

    frames_per_rep = int(np.floor(stim_dur * sf))

    # Create data
    trace_times = np.arange(frames_per_rep) * 1 / sf
    trace = np.sin(sin_f * (trace_times - 2) * np.pi * 2)

    # Make on response
    trace[(trace_times < 2) | (trace_times > 2 + 0.5 / sin_f)] = 0
    trace_times += t_start

    snippets_times = np.concatenate([trace_times + i * stim_dur for i in range(n_reps)]).reshape(n_reps, -1).T
    snippets = np.tile(trace, (n_reps, 1)).T

    trigger_times = np.concatenate([[t_start + i * stim_dur, t_start + i * stim_dur] for i in range(n_reps)])

    obs = compute_on_off_index(snippets, snippets_times, trigger_times, light_step_duration=1)
    exp_lb = 0.95
    assert obs > exp_lb


def test_compute_on_off_index_off_response():
    t_start = 17.4
    n_reps = 3
    sf = 7.81
    stim_dur = sf * 4.2
    sin_f = 0.2

    frames_per_rep = int(np.floor(stim_dur * sf))

    # Create data
    trace_times = np.arange(frames_per_rep) * 1 / sf

    response_at_t = 5
    trace = np.sin(sin_f * (trace_times - response_at_t) * np.pi * 2)
    trace[(trace_times < response_at_t) | (trace_times > response_at_t + 0.5 / sin_f)] = 0
    trace_times += t_start

    snippets_times = np.concatenate([trace_times + i * stim_dur for i in range(n_reps)]).reshape(n_reps, -1).T
    snippets = np.tile(trace, (n_reps, 1)).T

    trigger_times = np.concatenate([[t_start + i * stim_dur, t_start + i * stim_dur] for i in range(n_reps)])

    obs = compute_on_off_index(snippets, snippets_times, trigger_times, light_step_duration=1)
    exp_ub = -0.95
    assert obs < exp_ub


def test_compute_on_off_index_no_response():
    t_start = 0.0
    n_reps = 3
    sf = 7.81
    stim_dur = sf * 4.2

    frames_per_rep = int(np.floor(stim_dur * sf))

    # Create data
    trace_times = np.arange(frames_per_rep) * 1 / sf
    snippets_times = np.concatenate([trace_times + i * stim_dur for i in range(n_reps)]).reshape(n_reps, -1).T
    snippets = np.zeros(snippets_times.shape)

    trigger_times = np.concatenate([[t_start + i * stim_dur, t_start + i * stim_dur] for i in range(n_reps)])

    obs = compute_on_off_index(snippets, snippets_times, trigger_times, light_step_duration=1)
    exp = 0.0
    assert obs == exp
