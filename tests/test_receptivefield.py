import numpy as np

from djimaging.tables.optional.receptivefield import get_sets, resample_trace, compute_receptive_field


def generate_data_3d(seed, trace_dt=0.1, trace_trng=(0.3, 64), dims_xy=(6, 8), stim_dt=0.33, stim_trng=(1, 55)):
    np.random.seed(seed)

    tracetime = np.arange(trace_trng[0], trace_trng[1] + trace_dt / 2, trace_dt)
    stimtime = np.arange(stim_trng[0], stim_trng[1] + stim_dt / 2, stim_dt)

    stim = np.random.choice([0, 1], (stimtime.size,) + dims_xy)

    cx = dims_xy[0] // 2
    cy = dims_xy[1] // 2

    stim_input = np.mean(stim[:, cx - 1:cx + 2, cy - 1:cy + 2], axis=(1, 2))

    trace = np.interp(x=tracetime, xp=stimtime + 2 * trace_dt, fp=stim_input.astype(float), left=0, right=0)
    trace += np.random.normal(0, np.max(np.abs(trace)) / 10, trace.size)

    return stim, stimtime, trace, tracetime, trace_dt


def test_split_train():
    stim, stimtime, trace, tracetime, trace_dt = generate_data_3d(seed=42)
    X_dict, y_dict, dt = get_sets(stim=stim, stimtime=stimtime, trace=trace, tracetime=tracetime,
                                  frac_train=1., frac_dev=0., fupsample=1, gradient=False)
    assert set(X_dict.keys()) == {'train'}


def test_split_train_dev():
    stim, stimtime, trace, tracetime, trace_dt = generate_data_3d(seed=42)
    X_dict, y_dict, dt = get_sets(stim=stim, stimtime=stimtime, trace=trace, tracetime=tracetime,
                                  frac_train=0.8, frac_dev=0.2, fupsample=2, gradient=False)

    assert set(X_dict.keys()) == {'train', 'dev'}
    assert np.isclose(y_dict['dev'].size / y_dict['train'].size, 0.2 / 0.8, atol=0.01, rtol=0.01)


def test_split_train_test():
    stim, stimtime, trace, tracetime, trace_dt = generate_data_3d(seed=461)
    X_dict, y_dict, dt = get_sets(stim=stim, stimtime=stimtime, trace=trace, tracetime=tracetime,
                                  frac_train=0.8, frac_dev=0.0, fupsample=2, gradient=False)

    assert set(X_dict.keys()) == {'train', 'test'}
    assert np.isclose(y_dict['test'].size / y_dict['train'].size, 0.2 / 0.8, atol=0.01, rtol=0.01)


def test_split_train_dev_test():
    stim, stimtime, trace, tracetime, trace_dt = generate_data_3d(seed=50)
    X_dict, y_dict, dt = get_sets(stim=stim, stimtime=stimtime, trace=trace, tracetime=tracetime,
                                  frac_train=0.6, frac_dev=0.2, fupsample=1, gradient=True)

    assert set(X_dict.keys()) == {'train', 'dev', 'test'}
    assert np.isclose(y_dict['dev'].size / y_dict['train'].size, 0.2 / 0.6, atol=0.01, rtol=0.01)
    assert np.isclose(y_dict['test'].size / y_dict['train'].size, 0.2 / 0.6, atol=0.01, rtol=0.01)
    assert np.isclose(y_dict['dev'].size / y_dict['test'].size, 1., atol=0.01, rtol=0.01)


def test_resample_trace():
    dt_old = 0.1
    dt_new = 0.01
    tracetime = np.arange(10) * dt_old
    trace = np.random.normal(0, 1, tracetime.size)
    tracetime_resampled, trace_resampled = resample_trace(tracetime, trace, dt_new)

    dts_new = np.diff(tracetime_resampled)

    assert np.isclose(np.mean(dts_new), dt_new)
    assert np.isclose(np.std(dts_new), 0.0)
    assert np.isclose(tracetime_resampled[0], tracetime[0])


def test_sta():
    stim, stimtime, trace, tracetime, trace_dt = generate_data_3d(seed=4897)
    rf, rf_pred, X, y, dt = compute_receptive_field(
        trace=trace, tracetime=tracetime, stim=stim, stimtime=stimtime,
        frac_train=0.6, frac_dev=0.0, dur_filter_s=1.0,
        kind='sta', fupsample=1, gradient=False)

    assert rf_pred['y_pred_train'].size + rf_pred['burn_in'] == y['train'].size
    assert rf_pred['y_pred_test'].size + rf_pred['burn_in'] == y['test'].size
    assert 0.5 < rf_pred['cc_train'] < 1.0
    assert 0.1 < rf_pred['cc_test'] < 1.0


def test_sta_overfitting():
    stim, stimtime, trace, tracetime, trace_dt = generate_data_3d(seed=4897)
    rf, rf_pred, X, y, dt = compute_receptive_field(
        trace=trace, tracetime=tracetime, stim=stim, stimtime=stimtime,
        frac_train=0.6, frac_dev=0.0, dur_filter_s=3.0,
        kind='sta', fupsample=1, gradient=False)

    assert rf_pred['y_pred_train'].size + rf_pred['burn_in'] == y['train'].size
    assert rf_pred['y_pred_test'].size + rf_pred['burn_in'] == y['test'].size
    assert 0.5 < rf_pred['cc_train']
    assert rf_pred['cc_test'] + 0.1 < rf_pred['cc_train']


def test_mle():
    stim, stimtime, trace, tracetime, trace_dt = generate_data_3d(seed=1324)
    rf, rf_pred, X, y, dt = compute_receptive_field(
        trace=trace, tracetime=tracetime, stim=stim, stimtime=stimtime,
        frac_train=0.6, frac_dev=0.0, dur_filter_s=0.2,
        kind='mle', fupsample=1, gradient=True)

    assert rf_pred['y_pred_train'].size + rf_pred['burn_in'] == y['train'].size
    assert rf_pred['y_pred_test'].size + rf_pred['burn_in'] == y['test'].size
    assert 0.1 < rf_pred['cc_train'] < 0.8
    assert 0.1 < rf_pred['cc_test'] < 0.8


def test_mle_overfitting():
    stim, stimtime, trace, tracetime, trace_dt = generate_data_3d(seed=1324)
    rf, rf_pred, X, y, dt = compute_receptive_field(
        trace=trace, tracetime=tracetime, stim=stim, stimtime=stimtime,
        frac_train=0.6, frac_dev=0.0, dur_filter_s=2.0,
        kind='mle', fupsample=1, gradient=True)

    assert rf_pred['y_pred_train'].size + rf_pred['burn_in'] == y['train'].size
    assert rf_pred['y_pred_test'].size + rf_pred['burn_in'] == y['test'].size
    assert 0.99 < rf_pred['cc_train']
    assert 0.1 < rf_pred['cc_test'] < 1.0