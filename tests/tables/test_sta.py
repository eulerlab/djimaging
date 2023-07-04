import numpy as np
import pytest

from djimaging.tables.receptivefield import rf_utils
from djimaging.tables.receptivefield.rf_utils import prepare_noise_data, split_data

try:
    import rfest
except ImportError:
    rfest = None


def compute_norm_mse(w1, w2):
    return np.mean((rfest.utils.uvec(w1.flat) - rfest.utils.uvec(w2.flat)) ** 2)


def compare_fits(w_true, w_fit):
    random_rfs = [np.random.normal(np.mean(rfest.utils.uvec(w_true)), np.std(rfest.utils.uvec(w_true)), w_true.shape)
                  for _ in range(100)]
    random_mses = [compute_norm_mse(w1=random_rf, w2=w_true) for random_rf in random_rfs]
    fit_mse = compute_norm_mse(w1=w_fit, w2=w_true)
    return fit_mse, random_mses


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
    stim, trace, dt, t0, dt_rel_error = prepare_noise_data(
        trace=trace, tracetime=tracetime, stim=stim, triggertimes=stimtime,
        fupsample_trace=2, fit_kind='trace', lowpass_cutoff=0)

    x_dict, y_dict = split_data(x=stim, y=trace, frac_train=1., frac_dev=0., as_dict=True)
    assert set(x_dict.keys()) == {'train'}


def test_split_train_dev():
    stim, stimtime, trace, tracetime, trace_dt = generate_data_3d(seed=42)

    stim, trace, dt, t0, dt_rel_error = prepare_noise_data(
        trace=trace, tracetime=tracetime, stim=stim, triggertimes=stimtime,
        fupsample_trace=2, fit_kind='trace', lowpass_cutoff=0)

    x_dict, y_dict = split_data(x=stim, y=trace, frac_train=0.8, frac_dev=0.2, as_dict=True)

    assert set(x_dict.keys()) == {'train', 'dev'}
    assert np.isclose(y_dict['dev'].size / y_dict['train'].size, 0.2 / 0.8, atol=0.01, rtol=0.01)


def test_split_train_test():
    stim, stimtime, trace, tracetime, trace_dt = generate_data_3d(seed=461)

    stim, trace, dt, t0, dt_rel_error = prepare_noise_data(
        trace=trace, tracetime=tracetime, stim=stim, triggertimes=stimtime,
        fupsample_trace=2, fit_kind='trace', lowpass_cutoff=0)

    x_dict, y_dict = split_data(x=stim, y=trace, frac_train=0.8, frac_dev=0.0, as_dict=True)

    assert set(x_dict.keys()) == {'train', 'test'}
    assert np.isclose(y_dict['test'].size / y_dict['train'].size, 0.2 / 0.8, atol=0.01, rtol=0.01)


def test_split_train_dev_test():
    stim, stimtime, trace, tracetime, trace_dt = generate_data_3d(seed=50)

    stim, trace, dt, t0, dt_rel_error = prepare_noise_data(
        trace=trace, tracetime=tracetime, stim=stim, triggertimes=stimtime,
        fupsample_trace=1, fit_kind='trace', lowpass_cutoff=0)

    x_dict, y_dict = split_data(x=stim, y=trace, frac_train=0.6, frac_dev=0.2, as_dict=True)

    assert set(x_dict.keys()) == {'train', 'dev', 'test'}
    assert np.isclose(y_dict['dev'].size / y_dict['train'].size, 0.2 / 0.6, atol=0.01, rtol=0.01)
    assert np.isclose(y_dict['test'].size / y_dict['train'].size, 0.2 / 0.6, atol=0.01, rtol=0.01)
    assert np.isclose(y_dict['dev'].size / y_dict['test'].size, 1., atol=0.01, rtol=0.01)


@pytest.mark.skipif(rfest is None, reason="requires rfest")
def test_sta_fit():
    np.random.seed(13488)

    w_true, X, y, dt, dims = rfest.simulate.generate_data_3d_stim(
        stim_noise='white', rf_kind='gauss', response_noise='none', design_matrix=True,
        n_stim_frames=500, n_reps_per_frame=3, shift=0)

    w_fit = rf_utils.compute_rf_sta(X=X, y=y)
    fit_mse, random_mses = compare_fits(w_true=w_true, w_fit=w_fit)
    assert fit_mse < np.mean(random_mses) - 2 * np.std(random_mses)


@pytest.mark.skipif(rfest is None, reason="requires rfest")
def test_sta_fit_single_batch():
    np.random.seed(13488)

    w_true, X, y, dt, dims = rfest.simulate.generate_data_3d_stim(
        stim_noise='white', rf_kind='gauss', response_noise='none', design_matrix=False,
        n_stim_frames=500, n_reps_per_frame=3, shift=0)

    assert w_true.shape[1:] == X.shape[1:]

    w_fit, _ = rf_utils._compute_linear_rf_single_batch(
        x_train=X, y_train=y, kind='sta', dim_t=w_true.shape[0], shift=0, burn_in=w_true.shape[0] - 1,
        threshold_pred=False, is_spikes=False, dtype=y.dtype)

    w_fit = w_fit.reshape(w_true.shape)

    fit_mse, random_mses = compare_fits(w_true=w_true, w_fit=w_fit)
    assert fit_mse < np.mean(random_mses) - 2 * np.std(random_mses)


@pytest.mark.skipif(rfest is None, reason="requires rfest")
def test_sta_fit_batchwise():
    np.random.seed(13488)

    w_true, X, y, dt, dims = rfest.simulate.generate_data_3d_stim(
        stim_noise='white', rf_kind='gauss', response_noise='none', design_matrix=False,
        n_stim_frames=500, n_reps_per_frame=3, shift=0)

    assert w_true.shape[1:] == X.shape[1:]

    w_fit, _ = rf_utils._compute_sta_batchwise(
        x_train=X, y_train=y, kind='sta', dim_t=w_true.shape[0], shift=0, burn_in=w_true.shape[0] - 1,
        threshold_pred=False, is_spikes=False, dtype=y.dtype, batch_size=3)

    fit_mse, random_mses = compare_fits(w_true=w_true, w_fit=w_fit)
    assert fit_mse < np.mean(random_mses) - 2 * np.std(random_mses)


@pytest.mark.skipif(rfest is None, reason="requires rfest")
def test_sta_fit_single_batch_vs_batchwise():
    np.random.seed(13488)

    w_true, X, y, dt, dims = rfest.simulate.generate_data_3d_stim(
        stim_noise='white', rf_kind='gauss', response_noise='none', design_matrix=False,
        n_stim_frames=500, n_reps_per_frame=3, shift=0)

    assert w_true.shape[1:] == X.shape[1:]

    w_fit, y_pred = rf_utils._compute_linear_rf_single_batch(
        x_train=X, y_train=y, kind='sta', dim_t=w_true.shape[0], shift=0, burn_in=w_true.shape[0] - 1,
        threshold_pred=False, is_spikes=False, dtype=y.dtype)

    w_fit = w_fit.reshape(w_true.shape)

    for batch_size in [1, 7, 400, 1000]:
        w_fit2, y_pred2 = rf_utils._compute_sta_batchwise(
            x_train=X, y_train=y, kind='sta', dim_t=w_true.shape[0], shift=0, burn_in=w_true.shape[0] - 1,
            threshold_pred=False, is_spikes=False, dtype=y.dtype, batch_size=batch_size)

        w_fit2 = w_fit.reshape(w_true.shape)
        assert np.allclose(w_fit, w_fit2)
        assert np.allclose(y_pred, y_pred2)


@pytest.mark.skipif(rfest is None, reason="requires rfest")
def test_sta_fit_few_datapoints():
    np.random.seed(364)

    w_true, X, y, dt, dims = rfest.simulate.generate_data_3d_stim(
        stim_noise='white', rf_kind='gauss', response_noise='gaussian', design_matrix=True,
        n_stim_frames=13, n_reps_per_frame=3, shift=0)

    w_fit = rf_utils.compute_rf_sta(X=X, y=y)
    fit_mse, random_mses = compare_fits(w_true=w_true, w_fit=w_fit)
    assert fit_mse < np.mean(random_mses)


@pytest.mark.skipif(rfest is None, reason="requires rfest")
def test_sta_fit_shift():
    np.random.seed(434)

    w_true, X, y, dt, dims = rfest.simulate.generate_data_3d_stim(
        stim_noise='white', rf_kind='gauss', response_noise='gaussian', design_matrix=True,
        n_stim_frames=1000, n_reps_per_frame=1, shift=5)

    w_fit = rf_utils.compute_rf_sta(X=X, y=y)
    fit_mse, random_mses = compare_fits(w_true=w_true, w_fit=w_fit)
    assert fit_mse < np.mean(random_mses)


@pytest.mark.skipif(rfest is None, reason="requires rfest")
def test_sta_response():
    np.random.seed(12234)

    w_true, X, y, dt, dims = rfest.simulate.generate_data_3d_stim(
        stim_noise='white', rf_kind='gauss', response_noise='gaussian', design_matrix=True,
        n_stim_frames=1000, n_reps_per_frame=1, shift=5)
    w_fit = rf_utils.compute_rf_sta(X=X, y=y)
    y_pred = rf_utils.predict_linear_rf_response(w_fit.flat, X, threshold=False, dtype=np.float32)
    assert np.corrcoef(y, y_pred)[0, 1] > 0.5


@pytest.mark.skipif(rfest is None, reason="requires rfest")
def test_sta_response_splits():
    np.random.seed(42)

    w_true, X, y, dt, dims = rfest.simulate.generate_data_3d_stim(
        stim_noise='white', rf_kind='gauss', response_noise='gaussian', design_matrix=True,
        n_stim_frames=3000, n_reps_per_frame=1, shift=5)

    (x_trn, y_trn), (_, _), (x_tst, y_tst) = split_data(X, y, frac_train=0.8, frac_dev=0.)

    w_fit = rf_utils.compute_rf_sta(X=x_trn, y=y_trn)

    y_pred_train = rf_utils.predict_linear_rf_response(w_fit.flat, x_trn, threshold=False, dtype=np.float32)
    y_pred_test = rf_utils.predict_linear_rf_response(w_fit.flat, x_tst, threshold=False, dtype=np.float32)

    assert np.corrcoef(y_trn, y_pred_train)[0, 1] > 0.5
    assert np.corrcoef(y_tst, y_pred_test)[0, 1] > 0.25


@pytest.mark.skipif(rfest is None, reason="requires rfest")
def test_mle_fit():
    np.random.seed(366)

    w_true, X, y, dt, dims = rfest.simulate.generate_data_3d_stim(
        stim_noise='pink', rf_kind='gauss', response_noise='none', design_matrix=True,
        n_stim_frames=500, n_reps_per_frame=3, shift=3)

    w_fit = rf_utils.compute_rf_mle(X=X, y=y)
    fit_mse, random_mses = compare_fits(w_true=w_true, w_fit=w_fit)
    assert fit_mse < np.mean(random_mses) - 2 * np.std(random_mses)
