import numpy as np
import pytest

from djimaging.utils.receptive_fields import fit_rf_utils
from djimaging.utils.receptive_fields.preprocess_rf_utils import prepare_noise_data
from tests.tables.sim_data import generate_data_3d_stim, uvec

try:
    import rfest
except ImportError:
    rfest = None


def compute_norm_mse(w1, w2):
    return np.mean((uvec(w1.flat) - uvec(w2.flat)) ** 2)


def compare_fits(w_true, w_fit):
    random_rfs = [np.random.normal(np.mean(uvec(w_true)), np.std(uvec(w_true)), w_true.shape)
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

    x_dict, y_dict = fit_rf_utils.split_data(x=stim, y=trace, frac_train=1., frac_dev=0., as_dict=True)
    assert set(x_dict.keys()) == {'train'}


def test_split_train_dev():
    stim, stimtime, trace, tracetime, trace_dt = generate_data_3d(seed=42)

    stim, trace, dt, t0, dt_rel_error = prepare_noise_data(
        trace=trace, tracetime=tracetime, stim=stim, triggertimes=stimtime,
        fupsample_trace=2, fit_kind='trace', lowpass_cutoff=0)

    x_dict, y_dict = fit_rf_utils.split_data(x=stim, y=trace, frac_train=0.8, frac_dev=0.2, as_dict=True)

    assert set(x_dict.keys()) == {'train', 'dev'}
    assert np.isclose(y_dict['dev'].size / y_dict['train'].size, 0.2 / 0.8, atol=0.01, rtol=0.01)


def test_split_train_test():
    stim, stimtime, trace, tracetime, trace_dt = generate_data_3d(seed=461)

    stim, trace, dt, t0, dt_rel_error = prepare_noise_data(
        trace=trace, tracetime=tracetime, stim=stim, triggertimes=stimtime,
        fupsample_trace=2, fit_kind='trace', lowpass_cutoff=0)

    x_dict, y_dict = fit_rf_utils.split_data(x=stim, y=trace, frac_train=0.8, frac_dev=0.0, as_dict=True)

    assert set(x_dict.keys()) == {'train', 'test'}
    assert np.isclose(y_dict['test'].size / y_dict['train'].size, 0.2 / 0.8, atol=0.01, rtol=0.01)


def test_split_train_dev_test():
    stim, stimtime, trace, tracetime, trace_dt = generate_data_3d(seed=50)

    stim, trace, dt, t0, dt_rel_error = prepare_noise_data(
        trace=trace, tracetime=tracetime, stim=stim, triggertimes=stimtime,
        fupsample_trace=1, fit_kind='trace', lowpass_cutoff=0)

    x_dict, y_dict = fit_rf_utils.split_data(x=stim, y=trace, frac_train=0.6, frac_dev=0.2, as_dict=True)

    assert set(x_dict.keys()) == {'train', 'dev', 'test'}
    assert np.isclose(y_dict['dev'].size / y_dict['train'].size, 0.2 / 0.6, atol=0.01, rtol=0.01)
    assert np.isclose(y_dict['test'].size / y_dict['train'].size, 0.2 / 0.6, atol=0.01, rtol=0.01)
    assert np.isclose(y_dict['dev'].size / y_dict['test'].size, 1., atol=0.01, rtol=0.01)


@pytest.mark.skipif(rfest is None, reason="requires rfest")
def test_sta_fit():
    np.random.seed(13488)

    w_true, X, y, dt, dims = generate_data_3d_stim(
        stim_noise='white', rf_kind='gauss', response_noise='none', design_matrix=True,
        n_stim_frames=500, n_reps_per_frame=3, shift=0)

    w_fit = fit_rf_utils.compute_rf_sta(X=X, y=y)
    fit_mse, random_mses = compare_fits(w_true=w_true, w_fit=w_fit)
    assert fit_mse < np.mean(random_mses) - 2 * np.std(random_mses)


@pytest.mark.skipif(rfest is None, reason="requires rfest")
def test_sta_fit_few_datapoints():
    np.random.seed(364)

    w_true, X, y, dt, dims = generate_data_3d_stim(
        stim_noise='white', rf_kind='gauss', response_noise='gaussian', design_matrix=True,
        n_stim_frames=13, n_reps_per_frame=3, shift=0)

    w_fit = fit_rf_utils.compute_rf_sta(X=X, y=y)
    fit_mse, random_mses = compare_fits(w_true=w_true, w_fit=w_fit)
    assert fit_mse < np.mean(random_mses)


@pytest.mark.skipif(rfest is None, reason="requires rfest")
def test_sta_fit_shift():
    np.random.seed(434)

    w_true, X, y, dt, dims = generate_data_3d_stim(
        stim_noise='white', rf_kind='gauss', response_noise='gaussian', design_matrix=True,
        n_stim_frames=1000, n_reps_per_frame=1, shift=5)

    w_fit = fit_rf_utils.compute_rf_sta(X=X, y=y)
    fit_mse, random_mses = compare_fits(w_true=w_true, w_fit=w_fit)
    assert fit_mse < np.mean(random_mses)


@pytest.mark.skipif(rfest is None, reason="requires rfest")
def test_sta_response():
    np.random.seed(12234)

    w_true, X, y, dt, dims = generate_data_3d_stim(
        stim_noise='white', rf_kind='gauss', response_noise='gaussian', design_matrix=True,
        n_stim_frames=1000, n_reps_per_frame=1, shift=5)
    w_fit = fit_rf_utils.compute_rf_sta(X=X, y=y)
    y_pred = X @ w_fit.ravel()
    assert np.corrcoef(y, y_pred)[0, 1] > 0.5


@pytest.mark.skipif(rfest is None, reason="requires rfest")
def test_sta_response_splits():
    np.random.seed(42)

    w_true, X, y, dt, dims = generate_data_3d_stim(
        stim_noise='white', rf_kind='gauss', response_noise='gaussian', design_matrix=True,
        n_stim_frames=3000, n_reps_per_frame=1, shift=5)

    (x_trn, y_trn), (_, _), (x_tst, y_tst) = fit_rf_utils.split_data(X, y, frac_train=0.8, frac_dev=0.)

    w_fit = fit_rf_utils.compute_rf_sta(X=x_trn, y=y_trn)

    y_pred_train = x_trn @ w_fit.ravel()
    y_pred_test = x_tst @ w_fit.ravel()

    assert np.corrcoef(y_trn, y_pred_train)[0, 1] > 0.5
    assert np.corrcoef(y_tst, y_pred_test)[0, 1] > 0.25


def _correlated_stim_data(n=20000, d=6, seed=0):
    rng = np.random.default_rng(seed)
    x = np.ascontiguousarray((rng.standard_normal((n, d)) @ rng.standard_normal((d, d))).astype(np.float32))
    w_true = rng.standard_normal(d).astype(np.float32)
    u = x @ w_true
    y = np.exp(0.5 * u / np.std(u)).astype(np.float32)
    return x, y, w_true


def test_whitening_matrix_inverts_covariance():
    x, _, _ = _correlated_stim_data()
    w = fit_rf_utils.compute_whitening_matrix(x, reg=0.)
    cov = x.T.astype(np.float64) @ x / x.shape[0]
    assert w.dtype == x.dtype
    np.testing.assert_allclose(w.astype(np.float64) @ cov, np.eye(x.shape[1]), atol=1e-3)


def test_whitened_sta_recovers_filter_for_correlated_stimulus():
    x, y, w_true = _correlated_stim_data()
    sta = fit_rf_utils.compute_rf_sta(x, y)
    wsta = fit_rf_utils.compute_whitening_matrix(x, reg=1e-6) @ sta

    corr_sta = np.corrcoef(sta, w_true)[0, 1]
    corr_wsta = np.corrcoef(wsta, w_true)[0, 1]
    assert corr_wsta > 0.99
    assert corr_wsta > corr_sta


def test_whitening_regularization_shrinks_towards_sta_direction():
    x, _, _ = _correlated_stim_data()
    # Large ridge: C + lam*I ~ lam*I, so the whitening matrix becomes (nearly) a scaled identity.
    w = fit_rf_utils.compute_whitening_matrix(x, reg=1e6)
    off_diag = w - np.diag(np.diag(w))
    assert np.max(np.abs(off_diag)) < 1e-3 * np.max(np.abs(np.diag(w)))
