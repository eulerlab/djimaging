import warnings

import numpy as np
from numba import jit
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
from sklearn.decomposition import randomized_svd

from djimaging.utils import math_utils, filter_utils
from djimaging.utils.filter_utils import upsample_stim, lowpass_filter_trace
from djimaging.utils.image_utils import resize_image
from djimaging.utils.trace_utils import align_stim_to_trace, get_mean_dt, align_trace_to_stim


def compute_linear_rf(dt, trace, stim, frac_train, frac_dev,
                      filter_dur_s_past, filter_dur_s_future=0.,
                      kind='sta', threshold_pred=False, dtype=np.float32):
    kind = kind.lower()

    assert trace.size == stim.shape[0]
    assert kind in ['sta', 'mle'], f"kind={kind}"

    rf_time, dim_t, shift, burn_in = get_rf_timing_params(filter_dur_s_past, filter_dur_s_future, dt)
    dims = (dim_t,) + stim.shape[1:]

    x, y = split_data(x=stim, y=trace, frac_train=frac_train, frac_dev=frac_dev, as_dict=True)

    x_train_dm = build_design_matrix(x['train'], n_lag=dim_t, shift=shift, dtype=dtype)[burn_in:]
    y_train_dm = y['train'][burn_in:].astype(dtype)

    if kind.lower() == 'sta':
        rf = compute_rf_sta(X=x_train_dm, y=y_train_dm, is_spikes=False)
    elif kind.lower() == 'mle':
        rf = compute_rf_mle(X=x_train_dm, y=y_train_dm)
    else:
        raise NotImplementedError()

    y_pred_train = predict_linear_rf_response(
        rf=rf, stim_design_matrix=x_train_dm, threshold=threshold_pred, dtype=dtype)

    model_eval = dict()
    model_eval['burn_in'] = burn_in
    model_eval['y_pred_train'] = y_pred_train
    model_eval['cc_train'] = np.corrcoef(y['train'][burn_in:], y_pred_train)[0, 1]
    model_eval['mse_train'] = np.mean((y['train'][burn_in:] - y_pred_train) ** 2)

    if 'test' in x:
        x_test_dm = build_design_matrix(x['test'], n_lag=dim_t, shift=shift, dtype=dtype)[burn_in:]
        y_pred_test = predict_linear_rf_response(
            rf=rf, stim_design_matrix=x_test_dm, threshold=threshold_pred, dtype=dtype)

        model_eval['y_pred_test'] = y_pred_test
        model_eval['cc_test'] = np.corrcoef(y['test'][burn_in:], y_pred_test)[0, 1]
        model_eval['mse_test'] = np.mean((y['test'][burn_in:] - y_pred_test) ** 2)

    rf = rf.reshape(dims)
    return rf, rf_time, model_eval, x, y, shift


def get_rf_timing_params(filter_dur_s_past, filter_dur_s_future, dt):
    n_t_past = int(np.ceil(filter_dur_s_past / dt))
    n_t_future = int(np.ceil(filter_dur_s_future / dt))
    shift = -n_t_future
    dim_t = n_t_past + n_t_future
    rf_time = np.arange(-n_t_past + 1, n_t_future + 1) * dt
    burn_in = n_t_past
    return rf_time, dim_t, shift, burn_in


@jit(nopython=True)
def compute_rf_sta(X, y, is_spikes=False):
    """From RFEst"""
    XtY = X.T @ y
    if is_spikes:
        w = XtY / np.sum(y)
    else:
        w = XtY / len(y)

    return w


@jit(nopython=True)
def compute_rf_mle(X, y):
    """From RFEst"""
    XtY = X.T @ y
    XtX = X.T @ X
    w = np.linalg.lstsq(XtX, XtY)[0]
    return w


def predict_linear_rf_response(rf, stim_design_matrix, threshold=False, dtype=np.float32):
    y_pred = stim_design_matrix @ rf

    if threshold:
        y_pred = np.clip(y_pred, 0, None)

    return y_pred.astype(dtype)


def prepare_data(stim, stimtime, trace, tracetime, fupsample_trace=None, fupsample_stim=None, fit_kind='trace',
                 lowpass_cutoff=0, ref_time='trace', pre_blur_sigma_s=0, post_blur_sigma_s=0, dt_rtol=0.1):
    assert stimtime.ndim == 1, stimtime.shape
    assert trace.ndim == 1, trace.shape
    assert tracetime.ndim == 1, tracetime.shape
    assert trace.size == tracetime.size

    if stim.shape[0] < stimtime.size:
        raise ValueError(f"More triggertimes than expected: stim-len {stim.shape[0]} != stimtime-len {stimtime.size}")
    elif stim.shape[0] > stimtime.size:
        warnings.warn(f"Less triggertimes than expected: stim-len {stim.shape[0]} != stimtime-len {stimtime.size}")
        stim = stim[:stimtime.size].copy()

    dt, dt_rel_error = get_mean_dt(tracetime)

    if dt_rel_error > dt_rtol:
        warnings.warn('Inconsistent step-sizes in trace, resample trace.')
        tracetime, trace = filter_utils.resample_trace(tracetime=tracetime, trace=trace, dt=dt)

    if lowpass_cutoff > 0:
        trace = lowpass_filter_trace(trace=trace, fs=1. / dt, f_cutoff=lowpass_cutoff)

    if pre_blur_sigma_s > 0:
        trace = gaussian_filter(trace, sigma=pre_blur_sigma_s / dt, mode='nearest')

    tracetime, trace = prepare_trace(tracetime, trace, kind=fit_kind, fupsample=fupsample_trace, dt=dt)

    if (fupsample_stim is not None) and (fupsample_stim > 1):
        stimtime, stim = upsample_stim(stimtime, stim, fupsample=fupsample_stim)

    if ref_time == 'trace':
        stim, trace, dt, t0, dt_rel_error = align_stim_to_trace(
            stim=stim, stimtime=stimtime, trace=trace, tracetime=tracetime)
    elif ref_time == 'stim':
        stim, trace, dt, t0, dt_rel_error = align_trace_to_stim(
            stim=stim, stimtime=stimtime, trace=trace, tracetime=tracetime)
    else:
        raise NotImplementedError

    if post_blur_sigma_s > 0:
        trace = gaussian_filter(trace, sigma=post_blur_sigma_s / dt, mode='nearest')

    trace = trace / np.std(trace)

    if 'bool' in str(stim.dtype) or set(np.unique(stim).astype(float)) == {0., 1.}:
        stim = 2 * stim.astype(np.int8) - 1
    elif 'int' in str(stim.dtype):
        stim = stim - np.mean(stim, dtype='int')
    else:
        stim = math_utils.normalize_zscore(stim)

    assert stim.shape[0] == trace.shape[0], (stim.shape[0], trace.shape[0])

    return stim, trace, dt, t0, dt_rel_error


def prepare_trace(tracetime, trace, kind='trace', fupsample=None, dt=None):
    if fupsample is None:
        fupsample = 1
    else:
        assert dt is not None

    if kind == 'trace':
        if fupsample > 1:
            fit_tracetime, fit_trace = filter_utils.upsample_trace(
                tracetime=tracetime, trace=trace, fupsample=fupsample)
        else:
            fit_tracetime, fit_trace = tracetime, trace

    elif kind == 'gradient':
        diff_trace = np.append(0, np.diff(trace))
        if fupsample > 1:
            fit_tracetime, fit_trace = filter_utils.upsample_trace(
                tracetime=tracetime, trace=diff_trace, fupsample=fupsample)
        else:
            fit_tracetime, fit_trace = tracetime, diff_trace

        fit_trace = np.clip(fit_trace, 0, None)

    elif kind == 'events':
        # Baden et al 2016
        diff_trace = np.append(0, np.diff(trace))
        if fupsample > 1:
            fit_tracetime, diff_trace = filter_utils.upsample_trace(
                tracetime=tracetime, trace=diff_trace, fupsample=fupsample)
        else:
            fit_tracetime = tracetime

        robust_std = np.median(np.abs(diff_trace)) / 0.6745
        peaks, props = find_peaks(diff_trace, height=robust_std)

        fit_trace = np.zeros(fit_tracetime.size)
        fit_trace[peaks] = props['peak_heights']
    else:
        raise NotImplementedError(kind)

    assert fit_tracetime.size == fit_trace.size, f"{fit_tracetime.size} != {fit_trace.size}"

    return fit_tracetime, fit_trace


def split_data(x, y, frac_train=0.8, frac_dev=0.1, as_dict=False):
    """ Split data into training, development and test set.
     Modified from RFEst"""
    assert x.shape[0] == y.shape[0], 'X and y must be of same length.'
    assert frac_train + frac_dev <= 1, '`frac_train` + `frac_dev` must be < 1.'

    n_samples = x.shape[0]

    idx1 = int(n_samples * frac_train)
    idx2 = int(n_samples * (frac_train + frac_dev))

    x_trn, x_dev, x_tst = np.split(x, [idx1, idx2])
    y_trn, y_dev, y_tst = np.split(y, [idx1, idx2])

    if not as_dict:
        return (x_trn, y_trn), (x_dev, y_dev), (x_tst, y_tst)
    else:
        x_dict = dict(train=x_trn)
        y_dict = dict(train=y_trn)

        if frac_dev > 0. and y_dev.size > 0:
            x_dict['dev'] = x_dev
            y_dict['dev'] = y_dev

        if frac_dev + frac_train < 1. and y_tst.size > 0:
            x_dict['test'] = x_tst
            y_dict['test'] = y_tst

        return x_dict, y_dict


def compute_explained_rf(rf, rf_fit):
    return 1. - (np.var(rf - rf_fit) / np.var(rf))


def smooth_rf(rf, blur_std, blur_npix):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(rf, mode='nearest', sigma=blur_std, truncate=np.floor(blur_npix / blur_std), order=0)


def resize_srf(srf, scale=None, output_shape=None):
    """Change size of sRF, used for up or down sampling"""
    if scale is not None:
        output_shape = np.array(srf.shape) * scale
    else:
        assert output_shape is not None
    return resize_image(srf, output_shape=output_shape, order=1)


def compute_polarity_and_peak_idxs(trf, nstd=1., npeaks_max=None):
    """Estimate polarity. 1 for ON-cells, -1 for OFF-cells, 0 for uncertain cells"""

    trf = trf.copy()
    std_trf = np.std(trf)

    pos_peak_idxs, _ = find_peaks(trf, prominence=nstd * std_trf / 2., height=nstd * std_trf)
    neg_peak_idxs, _ = find_peaks(-trf, prominence=nstd * std_trf / 2., height=nstd * std_trf)

    peak_idxs = np.sort(np.concatenate([pos_peak_idxs, neg_peak_idxs]))

    if (npeaks_max is not None) and (peak_idxs.size > npeaks_max):
        peak_idxs = peak_idxs[-npeaks_max:]

    if peak_idxs.size > 0:
        polarity = (trf[peak_idxs[-1]] > 0) * 2 - 1
    else:
        polarity = 0

    return polarity, peak_idxs


def merge_strf(srf, trf):
    """Reconstruct STRF from sRF and tRF"""
    assert trf.ndim == 1, trf.ndim
    assert srf.ndim == 2, srf.ndim
    rf = np.kron(trf, srf.flat).reshape(trf.shape + srf.shape)
    return rf


def build_design_matrix(X, n_lag, shift=0, n_c=1, dtype=None):
    """
    Build design matrix.
    Modified from RFEst.
    """
    if dtype is None:
        dtype = X.dtype

    n_sample = X.shape[0]
    n_feature = np.product(X.shape[1:])
    X = np.reshape(X, (n_sample, n_feature))

    if n_lag + shift > 0:
        X_padded = np.vstack([np.zeros([n_lag + shift - 1, n_feature]), X])
    else:
        X_padded = X

    if shift < 0:
        X_padded = np.vstack([X_padded, np.zeros([-shift, n_feature])])

    X_design = np.hstack([X_padded[i:n_sample + i] for i in range(n_lag)])

    if n_c > 1:
        X_design = np.reshape(X_design, (X_design.shape[0], -1, n_c)).astype(dtype)
    else:
        X_design = X_design.astype(dtype)

    return X_design


def split_strf(strf, blur_std: float = 0, blur_npix: int = 1, upsample_srf_scale: int = 0):
    """Split STRF into sRF and tRF"""
    if blur_std > 0:
        strf = np.stack([smooth_rf(rf=srf_i, blur_std=blur_std, blur_npix=blur_npix) for srf_i in strf])

    srf, trf = svd_rf(strf)

    if upsample_srf_scale > 1:
        srf = resize_srf(srf, scale=upsample_srf_scale)

    return srf, trf


def svd_rf(w):
    """
    Assuming an RF is time-space separable, get spatial and temporal filters using SVD.
    From RFEst.
    """
    dims = w.shape

    if len(dims) == 3:
        dims_tRF = dims[0]
        dims_sRF = dims[1:]
        U, S, Vt = randomized_svd(w.reshape(dims_tRF, -1), 3, random_state=0)
        srf = Vt[0].reshape(*dims_sRF)
        trf = U[:, 0]

    elif len(dims) == 2:
        dims_tRF = dims[0]
        dims_sRF = dims[1]
        U, S, Vt = randomized_svd(w.reshape(dims_tRF, dims_sRF), 3, random_state=0)
        srf = Vt[0]
        trf = U[:, 0]

    else:
        raise NotImplementedError

    return srf, trf
