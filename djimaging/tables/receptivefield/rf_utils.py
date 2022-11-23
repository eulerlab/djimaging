import warnings

import numpy as np
from rfest import get_spatial_and_temporal_filters
from scipy.signal import find_peaks

from djimaging.utils import math_utils, filter_utils

try:
    import rfest
    from jax.lib import xla_bridge

    xla_bridge.get_backend()
    del xla_bridge
except ImportError:
    warnings.warn('Failed to import RFEst: Cannot compute receptive fields.')
    rfest = None


def apply_filter_to_trace(trace, dt, cutoff):
    """Apply lowpass filter"""
    flt = filter_utils.LowPassFilter(fs=1 / dt, cutoff=cutoff, direction='ff')
    return flt.filter_data(trace)


def resample_trace(tracetime, trace, dt):
    """Resample trace through linear interpolation"""
    tracetime_resampled = np.arange(tracetime[0], tracetime[-1] + dt / 10., dt)
    trace_resampled = np.interp(x=tracetime_resampled, xp=tracetime, fp=trace)
    return tracetime_resampled, trace_resampled


def upsample_trace(tracetime, trace, fupsample):
    assert isinstance(fupsample, int)
    assert fupsample > 1

    dt = np.mean(np.diff(tracetime))
    tracetime_upsampled = np.tile(tracetime, (fupsample, 1)).T
    tracetime_upsampled += np.linspace(0, 1, fupsample, endpoint=False) * dt
    tracetime_upsampled = tracetime_upsampled.flatten()

    diffs = np.diff(tracetime_upsampled)
    mu = np.mean(diffs)
    std = np.std(diffs)
    assert np.isclose(mu, dt / fupsample, atol=mu / 10.), f"{mu} {dt} {fupsample}"
    assert mu > 10 * std

    trace_upsampled = np.interp(tracetime_upsampled, tracetime, trace)

    return tracetime_upsampled, trace_upsampled


def get_sets(stim, stimtime, trace, tracetime, frac_train=1., frac_dev=0., fupsample=1, gradient=False,
             norm_stim=True, norm_trace=True, filter_trace=False, cutoff=10.):
    """Split data into sets"""
    assert rfest is not None
    assert frac_dev + frac_train <= 1.0

    assert stimtime.ndim == 1, stimtime.shape
    assert trace.ndim == 1, trace.shape
    assert tracetime.ndim == 1, tracetime.shape
    assert stim.shape[0] == stimtime.size
    assert trace.size == tracetime.size

    dts = np.diff(tracetime)
    dt = np.mean(dts)
    dt_max_diff = np.max(dts) - np.max(dts)

    if (dt_max_diff / dt) > 0.1 or fupsample > 1:  # No large difference between dts
        if fupsample <= 1:
            warnings.warn('Inconsistent step-sizes in trace, resample trace.')
        tracetime, trace = resample_trace(tracetime=tracetime, trace=trace, dt=dt / fupsample)

    X, y, dt = rfest.utils.upsample_data(
        stim=stim, stimtime=stimtime, trace=trace, tracetime=tracetime, gradient=gradient)

    if norm_stim:
        X = math_utils.normalize_zscore(X)

    if filter_trace:
        y = apply_filter_to_trace(y, dt=dt, cutoff=cutoff)

    if norm_trace:
        y = math_utils.normalize_zscore(y)

    (X_train, y_train), (X_dev, y_dev), (X_test, y_test) = rfest.utils.split_data(
        X, y, dt, verbose=False, frac_train=frac_train, frac_dev=frac_dev)

    X_dict = dict(train=X_train)
    y_dict = dict(train=y_train)

    if frac_dev > 0. or y_dev.size > 0:
        X_dict['dev'] = X_dev
        y_dict['dev'] = y_dev

    if frac_dev + frac_train < 1. and y_test.size > 0:
        X_dict['test'] = X_test
        y_dict['test'] = y_test

    return X_dict, y_dict, dt


def compute_explained_rf(rf, rf_fit):
    return np.maximum(1. - np.var(rf - rf_fit) / np.var(rf), 0.)


def smooth_rf(rf, blur_std, blur_npix):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(rf, mode='nearest', sigma=blur_std, truncate=np.floor(blur_npix / blur_std), order=0)


def resize_srf(srf, scale=None, output_shape=None):
    """Change size of sRF, used for up or down sampling"""
    from skimage.transform import resize
    if scale is not None:
        output_shape = np.array(srf.shape) * scale
    else:
        assert output_shape is not None
    return resize(srf, output_shape=output_shape, mode='constant', order=1)


def split_strf(strf, blur_std: float = 0, blur_npix: int = 1, upsample_srf_scale: int = 0):
    """Split STRF into sRF and tRF"""
    if blur_std > 0:
        strf = np.stack([smooth_rf(rf=srf_i, blur_std=blur_std, blur_npix=blur_npix) for srf_i in strf])

    srf, trf = get_spatial_and_temporal_filters(strf, strf.shape)

    if upsample_srf_scale > 1:
        srf = resize_srf(srf, scale=upsample_srf_scale)

    return srf, trf


def compute_polarity_and_peak_idxs(trf, nstd=1.):
    """Estimate polarity. 1 for ON-cells, -1 for OFF-cells, 0 for uncertain cells"""

    trf = trf.copy()
    std_trf = np.std(trf)

    pos_peak_idxs, _ = find_peaks(trf, prominence=nstd * std_trf / 2., height=nstd * std_trf)
    neg_peak_idxs, _ = find_peaks(-trf, prominence=nstd * std_trf / 2., height=nstd * std_trf)

    peak_idxs = np.sort(np.concatenate([pos_peak_idxs, neg_peak_idxs]))

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


def compute_sta(trace, tracetime, stim, stimtime, frac_train, frac_dev, dur_filter_s,
                kind='sta', fupsample=1, gradient=False, norm_stim=True, norm_trace=True):
    assert trace.ndim == 1
    assert tracetime.ndim == 1
    assert trace.size == tracetime.size
    assert stim.shape[0] == stimtime.shape[0], f"{stim.shape} vs. {stimtime.shape}"

    kind = kind.lower()

    X, y, dt = get_sets(
        stim=stim, stimtime=stimtime, trace=trace, tracetime=tracetime,
        frac_train=frac_train, frac_dev=frac_dev, fupsample=fupsample, gradient=gradient,
        norm_stim=norm_stim, norm_trace=norm_trace)

    if kind in ['sta', 'mle']:
        assert 'dev' not in X and 'dev' not in y, 'Development sets are not use for these rfs'
        rf, rf_pred = fit_sta(X, y, dur_filter_s, dt, kind=kind)
    else:
        raise NotImplementedError(f"kind={kind}")

    return rf, rf_pred, X, y, dt


def fit_sta(X, y, dur_filter_s, dt, kind='sta'):
    """Compute STA or MLE"""
    assert rfest is not None
    from rfest.GLM._base import Base as BaseModel

    kind = kind.lower()
    assert kind in ['sta', 'mle'], kind

    dim_t = int(np.ceil(dur_filter_s / dt))
    dims = (dim_t,) + X['train'].shape[1:]

    burn_in = dims[0] - 1

    X_train_dm = rfest.utils.build_design_matrix(X['train'], dims[0])[burn_in:]
    y_train_dm = y['train'][burn_in:]

    model = BaseModel(X=X_train_dm, y=y_train_dm, dims=dims, compute_mle=kind == 'mle')
    rf = model.w_sta if kind == 'sta' else model.w_mle

    rf_pred = dict()
    rf_pred['burn_in'] = burn_in
    y_pred_train = X_train_dm @ rf
    rf_pred['y_pred_train'] = y_pred_train
    rf_pred['cc_train'] = np.corrcoef(y['train'][burn_in:], y_pred_train)[0, 1]
    rf_pred['mse_train'] = np.mean((y['train'][burn_in:] - y_pred_train) ** 2)

    if 'test' in X:
        X_test_dm = rfest.utils.build_design_matrix(X['test'], dims[0])[burn_in:]
        y_pred_test = X_test_dm @ rf
        rf_pred['y_pred_test'] = y_pred_test
        rf_pred['cc_test'] = np.corrcoef(y['test'][burn_in:], y_pred_test)[0, 1]
        rf_pred['mse_test'] = np.mean((y['test'][burn_in:] - y_pred_test) ** 2)

    return rf.reshape(dims), rf_pred
