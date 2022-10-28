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
    assert np.isclose(mu, dt / fupsample, atol=mu/10.), f"{mu} {dt} {fupsample}"
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
