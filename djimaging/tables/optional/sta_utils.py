import warnings

import numpy as np
from djimaging.utils import math_utils


# TODO: Merge with rf_glm_utils

try:
    import rfest
    from jax.lib import xla_bridge
    xla_bridge.get_backend()
    del xla_bridge
except ImportError:
    warnings.warn('Failed to import RFEst: Cannot compute receptive fields.')
    rfest = None


def compute_receptive_field(trace, tracetime, stim, stimtime, frac_train, frac_dev, dur_filter_s,
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
        rf, rf_pred = compute_rf(X, y, dur_filter_s, dt, kind=kind)
    else:
        raise NotImplementedError(f"kind={kind}")

    return rf, rf_pred, X, y, dt


def resample_trace(tracetime, trace, dt):
    """Resample trace through linear interpolation"""
    tracetime_resampled = np.arange(tracetime[0], tracetime[-1] + dt / 10., dt)
    trace_resampled = np.interp(x=tracetime_resampled, xp=tracetime, fp=trace)
    return tracetime_resampled, trace_resampled


def get_sets(stim, stimtime, trace, tracetime, frac_train=1., frac_dev=0., fupsample=1, gradient=False,
             norm_stim=True, norm_trace=True):
    """Split data into sets"""
    assert rfest is not None
    assert frac_dev + frac_train <= 1.0

    dts = np.diff(tracetime)
    dt = np.mean(dts)
    dt_max_diff = np.max(dts) - np.max(dts)

    if (dt_max_diff / dt) > 0.1 or fupsample > 1:  # No large difference between dts
        tracetime, trace = resample_trace(tracetime=tracetime, trace=trace, dt=dt / fupsample)

    X, y, dt = rfest.utils.upsample_data(stim=stim, stimtime=stimtime, trace=trace,
                                         tracetime=tracetime, gradient=gradient)

    if norm_stim:
        X = math_utils.normalize_zscore(X)

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


def compute_rf(X, y, dur_filter_s, dt, kind='sta'):
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
