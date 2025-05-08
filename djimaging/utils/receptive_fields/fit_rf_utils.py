"""
This file contains code derived or copied from RFEst, which is licensed under
the GNU General Public License version 3.0 (GNU GPL v3.0).

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

For details see: https://github.com/berenslab/RFEst
"""

import numpy as np
from numba import jit
from tqdm.notebook import tqdm


def compute_linear_rf(dt, trace, stim, frac_train, frac_dev,
                      filter_dur_s_past, filter_dur_s_future=0.,
                      kind='sta', threshold_pred=False, dtype=np.float32,
                      batch_size_n=6_000_000, verbose=False):
    kind = kind.lower()

    assert trace.size == stim.shape[0]
    assert kind in ['sta', 'mle'], f"kind={kind}"

    rf_time, dim_t, shift, burn_in = get_rf_timing_params(filter_dur_s_past, filter_dur_s_future, dt)
    dims = (dim_t,) + stim.shape[1:]

    x, y = split_data(x=stim, y=trace, frac_train=frac_train, frac_dev=frac_dev, as_dict=True)

    rf, y_pred_train = compute_linear_rf_single_or_batch(
        x_train=x['train'], y_train=y['train'], dim_t=dim_t, shift=shift, burn_in=burn_in, kind=kind,
        threshold_pred=threshold_pred, batch_size_n=batch_size_n, stim_shape=stim.shape, dtype=dtype, verbose=verbose)

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


def compute_linear_rf_single_or_batch(x_train, y_train, dim_t, shift, burn_in, kind, threshold_pred,
                                      stim_shape, batch_size_n=6_000_000, is_design_matrix=False, dtype=np.float32,
                                      verbose=False):
    if np.product(stim_shape) <= batch_size_n:
        rf, y_pred_train = _compute_linear_rf_single_batch(
            x_train=x_train, y_train=y_train, kind=kind, dim_t=dim_t, shift=shift, burn_in=burn_in,
            is_design_matrix=is_design_matrix, threshold_pred=threshold_pred, dtype=dtype)
    else:
        assert kind == 'sta', f"kind={kind} not supported for compute_per='feature'"
        rf, y_pred_train = _compute_sta_batchwise(
            x_train=x_train, y_train=y_train, kind=kind, dim_t=dim_t, shift=shift, burn_in=burn_in,
            threshold_pred=threshold_pred, dtype=dtype,
            is_design_matrix=is_design_matrix, batch_size=np.maximum(batch_size_n // stim_shape[0], 1), verbose=verbose)

    return rf, y_pred_train


def _compute_linear_rf_single_batch(x_train, y_train, dim_t, shift, burn_in, kind, threshold_pred,
                                    is_design_matrix=False, dtype=np.float32):
    if is_design_matrix:
        x_train_dm = x_train[burn_in:].astype(dtype)
    else:
        x_train_dm = build_design_matrix(x_train, n_lag=dim_t, shift=shift, dtype=dtype)[burn_in:]

    y_train_dm = y_train[burn_in:].astype(dtype)

    if kind == 'sta':
        rf = compute_rf_sta(X=x_train_dm, y=y_train_dm)
    elif kind == 'mle':
        rf = compute_rf_mle(X=x_train_dm, y=y_train_dm)
    else:
        raise NotImplementedError(kind)

    y_pred_train = predict_linear_rf_response(
        rf=rf, stim_design_matrix=x_train_dm, threshold=threshold_pred, dtype=dtype)

    return rf, y_pred_train


def _compute_sta_batchwise(x_train, y_train, dim_t, shift, burn_in, kind, threshold_pred,
                           batch_size=400, is_design_matrix=False, dtype=np.float32, verbose=False):
    x_shape = x_train.shape
    n_frames = x_shape[0]
    n_features = np.product(x_shape[1:])

    rf = np.zeros((dim_t, n_features), dtype=dtype)

    x_train = x_train.reshape((n_frames, n_features))
    y_train_dm = y_train[burn_in:].astype(dtype)

    batches = np.array_split(np.arange(n_features), np.ceil(n_features / batch_size))
    y_pred_train_per_batch = np.zeros((n_frames - burn_in, len(batches)), dtype=dtype)

    bar = tqdm(batches, desc='STA batches', leave=False) if verbose else None

    for i, batch in enumerate(batches):
        if is_design_matrix:
            x_train_dm_i = x_train[burn_in:, batch]
        else:
            x_train_dm_i = build_design_matrix(x_train[:, batch], n_lag=dim_t, shift=shift, dtype=dtype)[burn_in:]

        if kind == 'sta':
            rf_i = compute_rf_sta(X=x_train_dm_i, y=y_train_dm)
        elif kind == 'mle':
            rf_i = compute_rf_mle(X=x_train_dm_i, y=y_train_dm)
        else:
            raise NotImplementedError()

        rf[:, batch] = rf_i.reshape(-1, len(batch))
        y_pred_train_per_batch[:, i] = predict_linear_rf_response(
            rf=rf_i, stim_design_matrix=x_train_dm_i, threshold=False, dtype=dtype)

        if bar is not None:
            bar.update(1)

    if bar is not None:
        bar.close()

    # Reshape back. Is this necessary for x?
    rf = rf.reshape((dim_t,) + x_shape[1:])
    x_train.reshape(x_shape)

    y_pred_train = np.sum(y_pred_train_per_batch, axis=1)

    if threshold_pred:
        y_pred_train = np.clip(y_pred_train, 0, None)

    return rf, y_pred_train


def get_rf_timing_params(filter_dur_s_past, filter_dur_s_future, dt):
    n_t_past = int(np.ceil(filter_dur_s_past / dt))
    n_t_future = int(np.ceil(filter_dur_s_future / dt))
    shift = -n_t_future
    dim_t = n_t_past + n_t_future
    rf_time = np.arange(-n_t_past + 1, n_t_future + 1) * dt
    burn_in = n_t_past
    return rf_time, dim_t, shift, burn_in


@jit(nopython=True)
def compute_rf_sta(X, y):
    """From RFEst"""
    w = (X.T @ y) / np.sum(y)
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


def build_design_matrix(X, n_lag, shift=0, n_c=1, dtype=None):
    """
    Build design matrix.
    Modified from RFEst.
    """
    if dtype is None:
        dtype = X.dtype

    n_frames = X.shape[0]
    n_feature = np.product(X.shape[1:])

    X_design = np.reshape(X.copy(), (n_frames, n_feature))

    if n_lag + shift > 0:
        X_design = np.vstack([np.zeros([n_lag + shift - 1, n_feature]), X_design])

    if shift < 0:
        X_design = np.vstack([X_design, np.zeros([-shift, n_feature])])

    X_design = np.hstack([X_design[i:n_frames + i] for i in range(n_lag)])

    if n_c > 1:
        X_design = np.reshape(X_design, (X_design.shape[0], -1, n_c))

    return X_design.astype(dtype)
