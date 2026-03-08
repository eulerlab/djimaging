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


def compute_linear_rf(dt: float, trace: np.ndarray, stim: np.ndarray,
                      frac_train: float, frac_dev: float,
                      filter_dur_s_past: float, filter_dur_s_future: float = 0.,
                      kind: str = 'sta', threshold_pred: bool = False,
                      dtype: type = np.float32,
                      batch_size_n: int = 6_000_000,
                      verbose: bool = False) -> tuple:
    """Compute a linear receptive field (STA or MLE) from stimulus and response.

    Parameters
    ----------
    dt : float
        Time step in seconds.
    trace : np.ndarray
        Neural response trace, shape (n_frames,).
    stim : np.ndarray
        Stimulus array, shape (n_frames, ...).
    frac_train : float
        Fraction of data used for training.
    frac_dev : float
        Fraction of data used for development/validation.
    filter_dur_s_past : float
        Duration of the filter in the past (seconds).
    filter_dur_s_future : float, optional
        Duration of the filter in the future (seconds). Default is 0.
    kind : str, optional
        Estimation method: 'sta' or 'mle'. Default is 'sta'.
    threshold_pred : bool, optional
        Whether to clip predictions at zero. Default is False.
    dtype : type, optional
        Data type for computations. Default is np.float32.
    batch_size_n : int, optional
        Maximum number of elements before switching to batch mode. Default is 6_000_000.
    verbose : bool, optional
        Whether to print progress. Default is False.

    Returns
    -------
    tuple
        rf : np.ndarray
            Estimated receptive field reshaped to (dim_t, ...).
        rf_time : np.ndarray
            Time axis for the RF.
        model_eval : dict
            Evaluation metrics including predictions and correlation coefficients.
        x : dict
            Split stimulus data.
        y : dict
            Split trace data.
        shift : int
            Temporal shift applied during design matrix construction.
    """
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


def compute_linear_rf_single_or_batch(x_train: np.ndarray, y_train: np.ndarray,
                                      dim_t: int, shift: int, burn_in: int,
                                      kind: str, threshold_pred: bool,
                                      stim_shape: tuple,
                                      batch_size_n: int = 6_000_000,
                                      is_design_matrix: bool = False,
                                      dtype: type = np.float32,
                                      verbose: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Compute linear RF either in one pass or in batches depending on data size.

    Parameters
    ----------
    x_train : np.ndarray
        Training stimulus data.
    y_train : np.ndarray
        Training neural response data.
    dim_t : int
        Number of temporal filter taps.
    shift : int
        Temporal shift for the design matrix.
    burn_in : int
        Number of initial frames to skip.
    kind : str
        Estimation method: 'sta' or 'mle'.
    threshold_pred : bool
        Whether to clip predictions at zero.
    stim_shape : tuple
        Original shape of the stimulus array.
    batch_size_n : int, optional
        Maximum number of elements before switching to batch mode. Default is 6_000_000.
    is_design_matrix : bool, optional
        Whether x_train is already a design matrix. Default is False.
    dtype : type, optional
        Data type for computations. Default is np.float32.
    verbose : bool, optional
        Whether to print progress. Default is False.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        rf : np.ndarray
            Estimated receptive field.
        y_pred_train : np.ndarray
            Predicted training responses.
    """
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


def _compute_linear_rf_single_batch(x_train: np.ndarray, y_train: np.ndarray,
                                    dim_t: int, shift: int, burn_in: int,
                                    kind: str, threshold_pred: bool,
                                    is_design_matrix: bool = False,
                                    dtype: type = np.float32) -> tuple[np.ndarray, np.ndarray]:
    """Compute linear RF from a single batch of training data.

    Parameters
    ----------
    x_train : np.ndarray
        Training stimulus data or design matrix.
    y_train : np.ndarray
        Training neural response data.
    dim_t : int
        Number of temporal filter taps.
    shift : int
        Temporal shift for the design matrix.
    burn_in : int
        Number of initial frames to skip.
    kind : str
        Estimation method: 'sta' or 'mle'.
    threshold_pred : bool
        Whether to clip predictions at zero.
    is_design_matrix : bool, optional
        Whether x_train is already a design matrix. Default is False.
    dtype : type, optional
        Data type for computations. Default is np.float32.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        rf : np.ndarray
            Estimated receptive field.
        y_pred_train : np.ndarray
            Predicted training responses.
    """
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


def _compute_sta_batchwise(x_train: np.ndarray, y_train: np.ndarray,
                           dim_t: int, shift: int, burn_in: int,
                           kind: str, threshold_pred: bool,
                           batch_size: int = 400,
                           is_design_matrix: bool = False,
                           dtype: type = np.float32,
                           verbose: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Compute STA receptive field in batches over feature dimensions.

    Parameters
    ----------
    x_train : np.ndarray
        Training stimulus data, shape (n_frames, ...).
    y_train : np.ndarray
        Training neural response data, shape (n_frames,).
    dim_t : int
        Number of temporal filter taps.
    shift : int
        Temporal shift for the design matrix.
    burn_in : int
        Number of initial frames to skip.
    kind : str
        Estimation method: 'sta' or 'mle'.
    threshold_pred : bool
        Whether to clip predictions at zero.
    batch_size : int, optional
        Number of features per batch. Default is 400.
    is_design_matrix : bool, optional
        Whether x_train is already a design matrix. Default is False.
    dtype : type, optional
        Data type for computations. Default is np.float32.
    verbose : bool, optional
        Whether to display a progress bar. Default is False.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        rf : np.ndarray
            Estimated receptive field, shape (dim_t, ...).
        y_pred_train : np.ndarray
            Summed predicted training responses across batches.
    """
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


def get_rf_timing_params(filter_dur_s_past: float,
                         filter_dur_s_future: float,
                         dt: float) -> tuple[np.ndarray, int, int, int]:
    """Compute temporal parameters for the receptive field filter.

    Parameters
    ----------
    filter_dur_s_past : float
        Duration of the filter extending into the past (seconds).
    filter_dur_s_future : float
        Duration of the filter extending into the future (seconds).
    dt : float
        Time step in seconds.

    Returns
    -------
    tuple[np.ndarray, int, int, int]
        rf_time : np.ndarray
            Time axis of the filter (seconds), negative = past.
        dim_t : int
            Total number of temporal filter taps.
        shift : int
            Temporal shift (negative for future components).
        burn_in : int
            Number of initial frames to discard.
    """
    n_t_past = int(np.ceil(filter_dur_s_past / dt))
    n_t_future = int(np.ceil(filter_dur_s_future / dt))
    shift = -n_t_future
    dim_t = n_t_past + n_t_future
    rf_time = np.arange(-n_t_past + 1, n_t_future + 1) * dt
    burn_in = n_t_past
    return rf_time, dim_t, shift, burn_in


@jit(nopython=True)
def compute_rf_sta(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute spike-triggered average (STA) receptive field. From RFEst.

    Parameters
    ----------
    X : np.ndarray
        Design matrix, shape (n_samples, n_features).
    y : np.ndarray
        Neural response vector, shape (n_samples,).

    Returns
    -------
    np.ndarray
        STA receptive field, shape (n_features,).
    """
    w = (X.T @ y) / np.sum(y)
    return w


@jit(nopython=True)
def compute_rf_mle(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute maximum likelihood estimate (MLE) receptive field. From RFEst.

    Parameters
    ----------
    X : np.ndarray
        Design matrix, shape (n_samples, n_features).
    y : np.ndarray
        Neural response vector, shape (n_samples,).

    Returns
    -------
    np.ndarray
        MLE receptive field, shape (n_features,).
    """
    XtY = X.T @ y
    XtX = X.T @ X
    w = np.linalg.lstsq(XtX, XtY)[0]
    return w


def predict_linear_rf_response(rf: np.ndarray, stim_design_matrix: np.ndarray,
                                threshold: bool = False,
                                dtype: type = np.float32) -> np.ndarray:
    """Predict neural response using a linear receptive field and design matrix.

    Parameters
    ----------
    rf : np.ndarray
        Receptive field weights, shape (n_features,).
    stim_design_matrix : np.ndarray
        Stimulus design matrix, shape (n_samples, n_features).
    threshold : bool, optional
        Whether to clip negative predictions to zero. Default is False.
    dtype : type, optional
        Output data type. Default is np.float32.

    Returns
    -------
    np.ndarray
        Predicted neural response, shape (n_samples,).
    """
    y_pred = stim_design_matrix @ rf

    if threshold:
        y_pred = np.clip(y_pred, 0, None)

    return y_pred.astype(dtype)


def split_data(x: np.ndarray, y: np.ndarray,
               frac_train: float = 0.8, frac_dev: float = 0.1,
               as_dict: bool = False):
    """Split data into training, development and test set.

    Modified from RFEst.

    Parameters
    ----------
    x : np.ndarray
        Input data (e.g. stimulus), shape (n_samples, ...).
    y : np.ndarray
        Target data (e.g. response), shape (n_samples, ...).
    frac_train : float, optional
        Fraction of data for training. Default is 0.8.
    frac_dev : float, optional
        Fraction of data for development/validation. Default is 0.1.
    as_dict : bool, optional
        If True, return dicts keyed by split name. Default is False.

    Returns
    -------
    tuple or tuple[dict, dict]
        If as_dict is False: ((x_trn, y_trn), (x_dev, y_dev), (x_tst, y_tst)).
        If as_dict is True: (x_dict, y_dict) with keys 'train', 'dev', 'test'.

    Raises
    ------
    AssertionError
        If x and y have different lengths or fractions exceed 1.
    """
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


def build_design_matrix(X: np.ndarray, n_lag: int, shift: int = 0,
                        n_c: int = 1, dtype: type = None) -> np.ndarray:
    """Build design matrix for linear RF estimation.

    Modified from RFEst.

    Parameters
    ----------
    X : np.ndarray
        Stimulus array, shape (n_frames, ...).
    n_lag : int
        Number of temporal lags to include.
    shift : int, optional
        Temporal shift applied to the design matrix (negative shifts into future). Default is 0.
    n_c : int, optional
        Number of color channels (adds an extra dimension if > 1). Default is 1.
    dtype : type, optional
        Output data type. If None, uses X.dtype.

    Returns
    -------
    np.ndarray
        Design matrix, shape (n_frames, n_lag * n_feature) or
        (n_frames, n_lag * n_feature / n_c, n_c) if n_c > 1.
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
