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
                        dtype: type = None) -> np.ndarray:
    """Build design matrix for linear RF estimation.

    Modified from RFEst. Works for multi-color stimuli by flattening all
    non-temporal dimensions (spatial and color) into a single feature axis.

    Parameters
    ----------
    X : np.ndarray
        Stimulus array, shape (n_frames, ...). For multi-color stimuli this
        can be e.g. (n_frames, n_x, n_y, n_colors).
    n_lag : int
        Number of temporal lags to include.
    shift : int, optional
        Temporal shift applied to the design matrix (negative shifts into future). Default is 0.
    dtype : type, optional
        Output data type. If None, uses X.dtype.

    Returns
    -------
    np.ndarray
        Design matrix, shape (n_frames, n_lag * n_feature) where
        n_feature = product of all non-temporal dimensions.
    """
    if dtype is None:
        dtype = X.dtype

    n_frames = X.shape[0]
    n_feature = np.prod(X.shape[1:])

    X_design = np.reshape(X.copy(), (n_frames, n_feature))

    if n_lag + shift > 0:
        X_design = np.vstack([np.zeros([n_lag + shift - 1, n_feature]), X_design])

    if shift < 0:
        X_design = np.vstack([X_design, np.zeros([-shift, n_feature])])

    X_design = np.hstack([X_design[i:n_frames + i] for i in range(n_lag)])

    return X_design.astype(dtype)
