import warnings

import numpy as np
from rfest import get_spatial_and_temporal_filters
from scipy.signal import find_peaks

from djimaging.utils import math_utils
from djimaging.utils.filter_utils import resample_trace

try:
    import rfest
    import jax

    jax.config.update('jax_platform_name', 'cpu')
except ImportError:
    warnings.warn('Failed to import RFEst: Cannot compute receptive fields.')
    rfest = None
    jax = None


def compute_sta(trace, tracetime, stim, stimtime, frac_train, frac_dev, dur_filter_s,
                kind='sta', norm_stim=True, norm_trace=True, fit_kind='trace'):
    kind = kind.lower()

    assert trace.ndim == 1
    assert tracetime.ndim == 1
    assert trace.size == tracetime.size
    assert stim.shape[0] == stimtime.shape[0], f"{stim.shape} vs. {stimtime.shape}"
    assert kind in ['sta', 'mle'], f"kind={kind}"
    assert rfest is not None
    from rfest.GLM._base import Base as BaseModel

    trace = prepare_trace(trace, kind=fit_kind)

    x, y, dt = get_sets(
        trace=trace, tracetime=tracetime, stim=stim, stimtime=stimtime,
        frac_train=frac_train, frac_dev=frac_dev, norm_stim=norm_stim, norm_trace=norm_trace)

    dim_t = int(np.ceil(dur_filter_s / dt))
    dims = (dim_t,) + x['train'].shape[1:]

    burn_in = dims[0] - 1

    X_train_dm = rfest.utils.build_design_matrix(x['train'], dims[0])[burn_in:]
    y_train_dm = y['train'][burn_in:]

    model = BaseModel(X=X_train_dm, y=y_train_dm, dims=dims, compute_mle=kind == 'mle')
    rf = model.w_sta if kind == 'sta' else model.w_mle

    y_pred_train = predict_sta_response(rf=rf, x=X_train_dm, fit_kind=fit_kind)

    model_eval = dict()
    model_eval['burn_in'] = burn_in
    model_eval['y_pred_train'] = y_pred_train
    model_eval['cc_train'] = np.corrcoef(y['train'][burn_in:], y_pred_train)[0, 1]
    model_eval['mse_train'] = np.mean((y['train'][burn_in:] - y_pred_train) ** 2)

    if 'test' in x:
        x_test_dm = rfest.utils.build_design_matrix(x['test'], dims[0])[burn_in:]
        y_pred_test = predict_sta_response(rf=rf, x=x_test_dm, fit_kind=fit_kind)

        model_eval['y_pred_test'] = y_pred_test
        model_eval['cc_test'] = np.corrcoef(y['test'][burn_in:], y_pred_test)[0, 1]
        model_eval['mse_test'] = np.mean((y['test'][burn_in:] - y_pred_test) ** 2)

    rf = rf.reshape(dims)
    return rf, model_eval, x, y, dt


def predict_sta_response(rf, x, fit_kind='trace'):
    y_pred = x @ rf

    if fit_kind in ['gradient', 'events']:
        y_pred = np.clip(y_pred, 0, None)

    return y_pred


def prepare_trace(trace, kind='trace'):
    if kind == 'trace':
        fit_trace = trace
    elif kind == 'gradient':
        fit_trace = np.clip(np.append(0, np.diff(trace)), 0, None)
    elif kind == 'events':
        # Baden et al 2016
        diff_trace = np.append(0, np.diff(trace))
        robust_std = np.median(np.abs(diff_trace)) / 0.6745
        peaks, _ = find_peaks(diff_trace, height=robust_std)
        fit_trace = np.zeros(diff_trace.size)
        fit_trace[peaks] = np.clip(diff_trace[peaks], 0, None)
    else:
        raise NotImplementedError(kind)

    return fit_trace


def align_data(stim, stimtime, trace, tracetime):
    """Align stimulus and trace.
     Modified from RFEst"""
    dt = np.mean(np.diff(tracetime))

    valid_idxs = (tracetime >= stimtime[0]) & (tracetime <= (stimtime[-1] + np.max(np.diff(stimtime))))
    r_tracetime, aligned_trace = resample_trace(tracetime=tracetime[valid_idxs], trace=trace[valid_idxs], dt=dt)
    t0 = r_tracetime[0]

    num_repeats = np.array([np.sum((r_tracetime > t_a) & (r_tracetime <= t_b))
                            for t_a, t_b in
                            zip(stimtime, np.append(stimtime[1:], stimtime[-1] + np.max(np.diff(stimtime))))])
    aligned_stim = np.repeat(stim, num_repeats, axis=0)

    return aligned_stim, aligned_trace, dt, t0


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


def get_sets(stim, stimtime, trace, tracetime, frac_train=1., frac_dev=0.,
             fupsample=None, norm_stim=False, norm_trace=False):
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

    if fupsample is None:
        fupsample = 1

    if (dt_max_diff / dt) > 0.1 or fupsample > 1:  # No large difference between dts
        if fupsample <= 1:
            warnings.warn('Inconsistent step-sizes in trace, resample trace.')
        tracetime, trace = resample_trace(tracetime=tracetime, trace=trace, dt=dt / fupsample)

    stim, trace, dt, t0 = align_data(stim=stim, stimtime=stimtime, trace=trace, tracetime=tracetime)

    if norm_stim:
        stim = math_utils.normalize_zscore(stim)

    if norm_trace:
        trace = math_utils.normalize_zscore(trace)

    x_dict, y_dict = split_data(stim, trace, frac_train=frac_train, frac_dev=frac_dev, as_dict=True)
    return x_dict, y_dict, dt


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
