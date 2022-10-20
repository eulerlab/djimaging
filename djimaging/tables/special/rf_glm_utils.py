import itertools
import time

import numpy as np
from scipy.stats import ttest_1samp
from sklearn.model_selection import KFold

from djimaging.tables.optional import sta_utils
from djimaging.utils import math_utils

# TODO: remove redundancy between logger and prints

try:
    import rfest
except ImportError:
    import warnings
    warnings.warn('Did not find RFEst package. Install it to estimates RFs.')
    rfest = None

ATOL = 1e-3
ALPHA = 1.


def compute_glm_receptive_field(
        stim, triggertime, trace, tracetime, dur_tfilter, df_ts, df_ws, betas, alpha, metric, tolerance,
        step_size, max_iters, min_iters, kfold, norm_stim=True, norm_trace=True, filter_trace=False, cutoff=10.,
        p_keep=1., gradient=False, output_nonlinearity='none', fupsample=0, init_method=None,
        n_perm=20, min_cc=0.2, seed=42, verbose=0, logger=None, fit_R=False, fit_intercept=True):

    """Estimate RF for given data. Return RF and quality."""
    np.random.seed(seed)

    starttime = time.time()

    # Split data
    X_dev, y_dev = None, None

    if kfold == 0:
        assert len(betas) == 1
        X_train, y_train, X_test, y_test, _, _, dt = get_sets(
            stim, triggertime, trace, tracetime, frac_train=0.8, frac_dev=0.2,
            p_keep=p_keep, gradient=gradient, fupsample=fupsample,
            norm_stim=norm_stim, norm_trace=norm_trace, filter_trace=filter_trace, cutoff=cutoff)
    elif kfold == 1:
        X_train, y_train, X_dev, y_dev, X_test, y_test, dt = get_sets(
            stim, triggertime, trace, tracetime, frac_train=0.6, frac_dev=0.2,
            p_keep=p_keep, gradient=gradient, fupsample=fupsample,
            norm_stim=norm_stim, norm_trace=norm_trace, filter_trace=filter_trace, cutoff=cutoff)
    else:
        X_train, y_train, X_test, y_test, _, _, dt = get_sets(
            stim, triggertime, trace, tracetime, frac_train=0.8, frac_dev=0.2,
            p_keep=p_keep, gradient=gradient, fupsample=fupsample,
            norm_stim=norm_stim, norm_trace=norm_trace, filter_trace=filter_trace, cutoff=cutoff)

    model, metric_dev_opt_hp_sets = compute_best_rf_model(
        X_train, y_train, X_dev=X_dev, y_dev=y_dev,
        kfold=kfold, step_size=step_size, max_iters=max_iters, min_iters=min_iters,
        dt=dt, dur_tfilter=dur_tfilter, df_ts=df_ts, df_ws=df_ws, init_method=init_method,
        betas=betas, verbose=verbose, logger=logger, output_nonlinearity=output_nonlinearity,
        fit_R=fit_R, fit_intercept=fit_intercept, alpha=alpha, metric=metric, tolerance=tolerance,
    )

    quality_dict = dict(metric=metric)

    if (n_perm is not None) and (n_perm > 0):
        assert y_test.size > 1, y_test
        score_trueX, score_permX = rfest.check.compute_permutation_test(
            model, X_test, y_test, n_perm=n_perm, metric=metric, history=False)
        p_value = quality_test(score_trueX=score_trueX, score_permX=score_permX, metric=metric)
        quality_dict = dict(score_test=float(score_trueX), score_permX=score_permX, perm_test_success=p_value < 0.001)

    y_pred_train = model.predict({'stimulus': X_train})

    for metric_ in ['corrcoef', 'mse']:
        quality_dict[f'{metric_}_train'] = model.compute_score(
            y=y_train[model.burn_in:], y_pred=y_pred_train, metric=metric_)

        if kfold == 1:
            y_pred_dev = model.predict({'stimulus': X_dev})
            quality_dict[f'{metric_}_dev'] = model.compute_score(
                y=y_dev[model.burn_in:], y_pred=y_pred_dev, metric=metric_)

        y_test_pred = model.predict({'stimulus': X_test})
        quality_dict[f'{metric_}_test'] = model.compute_score(
            y=y_test[model.burn_in:], y_pred=y_test_pred, metric=metric_)

    w = rfest.utils.uvec(model.w['opt']['stimulus'])
    if w.shape[1] != 1:
        raise NotImplementedError('Not implemented for more than one subunit')

    w = np.array(w.reshape(model.dims['stimulus']))

    model_dict = dict(
        df=model.df, dims=model.dims, shift=model.shift, alpha=model.alpha, beta=model.beta, metric=model.metric,
        intercept={k: float(v) for k, v in model.get_intercept(model.p['opt']).items()},
        R={k: float(v) for k, v in model.get_R(model.p['opt']).items()},
        return_model=model.return_model,
        metric_train=np.array(model.metric_train).astype(np.float32),
        cost_train=np.array(model.cost_train).astype(np.float32),
        metric_dev_opt=float(model.metric_dev_opt),
        best_iteration=int(model.best_iteration),
        train_stop=model.train_stop,
        p_keep=p_keep, dt=dt,
        y_test=np.array(y_test[model.burn_in:]).astype(np.float32),
        y_pred=np.array(y_test_pred).astype(np.float32),
        y_train=np.array(y_train[model.burn_in:]).astype(np.float32),
        y_pred_train=np.array(y_pred_train).astype(np.float32),
    )

    if np.any(np.isfinite(model.metric_dev)):
        model_dict['metric_dev'] = np.array(model.metric_dev).astype(np.float32)

    if np.any(np.isfinite(model.cost_dev)):
        model_dict['cost_dev'] = np.array(model.cost_dev).astype(np.float32)

    if kfold == 1:
        model_dict['y_dev'] = np.array(y_dev[model.burn_in:]).astype(np.float32)
        model_dict['y_pred_dev'] = np.array(y_pred_dev).astype(np.float32)

    model_dict['df_ts'] = df_ts
    model_dict['df_ws'] = df_ws
    model_dict['betas'] = betas
    model_dict['metrics_dev_opt'] = metric_dev_opt_hp_sets

    quality_dict['quality'] = \
        quality_dict.get('corrcoef_train', np.inf) > min_cc and \
        quality_dict.get('corrcoef_dev', np.inf) > min_cc and \
        quality_dict.get('corrcoef_test', np.inf) > min_cc and \
        quality_dict.get('perm_test_success', True)

    if logger is not None:
        logger.info(f' Total Time Elapsed: {time.time() - starttime:.0f} sec')
    elif verbose > 0:
        print(f'Total Time Elapsed: {time.time() - starttime:.0f} sec\n')

    return w, quality_dict, model_dict


def compute_best_rf_model(
        X_train, y_train, dt, dur_tfilter, betas, df_ts, df_ws, alpha, metric, tolerance,
        step_size, max_iters, min_iters, kfold,
        X_dev=None, y_dev=None, verbose=0, t_burn=5, init_method=None,
        logger=None, output_nonlinearity='none', fit_R=False, fit_intercept=True):
    """Compute the best RF model using K-Fold cross-validation."""

    df_ts = np.sort(np.asarray(df_ts))
    df_ws = np.sort(np.asarray(df_ws))
    betas = np.sort(np.asarray(betas))

    tfilterdims = int(np.round(dur_tfilter / dt)) + 1

    assert (((tfilterdims - 1) * dt) - dur_tfilter) < (dt / 10.), 'Choose a (approximate) multiple of df as dur_tfilter'

    wdims = X_train.shape[2]
    hdims = X_train.shape[1]

    df_ts = clip_dfs(df_ts, tfilterdims, logger)
    df_ws = clip_dfs(df_ws, wdims, logger)

    if (logger is not None) and (betas.size > 1):
        logger.info()
    elif verbose > 0:
        print(f"################## Optimizing dfs ##################\n" +\
              f"\tTrying {df_ts.size * df_ws.size} combination: df_ts={df_ts} and df_ws={df_ws}")

    n_burn = np.maximum(int(t_burn / dt), int(np.ceil(dur_tfilter / dt)))

    def dfh_from_dfw(_df_w):
        return np.minimum(int(np.round(_df_w * hdims / wdims)), hdims)

    # Folds
    if kfold == 0:
        splits = [(X_train, y_train, None, None)]
    elif kfold == 1:
        splits = [(X_train, y_train, X_dev, y_dev)]
    else:
        assert X_dev is None
        assert y_dev is None
        kf = KFold(n_splits=kfold)

        splits = []
        for train_idx, dev_idx in kf.split(X_train, y_train):
            train_idx = train_idx[(train_idx < np.min(dev_idx)) | (train_idx > np.max(dev_idx) + n_burn)]
            splits += [(X_train[train_idx], y_train[train_idx], X_train[dev_idx], y_train[dev_idx])]

    # HP space
    dfs = [(df_t, dfh_from_dfw(df_w), df_w) for df_t, df_w in list(itertools.product(df_ts, df_ws))]
    metrics_dev_opt = np.zeros((np.maximum(kfold, 1), len(dfs), len(betas)))

    best_model = None
    metric_dev_opt_hp_sets = None

    for idx_dfs, df in enumerate(dfs):
        if logger is not None:
            logger.info(f' Optimize for df={df}')
        elif verbose > 0:
            print(f'############ Optimize for df={df} ############')

        for idx_kf, (X_train_k, y_train_k, X_dev_k, y_dev_k) in enumerate(splits):
            if logger is not None and kfold > 1:
                logger.info(f"\tFold: {idx_kf + 1}/{kfold}")
            elif verbose > 0:
                print(f"###### Fold: {idx_kf + 1}/{kfold} ######")

            best_model, metric_dev_opt_hp_sets = create_and_fit_glm(
                X_train=X_train_k, y_train=y_train_k, X_dev=X_dev_k, y_dev=y_dev_k, init_method=init_method,
                dt=dt, alphas=[alpha], betas=betas, df=df, tfilterdims=tfilterdims, verbose=verbose,
                step_size=step_size, max_iters=max_iters, min_iters=min_iters,
                fit_R=fit_R, fit_intercept=fit_intercept, logger=logger,
                early_stopping=False, num_subunits=1, output_nonlinearity=output_nonlinearity,
                metric=metric, tolerance=tolerance)

            metrics_dev_opt[idx_kf, idx_dfs, :] = metric_dev_opt_hp_sets

    if kfold > 0:
        assert np.all(np.isfinite(metrics_dev_opt)), metrics_dev_opt

        # Get best model
        mean_metrics_dev_opt = np.mean(metrics_dev_opt, axis=0)

        if metric == 'corrcoef':
            best_df_idx, best_beta_idx = \
                np.unravel_index(mean_metrics_dev_opt.argmax(), mean_metrics_dev_opt.shape)
        else:
            best_df_idx, best_beta_idx = \
                np.unravel_index(mean_metrics_dev_opt.argmin(), mean_metrics_dev_opt.shape)
    else:
        assert metrics_dev_opt.size == 1
        best_df_idx = 0
        best_beta_idx = 0

    best_df = dfs[best_df_idx]
    best_beta = betas[best_beta_idx]

    if logger is not None:
        logger.info(f' ---OPTIMIZE on all data with best HPs---\n' +
                    f' df={best_df} and beta={best_beta:.4g}')
    elif verbose > 0:
        print(f'###### Optimize on all data with best HPs ######\n' +
              f'\tdf={best_df} and beta={best_beta:.4g}')

    if kfold >= 1:
        best_model, _ = create_and_fit_glm(
            X_train=X_train, y_train=y_train, X_dev=None, y_dev=None,  init_method=init_method,
            dt=dt, alphas=[alpha], betas=[best_beta], df=best_df, tfilterdims=tfilterdims, verbose=verbose,
            step_size=step_size, max_iters=max_iters, min_iters=min_iters, logger=logger,
            fit_R=fit_R, fit_intercept=fit_intercept,
            early_stopping=False, num_subunits=1, output_nonlinearity=output_nonlinearity,
            metric=metric, tolerance=tolerance)
        best_model.metric_dev_opt = np.mean(metrics_dev_opt[:, best_df_idx, best_beta_idx])

    if logger is not None:
        logger.info(
            f' ###### Finished HP optimization ######' +
            f' {metric}={np.mean(metrics_dev_opt[:, best_df_idx, best_beta_idx]):.3f}'
            f'+-{np.std(metrics_dev_opt[:, best_df_idx, best_beta_idx]) / np.sqrt(kfold):.3f}')
    elif verbose > 0:
        print(
            f'################## Finished HP optimization ##################\n' +
            f'\t{metric}={np.mean(metrics_dev_opt[:, best_df_idx, best_beta_idx]):.3f}'
            f'\t+-{np.std(metrics_dev_opt[:, best_df_idx, best_beta_idx]) / np.sqrt(kfold):.3f}\n')

    return best_model, metric_dev_opt_hp_sets


def clip_dfs(dfs, dims, logger):
    if np.all(dfs <= dims):
        return dfs
    elif np.any(dfs <= dims):
        if logger is not None:
            logger.warning(' No using df_ts larger than dims')
        return dfs[dfs <= dims]
    else:
        if logger is not None:
            logger.warning(' All df_ts larger than dims, use only dims as df')
        return np.array([dims])


def _df_w2df_h(df_w, X_train, hdims):
    df_h = np.minimum(int(np.round(df_w * X_train.shape[1] / X_train.shape[2])), hdims)
    return df_h


def create_and_fit_glm(
        X_train, y_train, X_dev, y_dev, dt, tfilterdims, df, alphas, betas,
        num_subunits, output_nonlinearity, init_method, fit_R, fit_intercept, early_stopping, metric,
        tolerance, step_size=None, max_iters=None, min_iters=None, verbose=0, logger=None):
    """Fit GLM model"""

    model = create_glm(
        tfilterdims=tfilterdims, df=df, X_train=X_train, y_train=y_train, X_dev=X_dev, dt=dt,
        num_subunits=num_subunits, output_nonlinearity=output_nonlinearity, init_method=init_method,
        fit_R=fit_R, fit_intercept=fit_intercept, logger=logger)

    model, metric_dev_opt_hp_sets = fit_glm(
        model=model, y_train=y_train, X_dev=X_dev, y_dev=y_dev, alphas=alphas, betas=betas,
        step_size=step_size, max_iters=max_iters, min_iters=min_iters, early_stopping=early_stopping,
        verbose=verbose, logger=logger, metric=metric, tolerance=tolerance)

    assert model.fit_R == fit_R
    assert model.fit_intercept == fit_intercept

    return model, metric_dev_opt_hp_sets


def create_glm(
        tfilterdims, df, X_train, y_train, X_dev, dt, fit_R, fit_intercept,
        num_subunits=1, output_nonlinearity='none', init_method=None, logger=None,
):
    assert tfilterdims > 0

    if X_train.ndim == 3:
        dims = (tfilterdims, X_train.shape[1], X_train.shape[2])
    elif X_train.ndim == 1:
        dims = (tfilterdims,)
    else:
        raise NotImplementedError()

    if len(dims) != len(df):
        raise ValueError(f"size mismatch {len(dims)} != {len(df)}")

    # Never use more dfs than necessary:
    df = np.asarray(df)
    for i, (dim, dfi) in enumerate(zip(dims, df)):
        if dfi > dim:
            if logger is not None:
                logger.info(f' More degrees of freedom than necessary, dims={dims}, df={df}. Set to max.')
            df[i] = dim
    df = tuple(df)

    # Create model
    model = initialize_model(
        dims=dims, df=df, output_nonlinearity=output_nonlinearity,
        num_subunits=num_subunits, dt=dt, X_train=X_train, y_train=y_train, X_dev=X_dev, init_method=init_method,
        fit_R=fit_R, fit_intercept=fit_intercept,
    )

    return model


def fit_glm(
        model, y_train, X_dev, y_dev, alphas, betas,
        step_size, max_iters, min_iters, tolerance, metric,
        early_stopping=False, verbose=0, logger=None):
    """Fit GLM model"""
    min_iters_other = min_iters // 5

    # Fit
    if logger is not None:
        logger.debug(
            f' \tFit GLM: step_size={step_size}, tol={tolerance}, iters=[{min_iters}, {max_iters}]')

    if len(alphas) == 1 and len(betas) == 1:
        y = {'train': y_train, 'dev': y_dev} if early_stopping else {'train': y_train}
        return_model = 'best_dev_metric' if early_stopping else 'best_train_cost'

        model.fit(
            y=y, num_iters=max_iters, verbose=verbose, tolerance=tolerance, metric=metric,
            step_size=step_size, alpha=alphas[0], beta=betas[0], min_iters=min_iters,
            return_model=return_model, atol=ATOL)

        if not early_stopping and y_dev is not None:
            model.metric_dev_opt = model.score({'stimulus': X_dev}, y_test=y_dev, metric=metric)
        metric_dev_opt_hp_sets = [model.metric_dev_opt]
    else:
        metric_dev_opt_hp_sets = model.fit_hps(
            y={'train': y_train, 'dev': y_dev},
            num_iters=max_iters, verbose=verbose, tolerance=tolerance, metric=metric,
            step_size=step_size, alphas=alphas, betas=betas, min_iters=min_iters,
            min_iters_other=min_iters_other, atol=ATOL)

    return model, metric_dev_opt_hp_sets


def initialize_model(
        dims, df, output_nonlinearity, num_subunits, dt, X_train, y_train, fit_R, fit_intercept,
        X_dev=None, init_method=None):
    """Initialize model parameters for faster optimization"""

    model = rfest.GLM(distr='gaussian', output_nonlinearity=output_nonlinearity)
    model.add_design_matrix(X_train, dims=dims, df=df, smooth='cr', filter_nonlinearity='none', name='stimulus')

    if X_dev is not None:
        model.add_design_matrix(X_dev, dims=dims, name='stimulus', kind='dev')

    model.initialize(num_subunits=num_subunits, dt=dt,
                     method=init_method if init_method is not None else 'mle',
                     compute_ci=False, random_seed=42, y=y_train,
                     fit_R=fit_R, fit_intercept=fit_intercept)

    return model


def get_b_from_w(S, w):
    b, *_ = np.linalg.lstsq(S, w, rcond=None)
    return b


def get_sets(stim, triggertime, trace, tracetime, frac_train, frac_dev, p_keep=1.0, fupsample=0, gradient=False,
             norm_stim=True, norm_trace=True, filter_trace=False, cutoff=10.):
    """Split data into sets"""

    assert triggertime.ndim == 1, triggertime.shape
    assert trace.ndim == 1, trace.shape
    assert tracetime.ndim == 1, tracetime.shape
    assert stim.shape[0] == triggertime.size
    assert trace.size == tracetime.size

    dts = np.diff(tracetime)
    assert np.mean(dts) > 10 * np.std(
        dts), f'Highly variable sampling rate, std={np.std(dts):g} vs mu={np.mean(dts):.4g}'

    if fupsample > 1:
        tracetime, trace = upsample_trace(tracetime=tracetime, trace=trace, fupsample=fupsample)

    # Make stimulus have same time as trace
    X, y, dt = rfest.utils.upsample_data(stim, triggertime, trace, tracetime, gradient=gradient, threshold=gradient)

    if norm_stim:
        X = math_utils.normalize_zscore(X)

    if filter_trace:
        y = sta_utils.filter_trace(y, dt=dt, cutoff=cutoff)

    if norm_trace:
        y = math_utils.normalize_zscore(y)

    if p_keep < 1.0:
        assert X.shape[0] == y.shape[0]
        X = X[:int(X.shape[0] * p_keep)]
        y = y[:int(y.shape[0] * p_keep)]

    (X_train, y_train), (X_dev, y_dev), (X_test, y_test) = rfest.utils.split_data(
        X, y, dt, verbose=False, frac_train=frac_train, frac_dev=frac_dev)  # Changed order

    return X_train, y_train, X_dev, y_dev, X_test, y_test, dt


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


def _get_model_info(model):
    info = f"{model.metric}={model.metric_dev_opt:.4f} "
    info += f"for HP-set: alpha={model.alpha:.5g}, beta={model.beta:.5g}, df={model.df}, dims={model.dims}"
    return info


def quality_test(score_trueX, score_permX, metric):
    if metric == 'corrcoef':
        alternative = 'less'
    elif metric == 'mse':
        alternative = 'greater'
    else:
        alternative = 'two-sided'

    _, p_value = ttest_1samp(score_permX, score_trueX, nan_policy='raise', alternative=alternative)
    return p_value