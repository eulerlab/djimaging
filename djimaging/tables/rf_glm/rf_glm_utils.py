import itertools
import time
import warnings

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import ttest_1samp
from sklearn.model_selection import KFold

from djimaging.tables.receptivefield.rf_utils import split_strf, split_data, build_design_matrix, get_rf_timing_params

try:
    import rfest
    import jax

    jax.config.update('jax_platform_name', 'cpu')
except ImportError:
    warnings.warn('Failed to import RFEst: Cannot compute receptive fields.')
    rfest = None
    jax = None


class ReceptiveFieldGLM:
    def __init__(self, dt, stim, trace, filter_dur_s_past, filter_dur_s_future, df_ts, df_ws, shift, betas, metric,
                 output_nonlinearity='none', alphas=(1.,), kfold=1,
                 tolerance=5, atol=1e-5, step_size=0.01, max_iters=2000, min_iters=200,
                 init_method=None, n_perm=20, min_cc=0.2, seed=42, verbose=0, fit_R=False, fit_intercept=True,
                 num_subunits=1, distr='gaussian', frac_test=0.2, t_burn_min=0.):
        # Data
        self.dt = dt
        self.stim = stim
        self.trace = trace

        # Filter settings
        self.filter_dur_s_past = filter_dur_s_past
        self.filter_dur_s_future = filter_dur_s_future
        self.shift = shift
        self.output_nonlinearity = output_nonlinearity
        assert num_subunits == 1
        self.num_subunits = num_subunits

        # Loss and CV
        self.metric = metric
        self.betas = np.sort(np.asarray(betas))
        self.alphas = np.sort(np.asarray(alphas))
        self.kfold = kfold
        self.distr = distr
        self.fit_intercept = fit_intercept
        self.fit_R = fit_R

        # Optimizer settings
        self.init_method = init_method if init_method is not None else 'mle'
        self.tolerance = tolerance
        self.atol = atol
        self.step_size = step_size
        self.max_iters = max_iters
        self.min_iters = min_iters
        self.verbose = verbose
        self.seed = seed

        # Evaluation parameters
        self.frac_test = frac_test
        self.n_perm = n_perm
        self.min_cc = min_cc

        # Filter dims
        self.rf_time, self.dim_t, self.shift, burn_in = get_rf_timing_params(
            filter_dur_s_past, filter_dur_s_future, dt)

        self.burn_in = np.maximum(int(np.ceil(t_burn_min / dt)), burn_in)

        if stim.ndim == 3:
            self.hdims = stim.shape[1]
            self.wdims = stim.shape[2]
            self.dims = (self.dim_t, self.hdims, self.wdims)
        elif stim.ndim == 1:
            self.dims = (self.dim_t,)
        else:
            raise NotImplementedError()

        self.df_ts = self.clip_dfs(np.sort(np.asarray(df_ts)), self.dim_t)
        if stim.ndim == 3:
            self.df_ws = self.clip_dfs(np.sort(np.asarray(df_ws)), self.wdims)
        else:
            assert not df_ws

    def compute_glm_receptive_field(self):
        """Estimate RF for given data. Return RF and quality."""
        np.random.seed(self.seed)
        starttime = time.time()

        stim_dict, trace_dict = split_data(
            x=self.stim, y=self.trace, frac_train=1. - self.frac_test, frac_dev=0, as_dict=True)

        model, metric_dev_opt_hp_sets = self.compute_best_rf_model(stim=stim_dict['train'], trace=trace_dict['train'])

        quality_dict = dict(metric=self.metric)

        if self.verbose > 0:
            print("############ Evaluate model performance ############")

        if (self.n_perm is not None) and (self.n_perm > 0):
            assert trace_dict['test'].size > 1, trace_dict['test']
            score_trueX, score_permX = rfest.check.compute_permutation_test(
                model, stim_dict['test'], trace_dict['test'], n_perm=self.n_perm, metric=self.metric, history=False)
            p_value = quality_test(
                score_trueX=score_trueX, score_permX=score_permX, metric=self.metric)
            quality_dict = dict(
                score_test=float(score_trueX), score_permX=score_permX, perm_test_success=p_value < 0.001)

        y_pred_train = model.forwardpass(kind='train', p=model.p['opt'])

        for metric_ in ['corrcoef', 'mse']:
            quality_dict[f'{metric_}_train'] = model.compute_score(
                y=trace_dict['train'][model.burn_in:], y_pred=y_pred_train, metric=metric_)

            y_test_pred = model.predict({'stimulus': stim_dict['test']})
            quality_dict[f'{metric_}_test'] = model.compute_score(
                y=trace_dict['test'][model.burn_in:], y_pred=y_test_pred, metric=metric_)

        w = rfest.utils.uvec(model.w['opt']['stimulus'])
        if w.shape[1] != 1:
            raise NotImplementedError('Not implemented for more than one subunit')

        w = np.array(w.reshape(model.dims['stimulus']))

        model_dict = dict(
            rf_time=self.rf_time, dt=self.dt,
            df=model.df, dims=model.dims, shift=model.shift,
            alpha=model.alpha, beta=model.beta, metric=model.metric,
            output_nonlinearity=model.output_nonlinearity,
            intercept={k: float(v) for k, v in model.get_intercept(model.p['opt']).items()},
            R={k: float(v) for k, v in model.get_R(model.p['opt']).items()},
            return_model=model.return_model,
            metric_train=np.array(model.metric_train).astype(np.float32),
            cost_train=np.array(model.cost_train).astype(np.float32),
            metric_dev_opt=float(model.metric_dev_opt),
            best_iteration=int(model.best_iteration),
            train_stop=model.train_stop,
            y_test=np.array(trace_dict['test'][model.burn_in:]).astype(np.float32),
            y_pred_test=np.array(y_test_pred).astype(np.float32),
            y_train=np.array(trace_dict['train'][model.burn_in:]).astype(np.float32),
            y_pred_train=np.array(y_pred_train).astype(np.float32),
        )

        if np.any(np.isfinite(model.metric_dev)):
            model_dict['metric_dev'] = np.array(model.metric_dev).astype(np.float32)

        if np.any(np.isfinite(model.cost_dev)):
            model_dict['cost_dev'] = np.array(model.cost_dev).astype(np.float32)

        model_dict['df_ts'] = self.df_ts
        model_dict['df_ws'] = self.df_ws
        model_dict['betas'] = self.betas
        model_dict['metrics_dev_opt'] = metric_dev_opt_hp_sets

        quality_dict['quality'] = \
            quality_dict.get('corrcoef_train', np.inf) > self.min_cc and \
            quality_dict.get('corrcoef_dev', np.inf) > self.min_cc and \
            quality_dict.get('corrcoef_test', np.inf) > self.min_cc and \
            quality_dict.get('perm_test_success', True)

        if self.verbose > 0:
            print(f'Total Time Elapsed: {time.time() - starttime:.0f} sec\n')

        return w, quality_dict, model_dict

    def dfh_from_dfw(self, df_w):
        return np.minimum(int(np.round(df_w * self.hdims / self.wdims)), self.hdims)

    def compute_best_rf_model(self, trace, stim):
        """Compute the best RF model using K-Fold cross-validation."""

        if self.verbose > 0 and (self.betas.size > 1):
            print(f"################## Optimizing dfs ##################\n" + \
                  f"\tTrying {self.df_ts.size * self.df_ws.size} combination:" + \
                  f" df_ts={self.df_ts} and df_ws={self.df_ws}")

        # Define HP grid
        dfs = [(df_t, self.dfh_from_dfw(df_w), df_w) for df_t, df_w in list(itertools.product(self.df_ts, self.df_ws))]
        metrics_dev_opt = np.zeros((np.maximum(self.kfold, 1), len(dfs), len(self.betas)))

        # Prepare data
        X = build_design_matrix(X=stim, n_lag=self.dims[0], shift=self.shift, dtype=np.float32)
        y = trace.copy().astype(np.float32)

        # Folds
        if self.kfold == 0:
            splits = [(X, y, None, None)]
        elif self.kfold == 1:
            split_idx = int(y.size * 0.8)
            splits = [(X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:])]
        else:
            splits = []
            for train_idx, dev_idx in KFold(n_splits=self.kfold, shuffle=False).split(X, y):
                train_idx = train_idx[(train_idx < np.min(dev_idx)) | (train_idx > np.max(dev_idx) + self.burn_in)]
                splits += [(X[train_idx], y[train_idx], X[dev_idx], y[dev_idx])]

        best_model = None
        for idx_dfs, df in enumerate(dfs):
            if self.verbose > 0:
                print(f'############ Optimize for df={df} ############')

            for idx_kf, (X_train_k, y_train_k, X_dev_k, y_dev_k) in enumerate(splits):
                if self.verbose > 0 and self.kfold > 1:
                    print(f"###### Fold: {idx_kf + 1}/{self.kfold} ######")

                best_model, metrics_dev_opt_i = self.create_and_fit_glm(
                    df=df, X_train=X_train_k, y_train=y_train_k, X_dev=X_dev_k, y_dev=y_dev_k,
                    alphas=self.alphas, betas=self.betas)

                metrics_dev_opt[idx_kf, idx_dfs, :] = metrics_dev_opt_i

        if self.kfold > 0:
            assert np.all(np.isfinite(metrics_dev_opt)), metrics_dev_opt

            # Get best model
            mean_metrics_dev_opt = np.mean(metrics_dev_opt, axis=0)

            if self.metric == 'corrcoef':
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
        best_beta = self.betas[best_beta_idx]

        if self.kfold >= 1:
            if self.verbose > 0:
                print(f'###### Optimize on all data with best HPs ######\n' +
                      f'\tdf={best_df} and beta={best_beta:.4g}')

            best_model, _ = self.create_and_fit_glm(
                df=best_df, X_train=X, y_train=y, X_dev=None, y_dev=None,
                alphas=self.alphas, betas=[best_beta])
            best_model.metric_dev_opt = np.mean(metrics_dev_opt[:, best_df_idx, best_beta_idx])

            if self.verbose > 0:
                print(
                    f'###### Finished HP optimization ######\n' +
                    f'\t{self.metric}={np.mean(metrics_dev_opt[:, best_df_idx, best_beta_idx]):.3f}'
                    f'\t+-{np.std(metrics_dev_opt[:, best_df_idx, best_beta_idx]) / np.sqrt(self.kfold):.3f}\n')

        return best_model, metrics_dev_opt

    @staticmethod
    def clip_dfs(dfs, dims):
        if np.all(dfs <= dims):
            return dfs
        elif np.any(dfs <= dims):
            warnings.warn(' No using df_ts larger than dims')
            return dfs[dfs <= dims]
        else:
            warnings.warn(' All df_ts larger than dims, use only dims as df')
            return np.array([dims])

    @staticmethod
    def _df_w2df_h(df_w, X_train, hdims):
        df_h = np.minimum(int(np.round(df_w * X_train.shape[1] / X_train.shape[2])), hdims)
        return df_h

    def create_and_fit_glm(self, df, X_train, y_train, alphas, betas, X_dev=None, y_dev=None):
        """Fit GLM model"""
        assert X_train.shape[0] == y_train.shape[0]
        if X_dev is not None:
            assert y_dev is not None
            assert X_dev.shape[0] == y_dev.shape[0]
            assert X_train.shape[1:] == X_dev.shape[1:]

        model = self.create_glm(
            df=df, X_train=X_train, y_train=y_train, X_dev=X_dev)

        model, metric_dev_opt_hp_sets = self.fit_glm(
            model=model, y_train=y_train, y_dev=y_dev, alphas=alphas, betas=betas)

        return model, metric_dev_opt_hp_sets

    def create_glm(self, df, X_train, y_train, X_dev):

        if len(self.dims) != len(df):
            raise ValueError(f"size mismatch {len(self.dims)} != {len(df)}")

        # Never use more dfs than necessary:
        df = np.asarray(df)
        for i, (dim, dfi) in enumerate(zip(self.dims, df)):
            if dfi > dim:
                warnings.warn(f' More degrees of freedom than necessary, dims={self.dims}, df={df}. Set to max.')
                df[i] = dim

        df = tuple(df)
        model = self.initialize_model(df=df, X_train=X_train, y_train=y_train, X_dev=X_dev)
        return model

    def fit_glm(self, model, alphas, betas, y_train, y_dev=None):
        """Fit GLM model"""
        min_iters_other = self.min_iters // 5

        if len(alphas) == 1 and len(betas) == 1:

            if y_dev is not None:
                assert 'dev' in model.X
                y = {'train': y_train, 'dev': y_dev}
                return_model = 'best_dev_cost'
                early_stopping = True
            else:
                y = {'train': y_train}
                return_model = 'best_train_cost'
                early_stopping = False

            if self.verbose > 0:
                print(return_model)

            model.fit(
                y=y, metric=self.metric, num_iters=self.max_iters, min_iters=self.min_iters, step_size=self.step_size,
                atol=self.atol, tolerance=self.tolerance, alpha=alphas[0], beta=betas[0],
                verbose=self.verbose, return_model=return_model, early_stopping=early_stopping)

            metric_dev_opt_hp_sets = [model.metric_dev_opt]
        else:
            metric_dev_opt_hp_sets = model.fit_hps(
                y={'train': y_train, 'dev': y_dev},
                num_iters=self.max_iters, verbose=self.verbose, tolerance=self.tolerance, metric=self.metric,
                step_size=self.step_size, alphas=alphas, betas=betas, min_iters=self.min_iters,
                min_iters_other=min_iters_other, atol=self.atol, return_model='best_train_cost',
                early_stopping=False)

        return model, metric_dev_opt_hp_sets

    def initialize_model(self, df, X_train, y_train, X_dev=None):
        """Initialize model parameters for faster optimization"""

        model = rfest.GLM(distr=self.distr, output_nonlinearity=self.output_nonlinearity)
        model.burn_in = self.burn_in
        model.add_design_matrix(X_train, is_design_matrix=True, df=df, dims=self.dims, shift=self.shift,
                                smooth='cr', filter_nonlinearity='none', name='stimulus')

        if X_dev is not None:
            model.add_design_matrix(X_dev, is_design_matrix=True, dims=self.dims, shift=self.shift,
                                    name='stimulus', kind='dev')

        model.initialize(num_subunits=self.num_subunits, dt=self.dt,
                         method=self.init_method,
                         compute_ci=False, random_seed=42, y=y_train,
                         fit_R=self.fit_R, fit_intercept=self.fit_intercept)

        return model

    @staticmethod
    def get_b_from_w(S, w):
        b, *_ = np.linalg.lstsq(S, w, rcond=None)
        return b

    @staticmethod
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


def plot_rf_summary(rf, quality_dict, model_dict, title=""):
    """Plot tRF and sRFs and quality test"""

    # Create figure and axes
    n_rows = 5
    fig, axs = plt.subplots(n_rows, 4, figsize=(8, n_rows * 1.9), gridspec_kw=dict(height_ratios=(1.0, 1.5, 1, 1, 1)))

    axs_big = []
    for i in np.arange(2, n_rows):
        for ax in axs[i, :].flat:
            ax.remove()
        ax_big = fig.add_subplot(axs[0, 0].get_gridspec()[i, :])
        axs_big.append(ax_big)
    axs = np.append(axs.flatten(), np.array(axs_big))

    # Summarize RF
    if rf.ndim == 3:
        srf, trf = split_strf(rf)
        vabsmax = np.max([np.max(np.abs(srf)), np.max(np.abs(rf))])
    elif rf.ndim == 1:
        trf = rf.flatten()
        srf = None
        vabsmax = 1
    else:
        raise NotImplementedError(rf.shape)

    assert trf.size == rf.shape[0]

    # Plot
    ax = axs[0]
    ax.set(title='tRF')
    try:
        dt = model_dict['dt']
        shift = model_dict['shift']['stimulus']
        t_tRF = (np.arange(-trf.size + 1, 0.1, 1) - shift).astype(float) * dt
        ax.set_xlabel('Time')
    except KeyError:
        t_tRF = np.arange(-trf.size + 1, 0.1, 1)
        ax.set_xlabel('Frames')
    ax.fill_between(t_tRF, np.zeros_like(trf), trf)
    ax.set_ylim(-np.max(np.abs(trf)) * 1.1, np.max(np.abs(trf)) * 1.1)
    for line in [t_tRF[np.argmin(trf)], t_tRF[np.argmax(trf)]]:
        ax.axvline(line, color='red')

    if srf is not None:
        ax = axs[1]
        ax.set(title='sRF')
        im = ax.imshow(srf.T, cmap='bwr', vmin=-vabsmax, vmax=vabsmax, origin='lower')
        plt.colorbar(im, ax=ax)

        ax = axs[2]
        ax.set(title='pos. peak frame')
        im = ax.imshow(rf[np.argmax(trf)].T, cmap='bwr', vmin=-vabsmax, vmax=vabsmax, origin='lower')
        plt.colorbar(im, ax=ax)

        ax = axs[3]
        ax.set(title='neg. peak frame')
        im = ax.imshow(rf[np.argmin(trf)].T, cmap='bwr', vmin=-vabsmax, vmax=vabsmax, origin='lower')
        plt.colorbar(im, ax=ax)

    if not model_dict:
        return

    if model_dict is not None:
        ax = axs[4]
        ax.set(title=f"Metrics\n{model_dict['return_model']}", xlabel='Iteration', ylabel='Corrcoef')
        ax.plot(model_dict['metric_train'], label='train', c='blue')
        ax.axvline(model_dict['best_iteration'], ls='-', c='orange')
        if model_dict.get('metric_dev', False):
            ax.plot(model_dict.get('metric_dev', np.zeros(0)), label='dev', c='red')
        if model_dict.get('metric_dev_opt', False):
            ax.axhline(model_dict['metric_dev_opt'], label='dev-opt', ls='--', c='red')
        ax.legend()

        ax = axs[5]
        ax.set(title=f"Cost\n{model_dict['return_model']}", xlabel='Iteration', ylabel='Cost')
        ax.plot(model_dict['cost_train'], label='train', c='blue')
        ax.axvline(model_dict['best_iteration'], ls='-', c='orange')
        if model_dict.get('cost_dev', False):
            ax.plot(model_dict.get['cost_dev'], label='dev', c='red', marker='.')

        ax = axs[6]
        ax.set(title=f"loglog-Cost\n{model_dict['return_model']}", xlabel='Iteration', ylabel='Cost')
        ax.loglog(np.arange(1, model_dict['cost_train'].size + 1), model_dict['cost_train'], label='train', c='blue')
        ax.axvline(1 + model_dict['best_iteration'], ls='-', c='orange')
        if model_dict.get('cost_dev', False):
            ax.loglog(np.arange(1, model_dict['cost_dev'].size + 1), model_dict['cost_dev'], 'r.', label='dev')

    if (quality_dict is not None) and ('score_permX' in quality_dict):
        plot_test_rf_quality(
            ax=axs[7], score_permX=quality_dict['score_permX'],
            score_test=quality_dict.get('score_test', np.nan),
            cc_test=quality_dict.get('corrcoef_test', None),
            cc_dev=quality_dict.get('corrcoef_dev', None),
        )
        title += f"\nquality={quality_dict['quality']}"
        title_kw = dict(backgroundcolor='green' if quality_dict['quality'] else 'red')
    else:
        title_kw = dict()
        axs[7].axis('off')

    for i, (y_type, ax) in enumerate(zip(['train', 'dev', 'test'], axs_big)):
        y_data = model_dict.get(f'y_{y_type}', None)
        y_pred = model_dict.get(f'y_pred_{y_type}', model_dict.get(f'y_pred', None))

        cc = quality_dict.get(f'corrcoef_{y_type}', np.nan)
        mse = quality_dict.get(f'mse_{y_type}', np.nan)

        if y_data is None or y_pred is None:
            ax.set_title(y_type + ' data not found')
            continue

        y_time = np.arange(0, y_data.size) * model_dict.get('dt', 1)

        ax.plot(y_time, y_data, label='data')
        ax.plot(y_time, y_pred, label='prediction', alpha=0.7)
        ax.legend(frameon=True)
        ax.set_title(f'{y_type} data and prediction: cc={cc:.2f}, mse={mse:.2f}')

    plt.tight_layout(rect=(0.01, 0.01, 0.99, 0.90), h_pad=0.2, w_pad=0.2)
    plt.suptitle(title, y=0.99, va='top', **title_kw)

    return fig, axs


def plot_test_rf_quality(score_permX, score_test, cc_test=None, cc_dev=None, ax=None, metric=''):
    """Plot test results"""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 2))
    sns.boxplot(y=score_permX, ax=ax, width=0.8)
    sns.swarmplot(y=score_permX, ax=ax, color='k', alpha=0.5, label='perm X')
    ax.axhline(score_test, c='r', label=f'test {metric}', alpha=0.5, zorder=100)

    mu = np.mean(score_permX)
    std = np.std(score_permX)

    for i in range(0, 4):
        ax.axhline(mu + i * std, c='c', ls=':', alpha=0.5, zorder=0, label='_')

    title = ""
    if cc_dev is not None:
        title += f"cc_dev={cc_dev: .4f}"
    if cc_test is not None:
        title += f"\ncc_test={cc_test:.4f}"

    ax.set(title=title, ylabel='score')

    ax.set_xlim(-0.4, 0.4)
    ax.set_ylim(np.min([mu - 0.03, mu - 4 * std, score_test - 0.01]),
                np.max([mu + 0.03, mu + 4 * std, score_test + 0.01]))

    return ax
