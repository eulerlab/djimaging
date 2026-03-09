import itertools
import time
import warnings

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import ttest_1samp
from sklearn.model_selection import KFold

from djimaging.utils.receptive_fields.fit_rf_utils import split_data, build_design_matrix, get_rf_timing_params
from djimaging.utils.receptive_fields.split_rf_utils import split_strf

try:
    import rfest
    import jax

    jax.config.update('jax_platform_name', 'cpu')
except ImportError:
    rfest = None
    jax = None


class ReceptiveFieldGLM:
    def __init__(self, dt: float, stim: np.ndarray, trace: np.ndarray,
                 filter_dur_s_past: float, filter_dur_s_future: float,
                 df_ts, df_ws, betas, metric: str,
                 output_nonlinearity: str = 'none', alphas: tuple = (1.,),
                 kfold: int = 1, tolerance: int = 5, atol: float = 1e-5,
                 step_size: float = 0.01, step_size_finetune: float = 0.001,
                 max_iters: int = 2000, min_iters: int = 200,
                 init_method: str = None, n_perm: int = 20,
                 min_cc: float = 0.2, seed: int = 42, verbose: int = 0,
                 fit_R: bool = False, fit_intercept: bool = True,
                 num_subunits: int = 1, distr: str = 'gaussian',
                 frac_test: float = 0.2, t_burn_min: float = 0.,
                 shift: int = None):
        """Initialize a GLM-based receptive field estimator.

        Parameters
        ----------
        dt : float
            Time step in seconds.
        stim : np.ndarray
            Stimulus array, shape (n_frames,) or (n_frames, n_y, n_x).
        trace : np.ndarray
            Neural response trace, shape (n_frames,).
        filter_dur_s_past : float
            Duration of the filter extending into the past (seconds).
        filter_dur_s_future : float
            Duration of the filter extending into the future (seconds).
        df_ts : array-like
            Candidate degrees of freedom for temporal spline basis.
        df_ws : array-like
            Candidate degrees of freedom for spatial (width) spline basis.
        betas : array-like
            Candidate regularization strengths (beta values).
        metric : str
            Optimization metric, e.g. 'corrcoef' or 'mse'.
        output_nonlinearity : str, optional
            Output nonlinearity. Default is 'none'.
        alphas : tuple, optional
            Candidate alpha values for elastic-net mixing. Default is (1.,).
        kfold : int, optional
            Number of cross-validation folds. Default is 1.
        tolerance : int, optional
            Early-stopping tolerance (patience in iterations). Default is 5.
        atol : float, optional
            Absolute tolerance for convergence. Default is 1e-5.
        step_size : float, optional
            Gradient descent step size. Default is 0.01.
        step_size_finetune : float, optional
            Step size used for fine-tuning. Default is 0.001.
        max_iters : int, optional
            Maximum number of optimization iterations. Default is 2000.
        min_iters : int, optional
            Minimum number of optimization iterations. Default is 200.
        init_method : str, optional
            Initialization method for model parameters. Default is None (uses 'mle').
        n_perm : int, optional
            Number of permutations for the permutation test. Default is 20.
        min_cc : float, optional
            Minimum correlation coefficient threshold for quality. Default is 0.2.
        seed : int, optional
            Random seed. Default is 42.
        verbose : int, optional
            Verbosity level. Default is 0.
        fit_R : bool, optional
            Whether to fit the gain parameter R. Default is False.
        fit_intercept : bool, optional
            Whether to fit an intercept term. Default is True.
        num_subunits : int, optional
            Number of subunits (currently only 1 is supported). Default is 1.
        distr : str, optional
            GLM response distribution. Default is 'gaussian'.
        frac_test : float, optional
            Fraction of data reserved for testing. Default is 0.2.
        t_burn_min : float, optional
            Minimum burn-in time in seconds. Default is 0.
        shift : int, optional
            Expected temporal shift; if provided, validated against computed value.
        """
        # Data
        self.dt = dt
        self.stim = stim
        self.trace = trace

        # Filter settings
        self.filter_dur_s_past = filter_dur_s_past
        self.filter_dur_s_future = filter_dur_s_future
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
        self.step_size_finetune = step_size_finetune if step_size_finetune is not None else step_size
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

        if shift is not None:
            assert shift == self.shift

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

    def compute_glm_receptive_field(self) -> tuple[np.ndarray, dict, dict]:
        """Estimate RF for given data. Return RF and quality.

        Returns
        -------
        tuple[np.ndarray, dict, dict]
            w : np.ndarray
                Estimated receptive field.
            quality_dict : dict
                Quality metrics including correlation coefficients and permutation test results.
            model_dict : dict
                Model metadata including training curves and hyperparameters.
        """
        np.random.seed(self.seed)
        starttime = time.time()

        stim_dict, trace_dict = split_data(
            x=self.stim, y=self.trace, frac_train=1. - self.frac_test, frac_dev=0, as_dict=True)

        model, metric_dev_opt_hp_sets = self.compute_best_rf_model(stim=stim_dict['train'], trace=trace_dict['train'])

        quality_dict = dict(metric=self.metric)

        if self.verbose > 0:
            print("############ Evaluate model performance ############")

        if (self.n_perm is not None) and (self.n_perm > 0) and (self.frac_test > 0):
            assert trace_dict['test'].size > 1, trace_dict['test']
            score_trueX, score_permX = rfest.check.compute_permutation_test(
                model, stim_dict['test'], trace_dict['test'], n_perm=self.n_perm, metric=self.metric, history=False)
            p_value = quality_test(
                score_trueX=score_trueX, score_permX=score_permX, metric=self.metric)
            quality_dict = dict(
                score_test=float(score_trueX), score_permX=score_permX, perm_test_success=p_value < 0.001)

        if self.kfold != 1:
            y_pred_train = model.forwardpass(kind='train', p=model.p['opt'])
        else:
            y_pred_train = model.predict({'stimulus': stim_dict['train']})

        for metric_ in ['corrcoef', 'mse']:
            quality_dict[f'{metric_}_train'] = model.compute_score(
                y=trace_dict['train'][model.burn_in:], y_pred=y_pred_train, metric=metric_)

            if self.frac_test > 0:
                y_test_pred = model.predict({'stimulus': stim_dict['test']})
                quality_dict[f'{metric_}_test'] = model.compute_score(
                    y=trace_dict['test'][model.burn_in:], y_pred=y_test_pred, metric=metric_)
            else:
                y_test_pred = None

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
            y_train=np.array(trace_dict['train'][model.burn_in:]).astype(np.float32),
            y_pred_train=np.array(y_pred_train).astype(np.float32),
        )

        if self.frac_test > 0:
            model_dict['y_test'] = np.array(trace_dict['test'][model.burn_in:]).astype(np.float32)
            model_dict['y_pred_test'] = np.array(y_test_pred).astype(np.float32)

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

    def dfh_from_dfw(self, df_w: int) -> int:
        """Compute height degrees of freedom from width degrees of freedom.

        Parameters
        ----------
        df_w : int
            Degrees of freedom for the width dimension.

        Returns
        -------
        int
            Degrees of freedom for the height dimension, clipped to hdims.
        """
        return np.minimum(int(np.round(df_w * self.hdims / self.wdims)), self.hdims)

    def compute_best_rf_model(self, trace: np.ndarray, stim: np.ndarray) -> tuple:
        """Compute the best RF model using K-Fold cross-validation.

        Parameters
        ----------
        trace : np.ndarray
            Neural response trace.
        stim : np.ndarray
            Stimulus array.

        Returns
        -------
        tuple
            best_model : object
                The fitted GLM model with optimal hyperparameters.
            metrics_dev_opt : np.ndarray
                Array of development set metrics for all HP combinations and folds.
        """

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

        if self.kfold > 1:
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
    def clip_dfs(dfs: np.ndarray, dims: int) -> np.ndarray:
        """Clip degrees of freedom so they do not exceed the data dimensionality.

        Parameters
        ----------
        dfs : np.ndarray
            Array of candidate degrees of freedom.
        dims : int
            Maximum allowed degrees of freedom.

        Returns
        -------
        np.ndarray
            Clipped degrees of freedom array.
        """
        if np.all(dfs <= dims):
            return dfs
        elif np.any(dfs <= dims):
            warnings.warn(' No using df_ts larger than dims')
            return dfs[dfs <= dims]
        else:
            warnings.warn(' All df_ts larger than dims, use only dims as df')
            return np.array([dims])

    @staticmethod
    def _df_w2df_h(df_w: int, X_train: np.ndarray, hdims: int) -> int:
        """Convert width degrees of freedom to height degrees of freedom.

        Parameters
        ----------
        df_w : int
            Degrees of freedom for the width dimension.
        X_train : np.ndarray
            Training design matrix (used for shape reference).
        hdims : int
            Maximum allowed height degrees of freedom.

        Returns
        -------
        int
            Height degrees of freedom.
        """
        df_h = np.minimum(int(np.round(df_w * X_train.shape[1] / X_train.shape[2])), hdims)
        return df_h

    def create_and_fit_glm(self, df: tuple, X_train: np.ndarray, y_train: np.ndarray,
                           alphas, betas,
                           X_dev: np.ndarray = None,
                           y_dev: np.ndarray = None) -> tuple:
        """Create and fit a GLM model for given hyperparameters.

        Parameters
        ----------
        df : tuple
            Degrees of freedom per dimension (df_t, df_h, df_w) or (df_t,).
        X_train : np.ndarray
            Training design matrix.
        y_train : np.ndarray
            Training response.
        alphas : array-like
            Candidate alpha values.
        betas : array-like
            Candidate beta values.
        X_dev : np.ndarray, optional
            Development design matrix. Default is None.
        y_dev : np.ndarray, optional
            Development response. Default is None.

        Returns
        -------
        tuple
            model : object
                Fitted GLM model.
            metric_dev_opt_hp_sets : list or np.ndarray
                Development set metrics for each HP combination.
        """
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

    def create_glm(self, df: tuple, X_train: np.ndarray,
                   y_train: np.ndarray, X_dev: np.ndarray) -> object:
        """Create and initialize a GLM model.

        Parameters
        ----------
        df : tuple
            Degrees of freedom per dimension.
        X_train : np.ndarray
            Training design matrix.
        y_train : np.ndarray
            Training response.
        X_dev : np.ndarray
            Development design matrix (or None).

        Returns
        -------
        object
            Initialized GLM model.

        Raises
        ------
        ValueError
            If the size of df does not match self.dims.
        """

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

    def fit_glm(self, model: object, alphas, betas,
                y_train: np.ndarray, y_dev: np.ndarray = None) -> tuple:
        """Fit a GLM model by optimizing over alpha and beta hyperparameters.

        Parameters
        ----------
        model : object
            Initialized GLM model.
        alphas : array-like
            Candidate alpha values.
        betas : array-like
            Candidate beta values.
        y_train : np.ndarray
            Training response.
        y_dev : np.ndarray, optional
            Development response. Default is None.

        Returns
        -------
        tuple
            model : object
                Fitted GLM model.
            metric_dev_opt_hp_sets : list or np.ndarray
                Development set metrics for each HP combination.
        """
        min_iters_other = self.min_iters // 5

        if len(alphas) == 1 and len(betas) == 1:
            if y_dev is not None:
                assert 'dev' in model.X
                y = {'train': y_train, 'dev': y_dev}
                return_model = 'best_dev_metric'
                early_stopping = True
            else:
                y = {'train': y_train}
                return_model = 'best_train_cost'
                early_stopping = False

            if self.verbose > 0:
                print("return_model: ", return_model)

            model.fit(
                y=y, metric=self.metric, num_iters=self.max_iters, min_iters=self.min_iters, step_size=self.step_size,
                atol=self.atol, tolerance=self.tolerance, alpha=alphas[0], beta=betas[0],
                verbose=self.verbose, return_model=return_model, early_stopping=early_stopping)

            metric_dev_opt_hp_sets = [model.metric_dev_opt]
        else:
            metric_dev_opt_hp_sets = model.fit_hps(
                y={'train': y_train, 'dev': y_dev},
                num_iters=self.max_iters, verbose=self.verbose, tolerance=self.tolerance, metric=self.metric,
                step_size=self.step_size, step_size_finetune=self.step_size_finetune,
                alphas=alphas, betas=betas, min_iters=self.min_iters,
                min_iters_other=min_iters_other, atol=self.atol, return_model='best_train_cost',
                early_stopping=False)

        return model, metric_dev_opt_hp_sets

    def initialize_model(self, df: tuple, X_train: np.ndarray,
                         y_train: np.ndarray, X_dev: np.ndarray = None) -> object:
        """Initialize model parameters for faster optimization.

        Parameters
        ----------
        df : tuple
            Degrees of freedom per dimension.
        X_train : np.ndarray
            Training design matrix.
        y_train : np.ndarray
            Training response.
        X_dev : np.ndarray, optional
            Development design matrix. Default is None.

        Returns
        -------
        object
            Initialized rfest.GLM model.
        """

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
    def get_b_from_w(S: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Compute spline coefficients b from weights w and spline basis S.

        Parameters
        ----------
        S : np.ndarray
            Spline basis matrix, shape (n_features, n_basis).
        w : np.ndarray
            Weight vector, shape (n_features,).

        Returns
        -------
        np.ndarray
            Spline coefficients b, shape (n_basis,).
        """
        b, *_ = np.linalg.lstsq(S, w, rcond=None)
        return b

    @staticmethod
    def _get_model_info(model: object) -> str:
        """Format a summary string of key model hyperparameters and metrics.

        Parameters
        ----------
        model : object
            Fitted GLM model with attributes metric, metric_dev_opt, alpha, beta, df, dims.

        Returns
        -------
        str
            Human-readable info string.
        """
        info = f"{model.metric}={model.metric_dev_opt:.4f} "
        info += f"for HP-set: alpha={model.alpha:.5g}, beta={model.beta:.5g}, df={model.df}, dims={model.dims}"
        return info


def quality_test(score_trueX: float, score_permX: np.ndarray, metric: str) -> float:
    """Perform a one-sample t-test to assess whether the true score exceeds permuted scores.

    Parameters
    ----------
    score_trueX : float
        Model score on the true (non-permuted) test data.
    score_permX : np.ndarray
        Model scores on permuted test data, shape (n_perm,).
    metric : str
        Metric name used to determine the alternative hypothesis direction
        ('corrcoef' -> 'less', 'mse' -> 'greater', else 'two-sided').

    Returns
    -------
    float
        p-value of the one-sample t-test.
    """
    if metric == 'corrcoef':
        alternative = 'less'
    elif metric == 'mse':
        alternative = 'greater'
    else:
        alternative = 'two-sided'

    _, p_value = ttest_1samp(score_permX, score_trueX, nan_policy='raise', alternative=alternative)
    return p_value


def plot_rf_summary(rf: np.ndarray, quality_dict: dict, model_dict: dict,
                    title: str = "") -> tuple:
    """Plot tRF and sRFs and quality test.

    Parameters
    ----------
    rf : np.ndarray
        Receptive field array, shape (n_t,) for 1D or (n_t, n_y, n_x) for 3D.
    quality_dict : dict
        Quality metrics dict, may contain 'score_permX', 'quality', 'corrcoef_*', etc.
    model_dict : dict
        Model metadata dict, may contain 'metric_train', 'y_train', 'y_test', etc.
    title : str, optional
        Figure title. Default is "".

    Returns
    -------
    tuple
        fig : matplotlib.figure.Figure
            The figure object.
        axs : np.ndarray
            Array of axes objects.
    """

    plot_training = model_dict is not None and 'metric_train' in model_dict
    plot_y_train = model_dict is not None and 'y_train' in model_dict
    plot_y_dev = model_dict is not None and 'y_dev' in model_dict
    plot_y_test = model_dict is not None and 'y_test' in model_dict

    # Create figure and axes
    n_rows = 1 + plot_training + plot_y_train + plot_y_dev + plot_y_test
    fig, axs = plt.subplots(n_rows, 4, figsize=(8, n_rows * 1.9))

    axs_big = []
    for i in np.arange(2, n_rows):
        for ax in axs[i, :].flat:
            ax.remove()
        ax_big = fig.add_subplot(axs[0, 0].get_gridspec()[i, :])
        axs_big.append(ax_big)
    axs = np.append(axs.flatten(), np.array(axs_big))

    # Summarize RF
    if rf.ndim == 3:
        srf, trf = split_strf(rf)[:2]
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
        t_tRF = model_dict['rf_time']
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

    if not plot_training:
        plt.tight_layout(rect=(0.01, 0.01, 0.99, 0.90), h_pad=0.2, w_pad=0.2)
        return

    if model_dict is not None and 'metric_train' in model_dict:
        ax = axs[4]
        ax.set(title=f"Metrics\n{model_dict['return_model']}", xlabel='Iteration', ylabel='Corrcoef')
        ax.plot(model_dict['metric_train'], label='train', c='blue')
        ax.axvline(model_dict['best_iteration'], ls='-', c='orange')
        if model_dict.get('metric_dev', None) is not None:
            ax.plot(model_dict.get('metric_dev', np.zeros(0)), label='dev', c='red')
        if model_dict.get('metric_dev_opt', None) is not None:
            ax.axhline(model_dict['metric_dev_opt'], label='dev-opt', ls='--', c='red')
        ax.legend()

        ax = axs[5]
        ax.set(title=f"Cost\n{model_dict['return_model']}", xlabel='Iteration', ylabel='Cost')
        ax.plot(model_dict['cost_train'], label='train', c='blue')
        ax.axvline(model_dict['best_iteration'], ls='-', c='orange')
        if model_dict.get('cost_dev', None) is not None:
            ax.plot(model_dict['cost_dev'], label='dev', c='red', marker='.')

        ax = axs[6]
        ax.set(title=f"loglog-Cost\n{model_dict['return_model']}", xlabel='Iteration', ylabel='Cost')
        ax.loglog(np.arange(1, model_dict['cost_train'].size + 1), model_dict['cost_train'], label='train', c='blue')
        ax.axvline(1 + model_dict['best_iteration'], ls='-', c='orange')
        if model_dict.get('cost_dev', None) is not None:
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

    y_types = []
    if plot_y_train:
        y_types.append('train')
    if plot_y_dev:
        y_types.append('dev')
    if plot_y_test:
        y_types.append('test')

    for i, (y_type, ax) in enumerate(zip(y_types, axs_big)):
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


def plot_test_rf_quality(score_permX: np.ndarray, score_test: float,
                         cc_test: float = None, cc_dev: float = None,
                         ax=None, metric: str = ''):
    """Plot test results comparing true score to permuted scores.

    Parameters
    ----------
    score_permX : np.ndarray
        Model scores on permuted test data, shape (n_perm,).
    score_test : float
        Model score on true test data.
    cc_test : float, optional
        Correlation coefficient on test set. Default is None.
    cc_dev : float, optional
        Correlation coefficient on dev set. Default is None.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure. Default is None.
    metric : str, optional
        Metric name shown in legend. Default is ''.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.
    """
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
