"""
Extension of the official RFEst GLM (https://github.com/berenslab/RFEst, MIT license).

The official `rfest.GLM` does not support
    - passing a pre-computed design matrix (`is_design_matrix`),
    - fitting/fixing the intercept and the output gain (`fit_intercept`, `fit_R`),
    - hyperparameter search over (alpha, beta) (`fit_hps`),
    - early stopping control and `best_dev_metric` model selection.
These were previously provided by a custom RFEst fork; they are implemented here on top of the official package.
"""

import itertools
import time
import warnings
from copy import deepcopy

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    import jax.random as random
    from jax import jit, value_and_grad

    try:
        from jax.example_libraries import optimizers
    except ImportError:
        from jax.experimental import optimizers

    from rfest import GLM as _RFEstGLM
    from rfest.loss import loss_mse, loss_neglogli, loss_penalty
    from rfest.splines import build_spline_matrix

    jax.config.update('jax_platform_name', 'cpu')
except ImportError:
    jax = None
    jnp = None
    _RFEstGLM = object


def apply_nonlinearity(x, kind):
    """Apply a named nonlinearity."""
    if kind == 'softplus':
        return jnp.log(1. + jnp.exp(x))
    elif kind == 'exponential':
        return jnp.exp(x)
    elif kind == 'softmax':
        z = jnp.exp(x)
        return 100. * z / z.sum()
    elif kind == 'sigmoid':
        return 1. / (1. + jnp.exp(-x))
    elif kind == 'sigmoid_shifted':
        return 1. / (1. + jnp.exp(-x + 1))
    elif kind == 'tanh':
        return jnp.tanh(x)
    elif kind == 'relu':
        return jnp.where(x > 0., x, 0.)
    elif kind == 'leaky_relu':
        return jnp.where(x > 0., x, 0.01 * x)
    elif kind == 'selu':
        return jax.nn.selu(x)
    elif kind == 'swish':
        return jax.nn.swish(x)
    elif kind == 'elu':
        return jax.nn.elu(x)
    elif kind in ('none', 'identity'):
        return x
    else:
        raise ValueError(f'Input filter nonlinearity `{kind}` is not supported.')


class GLM(_RFEstGLM):
    """Official `rfest.GLM` plus pre-computed design matrices, fit_R/fit_intercept and hyperparameter search."""

    def __init__(self, distr='poisson', output_nonlinearity='none', dtype=None):
        if jax is None:
            raise ImportError("rfest and jax are required for the GLM, install with `pip install djimaging[rf]`.")
        super().__init__(distr=distr, output_nonlinearity=output_nonlinearity,
                         **({} if dtype is None else dict(dtype=dtype)))
        self.fit_R = None
        self.fit_intercept = None
        self.R = {}

    def fnl(self, x, kind, params=None):
        return apply_nonlinearity(x, kind=kind)

    def add_design_matrix(self, X, dims=None, df=None, smooth=None, lag=True, is_design_matrix=False,
                          filter_nonlinearity='none', kind='train', name='stimulus', shift=0, burn_in=None):
        """Add a design matrix. If `is_design_matrix`, X is used as is (after burn-in) instead of building lags."""
        if not is_design_matrix:
            return super().add_design_matrix(
                X, dims=dims, df=df, smooth=smooth, lag=lag, filter_nonlinearity=filter_nonlinearity,
                kind=kind, name=name, shift=shift, burn_in=burn_in)

        X = X[:, jnp.newaxis].astype(self.dtype) if len(X.shape) == 1 else X.astype(self.dtype)

        if kind not in self.X:
            self.X[kind] = {}

        if kind == 'train':
            self.filter_nonlinearity[name] = filter_nonlinearity
            self.filter_names.append(name)
            dims = dims if type(dims) is not int else [dims, ]
            self.dims[name] = dims
            self.shift[name] = shift
        else:
            dims = self.dims[name]

        if self.burn_in is None:
            self.burn_in = dims[0] - 1 if burn_in is None else burn_in

        self.X[kind][name] = X[self.burn_in:]

        if smooth is None:
            if name in self.S:
                self.XS.setdefault(kind, {})[name] = self.X[kind][name] @ self.S[name]
            elif kind == 'test':
                S = self.S['stimulus_s0'] if (self.num_subunits or 1) > 1 else self.S[name]
                self.XS.setdefault(kind, {})[name] = self.X[kind][name] @ S
            elif kind == 'train':
                self.n_features[name] = self.X['train'][name].shape[1]
        else:
            self.df[name] = df if type(df) is not int else [df, ]
            self.S[name] = build_spline_matrix(dims=dims, df=self.df[name], smooth=smooth, dtype=self.dtype)
            self.XS.setdefault(kind, {})[name] = self.X[kind][name] @ self.S[name]
            if kind == 'train':
                self.n_features[name] = self.XS['train'][name].shape[1]

    def initialize(self, y=None, num_subunits=1, dt=1., method='random', compute_ci=True, random_seed=2046,
                   verbose=0, add_noise_to_mle=False, fit_R=False, fit_intercept=True):
        self.dt = dt
        self.init_method = method
        self.num_subunits = num_subunits
        self.compute_ci = compute_ci
        self.fit_R = fit_R
        self.fit_intercept = fit_intercept

        if method == 'random':
            self._initialize_random(verbose=verbose, random_seed=random_seed)
        elif method == 'mle':
            self._initialize_mle(y)
        else:
            raise ValueError(f'`{method}` is not supported.')

        if num_subunits > 1:
            self._initialize_subunits(num_subunits=num_subunits, method=method, filter_names=self.filter_names.copy())

        self.p[method] = {}
        self._initialize_p0(method=method, add_noise_to_mle=add_noise_to_mle, random_seed=random_seed)

    def _initialize_random(self, verbose=0, random_seed=2046):
        self.b['random'] = {}
        self.w['random'] = {}
        self.intercept['random'] = {name: 0. for name in self.filter_names}
        self.intercept['random']['global'] = 0.
        self.R['random'] = {'global': 1.}

        for i, name in enumerate(self.filter_names):
            key = random.PRNGKey(random_seed + i)
            if name in self.S:
                self.b['random'][name] = random.normal(
                    key, shape=(self.XS['train'][name].shape[1], 1)).astype(self.dtype)
                self.w['random'][name] = self.S[name] @ self.b['random'][name]
            else:
                self.w['random'][name] = random.normal(
                    key, shape=(self.X['train'][name].shape[1], 1)).astype(self.dtype)

    def _initialize_mle(self, y, compute_ci=True):
        if self.mle_computed:
            warnings.warn('MLE was already computed. Recomputing ...')

        if self.compute_ci is None:
            self.compute_ci = compute_ci

        if type(y) is dict:
            y_train = y['train']
            y_dev = y.get('dev', None)
        else:
            y = {'train': y}
            y_train = y['train']
            y_dev = None

        n_samples = len(y_train) - self.burn_in
        if n_samples <= 0:
            raise ValueError('Training set is empty after burned in.')

        X = jnp.hstack([jnp.hstack(
            [jnp.ones(n_samples)[:, jnp.newaxis], self.XS['train'][name]]) if name in self.S else jnp.hstack(
            [jnp.ones(n_samples)[:, jnp.newaxis], self.X['train'][name]]) for name in self.filter_names])
        X = jnp.hstack([jnp.ones(n_samples)[:, jnp.newaxis], X])

        mle = jnp.linalg.lstsq(X.T @ X, X.T @ y_train[self.burn_in:], rcond=None)[0]

        self.R['mle'] = {'global': 1.}
        self.b['mle'] = {}
        self.w['mle'] = {}
        self.intercept['mle'] = {}

        # Slice the mle vector into the parts of each filter
        lims = np.cumsum([0] + [self.n_features[name] + 1 for name in self.n_features])
        idx = [jnp.array((lims[i], lims[i + 1])) for i in range(len(lims) - 1)]
        self.idx = idx

        for i, name in enumerate(self.filter_names):
            mle_params = mle[idx[i][0]:idx[i][1]][:, jnp.newaxis].astype(self.dtype)
            self.intercept['mle'][name] = mle_params[0]
            if name in self.S:
                self.b['mle'][name] = mle_params[1:]
                self.w['mle'][name] = self.S[name] @ self.b['mle'][name]
            else:
                self.w['mle'][name] = mle_params[1:]

        self.intercept['mle']['global'] = mle[0]

        self.p['mle'] = {}
        self.y_pred['mle'] = {}

        for name in self.filter_names:
            self.p['mle'][name] = self.b['mle'][name] if name in self.S else self.w['mle'][name]

        if self.fit_intercept:
            self.p['mle']['intercept'] = self.intercept['mle']
        if self.fit_R:
            self.p['mle']['R'] = self.R['mle']

        self.y['train'] = y_train[self.burn_in:]
        self.y_pred['mle']['train'] = self.forwardpass(self.p['mle'], kind='train')

        if self.compute_ci:
            self._get_filter_variance(w_type='mle')
            self._get_response_variance(w_type='mle', kind='train')

        if y_dev is not None:
            self.y['dev'] = y_dev[self.burn_in:]
            if len(self.y['dev']) == 0:
                raise ValueError('Dev set is empty after burned in.')
            self.y_pred['mle']['dev'] = self.forwardpass(self.p['mle'], kind='dev')
            if self.compute_ci:
                self._get_response_variance(w_type='mle', kind='dev')

        self.mle_computed = True

    def _initialize_p0(self, method, add_noise_to_mle=False, random_seed=2046):
        p0 = {}
        for i, name in enumerate(self.filter_names):
            val = self.b[method][name] if name in self.S else self.w[method][name]
            key = random.PRNGKey(random_seed + i)
            self.p[method][name] = val
            p0[name] = val + add_noise_to_mle * random.normal(key, shape=val.shape).astype(self.dtype)

        if self.fit_intercept:
            self.p[method]['intercept'] = self.intercept[method]
            p0['intercept'] = self.intercept[method]

        if self.fit_R:
            self.p[method]['R'] = self.R[method]
            p0['R'] = self.R[method]

        self.p0 = p0

    def get_intercept(self, p=None):
        return p['intercept'] if self.fit_intercept else self.intercept[self.init_method]

    def get_R(self, p=None):
        return p['R'] if self.fit_R else self.R[self.init_method]

    def forwardpass(self, p, kind):
        intercept = self.get_intercept(p)
        R = self.get_R(p)

        filters_output = []
        for name in self.X[kind]:
            input_term = self.XS[kind][name] if name in self.S else self.X[kind][name]
            filters_output.append(self.fnl(
                jnp.squeeze(input_term @ p[name]) + intercept[name], kind=self.filter_nonlinearity[name]))

        filters_output = jnp.array(filters_output).sum(0)

        return R['global'] * self.fnl(
            filters_output + intercept['global'], kind=self.output_nonlinearity).astype(self.dtype)

    def cost(self, p, kind='train', precomputed=None, penalize=True):
        """Cost function; the poisson likelihood is normalized by the number of samples."""
        y = self.y[kind]
        r = self.forwardpass(p, kind) if precomputed is None else precomputed

        if self.distr == 'gaussian':
            loss = loss_mse(y, r)
        elif self.distr == 'poisson':
            loss = loss_neglogli(y, r) / y.size
        else:
            raise NotImplementedError(self.distr)

        if penalize and (self.beta is not None and self.beta != 0) and kind == 'train':
            loss += loss_penalty(jnp.hstack([p[name].flatten() for name in self.filter_names]), self.alpha, self.beta)

        return jnp.squeeze(loss)

    def optimize(self, p0, num_iters, metric, step_size, tolerance, verbose, return_model, atol=1e-5, min_iters=300,
                 early_stopping=True):
        """Adam optimization with optional early stopping and several criteria for the returned model."""
        if return_model is None:
            return_model = 'best_dev_cost' if 'dev' in self.y else 'best_train_cost'

        assert ('dev' in self.y) or ('dev' not in return_model), 'Cannot use dev set if dev set is not given.'

        @jit
        def step(_i, _opt_state):
            p = get_params(_opt_state)
            loss, grad = value_and_grad(self.cost)(p)
            return loss, opt_update(_i, grad, _opt_state)

        opt_init, opt_update, get_params = optimizers.adam(step_size=step_size)
        opt_state = opt_init(p0)

        cost_train = np.full(num_iters, np.nan)
        cost_dev = np.full(num_iters, np.nan)
        metric_train = np.full(num_iters, np.nan)
        metric_dev = np.full(num_iters, np.nan)
        params_list = []

        has_dev = 'dev' in self.y

        if verbose:
            self.print_progress_header(
                c_train=True, c_dev=has_dev, m_train=metric is not None, m_dev=metric is not None and has_dev)

        time_start = time.time()
        for i in range(num_iters):
            cost_train[i], opt_state = step(i, opt_state)

            params = get_params(opt_state)
            params_list.append(params)

            y_pred_train = self.forwardpass(p=params, kind='train')
            metric_train[i] = self.compute_score(self.y['train'], y_pred_train, metric)

            if has_dev:
                y_pred_dev = self.forwardpass(p=params, kind='dev')
                cost_dev[i] = self.cost(p=params, kind='dev', precomputed=y_pred_dev, penalize=False)
                metric_dev[i] = self.compute_score(self.y['dev'], y_pred_dev, metric)

            time_elapsed = time.time() - time_start

            if verbose and i % int(verbose) == 0:
                self.print_progress(
                    i, time_elapsed, c_train=cost_train[i], c_dev=cost_dev[i],
                    m_train=metric_train[i], m_dev=metric_dev[i])

            if tolerance and i > min_iters:  # tolerance = 0: no early stop.
                total_time_elapsed = time.time() - time_start

                if early_stopping and has_dev and np.all(np.diff(cost_dev[i - tolerance:i]) > 0):
                    stop = 'dev_stop'
                    break

                if np.all(-np.diff(cost_train[i - tolerance:i]) < atol):
                    stop = 'train_stop'
                    break
        else:
            total_time_elapsed = time.time() - time_start
            stop = 'maxiter_stop'

        if verbose:
            self.print_progress(
                i, time_elapsed, c_train=cost_train[i], c_dev=cost_dev[i],
                m_train=metric_train[i], m_dev=metric_dev[i])
            print(f'Stop at {i + 1}/{num_iters} steps: {stop}')
            print(f'Total time elapsed: {total_time_elapsed:.3f}s.')

        if return_model == 'best_dev_cost':
            best = np.argmin(cost_dev[:i + 1])
        elif return_model == 'best_train_cost':
            best = np.argmin(cost_train[:i + 1])
        elif return_model == 'best_dev_metric':
            best = (np.argmin if metric in ['mse', 'gcv'] else np.argmax)(metric_dev[:i + 1])
        elif return_model == 'best_train_metric':
            best = (np.argmin if metric in ['mse', 'gcv'] else np.argmax)(metric_train[:i + 1])
        else:
            if return_model != 'last':
                print('Provided `return_model` is not supported. Fallback to `last`')
            best = i - tolerance if stop == 'dev_stop' else i

        if verbose:
            print(f'Returning model: {return_model} at iteration {best + 1} of {i + 1} (Max: {num_iters}).\n')

        self.best_iteration = best
        self.return_model = return_model
        self.train_stop = stop
        self.cost_train = cost_train[:i + 1]
        self.cost_dev = cost_dev[:i + 1]
        self.metric_train = metric_train[:i + 1]
        self.metric_dev = metric_dev[:i + 1]
        self.metric_dev_opt = metric_dev[best]
        self.total_time_elapsed = total_time_elapsed
        self.all_params = params_list[:i + 1]

        self.y_pred['opt'].update({'train': y_pred_train})
        if has_dev:
            self.y_pred['opt'].update({'dev': y_pred_dev})

        return params_list[best]

    def fit(self, y=None, num_iters=3, alpha=1, beta=0.01, metric='corrcoef', step_size=1e-3,
            tolerance=10, verbose=True, return_model=None, atol=1e-5, min_iters=300, early_stopping=False):
        self.alpha = alpha
        self.beta = beta
        self.metric = metric

        if y is None:
            if 'train' not in self.y:
                raise ValueError('No `y` is provided.')
        elif type(y) is dict:
            self.y['train'] = y['train'][self.burn_in:].astype(self.dtype)
            if 'dev' in y:
                self.y['dev'] = y['dev'][self.burn_in:].astype(self.dtype)
        else:
            self.y['train'] = y[self.burn_in:].astype(self.dtype)

        self.y_pred['opt'] = {}

        self.p['opt'] = self.optimize(
            self.p0, num_iters, metric, step_size, tolerance, verbose, return_model, atol, min_iters,
            early_stopping=early_stopping)
        self._extract_opt_params(self.p['opt'])

    def fit_hps(self, y=None, num_iters=300, metric='corrcoef', step_size=1e-3, step_size_finetune=None,
                tolerance=10, verbose=0, return_model='best_train_cost',
                atol=1e-5, min_iters=300, alphas=(1.,), betas=(1.,), min_iters_other=None,
                early_stopping=False):
        """Fit for all combinations of alphas and betas, warm-starting each fit from the previous one.

        Returns the best dev-set metric for each (alpha, beta) combination. The best model is kept.
        """
        if y is not None:
            assert type(y) is dict and 'train' in y and 'dev' in y
            self.y['train'] = y['train'][self.burn_in:].astype(self.dtype)
            self.y['dev'] = y['dev'][self.burn_in:].astype(self.dtype)

        assert 'train' in self.y and 'dev' in self.y

        self.metric = metric
        if min_iters_other is None:
            min_iters_other = min_iters

        hp_sets = list(itertools.product(alphas, betas))
        metric_dev_opt_hp_sets = np.full(len(hp_sets), np.nan)

        self.y_pred['opt'] = dict()
        self.y_pred['hp_opt'] = dict()
        self.p['hp_opt'] = dict(hp_sets=[{"alpha": alpha, "beta": beta} for alpha, beta in hp_sets])

        for i, (alpha, beta) in enumerate(hp_sets):
            step_size_i = step_size if ((i == 0) or (step_size_finetune is None)) else step_size_finetune

            self.alpha = alpha
            self.beta = beta

            p0 = self.p0 if i == 0 else self.p['hp_opt'][f'set_{i - 1}']
            assert self.p0.keys() == p0.keys()

            self.p['hp_opt'][f'set_{i}'] = self.optimize(
                p0=deepcopy(p0), num_iters=num_iters, metric=metric, step_size=step_size_i, tolerance=tolerance,
                verbose=verbose, return_model=return_model, atol=atol,
                min_iters=min_iters if i == 0 else min_iters_other, early_stopping=early_stopping)

            self.y_pred['hp_opt'][f'set_{i}'] = deepcopy(self.y_pred['opt'])
            metric_dev_opt_hp_sets[i] = self.metric_dev_opt

            if verbose:
                print(f"\talpha={alpha:.2g}, beta={beta:.2g}, step_size={step_size_i:.2g}")
                print(f"--> {self.metric}={metric_dev_opt_hp_sets[i]:.2g}\n")

        if metric in ['mse', 'gcv']:
            best_hp_set = np.argmin(metric_dev_opt_hp_sets)
        else:
            best_hp_set = np.argmax(metric_dev_opt_hp_sets)

        self.alpha, self.beta = hp_sets[best_hp_set]
        self.metric_dev_opt = metric_dev_opt_hp_sets[best_hp_set]

        self.p['opt'] = self.p['hp_opt'][f'set_{best_hp_set}']
        self.y_pred['opt'] = deepcopy(self.y_pred['hp_opt'][f'set_{best_hp_set}'])

        self._extract_opt_params(self.p['opt'])

        return metric_dev_opt_hp_sets

    def _extract_opt_params(self, p):
        self.b['opt'] = {}
        self.w['opt'] = {}
        for name in self.filter_names:
            if name in self.S:
                self.b['opt'][name] = p[name]
                self.w['opt'][name] = self.S[name] @ self.b['opt'][name]
            else:
                self.w['opt'][name] = p[name]

        if self.fit_intercept:
            self.intercept['opt'] = p['intercept']

        if self.fit_R:
            self.R['opt'] = p['R']

        if self.compute_ci:
            self._get_filter_variance(w_type='opt')
            self._get_response_variance(w_type='opt', kind='train')
            if 'dev' in self.y:
                self._get_response_variance(w_type='opt', kind='dev')

    def predict(self, X, w_type='opt', is_design_matrix=True):
        """Predict the response for the test set.

        X : dict or array
            Per-filter design matrices (or raw stimuli if `is_design_matrix` is False).
        """
        p = self.p[w_type]

        self.X['test'] = {}
        self.XS['test'] = {}

        if type(X) is not dict:
            X = {'stimulus': X}

        for name in X:
            self.add_design_matrix(X[name], dims=self.dims[name], shift=self.shift[name], name=name, kind='test',
                                   is_design_matrix=is_design_matrix)

        if self.num_subunits != 1:
            for name in self.filter_names:
                if 'stimulus' in name:
                    self.X['test'][name] = self.X['test']['stimulus']
                    self.XS['test'][name] = self.XS['test']['stimulus']
            self.X['test'].pop('stimulus')
            self.XS['test'].pop('stimulus')

        y_pred = self.forwardpass(p, kind='test')
        if self.compute_ci:
            self._get_response_variance(w_type=w_type, kind='test')

        return y_pred

    def _get_response_variance(self, w_type='opt', kind='train'):
        """Compute the variance and standard error of the predicted response."""
        S = self.S
        X = self.X[kind]
        XS = self.XS.get(kind, None)

        b = self.b[w_type]
        w = self.w[w_type]
        V = self.V[w_type]

        y_se = {}
        y_pred_filters = {}
        y_pred_filters_upper = {}
        y_pred_filters_lower = {}

        intercept = self.intercept['opt' if (w_type == 'opt') and self.fit_intercept else self.init_method]

        for name in X:
            if name in S:
                y_se[name] = jnp.sqrt(jnp.sum(XS[name] @ V[name] * XS[name], 1))
                y_pred_filters[name] = self.fnl((XS[name] @ b[name] + intercept[name]).flatten(),
                                                kind=self.filter_nonlinearity[name])
            else:
                y_se[name] = jnp.sqrt(jnp.sum(X[name] @ V[name] * X[name], 1))
                y_pred_filters[name] = self.fnl((X[name] @ w[name] + intercept[name]).flatten(),
                                                kind=self.filter_nonlinearity[name])

            lin = (X[name] @ w[name] + intercept[name]).flatten()
            y_pred_filters_upper[name] = self.fnl(lin + 2 * y_se[name], kind=self.filter_nonlinearity[name])
            y_pred_filters_lower[name] = self.fnl(lin - 2 * y_se[name], kind=self.filter_nonlinearity[name])

        def _out(filters):
            return self.fnl(jnp.array([filters[name] for name in X]).sum(0) + intercept['global'],
                            kind=self.output_nonlinearity)

        if w_type not in self.y_pred_lower:
            self.y_pred_lower[w_type] = {}
            self.y_pred_upper[w_type] = {}

        self.y_pred[w_type][kind] = _out(y_pred_filters)
        self.y_pred_upper[w_type][kind] = _out(y_pred_filters_upper)
        self.y_pred_lower[w_type][kind] = _out(y_pred_filters_lower)
