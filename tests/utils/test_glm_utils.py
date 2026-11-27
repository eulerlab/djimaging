from types import SimpleNamespace

import numpy as np
import pytest

from djimaging.utils.receptive_fields.fit_rf_utils import build_design_matrix, get_rf_timing_params
from djimaging.utils.receptive_fields.glm_utils import ReceptiveFieldGLM


def _make_glm(stim, filter_dur_s_past=0.3, filter_dur_s_future=0.1, dt=0.1, **kwargs):
    trace = np.random.default_rng(0).normal(size=stim.shape[0]).astype(np.float32)
    return ReceptiveFieldGLM.from_stim(
        dt=dt, stim=stim, trace=trace, filter_dur_s_past=filter_dur_s_past, filter_dur_s_future=filter_dur_s_future,
        df_ts=(3,), df_ws=(2,) if stim.ndim >= 3 else (), betas=(0.1,), metric='mse', **kwargs)


@pytest.mark.parametrize('shape', [(50,), (50, 3), (50, 4, 5), (50, 4, 5, 2)])
def test_dims_and_channels(shape):
    stim = np.random.default_rng(1).choice([-1., 1.], size=shape).astype(np.float32)
    glm = _make_glm(stim)
    _, dim_t, _, _ = get_rf_timing_params(0.3, 0.1, 0.1)

    assert glm.stim_dims == (dim_t,) + shape[1:]
    assert glm.n_colors == (shape[-1] if len(shape) in (2, 4) else 1)
    assert glm.is_spatial == (len(shape) >= 3)
    assert glm.dims == ((dim_t,) + shape[1:3] if glm.is_spatial else (dim_t,))
    assert len(glm.channel_names) == glm.n_colors


@pytest.mark.parametrize('shape', [(50, 3), (50, 4, 5, 2)])
def test_split_channels_matches_single_channel_design_matrix(shape):
    """Splitting the shared design matrix must equal building the design matrix for each channel separately."""
    stim = np.random.default_rng(2).normal(size=shape).astype(np.float32)
    glm = _make_glm(stim)

    xs = glm._split_channels(glm.x_stimulus)
    assert list(xs) == glm.channel_names

    for c, name in enumerate(glm.channel_names):
        expected = build_design_matrix(stim[..., c], n_lag=glm.dim_t, shift=glm.shift, dtype=np.float32)
        np.testing.assert_array_equal(xs[name], expected)


def test_split_channels_single_channel_is_passthrough():
    stim = np.random.default_rng(3).normal(size=(40, 4, 5)).astype(np.float32)
    glm = _make_glm(stim)
    xs = glm._split_channels(glm.x_stimulus)
    assert list(xs) == ['stimulus']
    assert xs['stimulus'] is glm.x_stimulus


def test_inconsistent_inputs_raise():
    stim = np.random.default_rng(4).normal(size=(40, 4, 5)).astype(np.float32)
    glm = _make_glm(stim)
    kwargs = dict(dt=0.1, rf_time=glm.rf_time, shift=glm.shift, burn_in=glm.burn_in,
                  df_ts=(3,), df_ws=(2,), betas=(0.1,), metric='mse')

    with pytest.raises(ValueError, match="features"):
        ReceptiveFieldGLM(x_stimulus=glm.x_stimulus[:, :-1], trace=glm.trace, dims=glm.stim_dims, **kwargs)
    with pytest.raises(ValueError, match="length"):
        ReceptiveFieldGLM(x_stimulus=glm.x_stimulus, trace=glm.trace[:-1], dims=glm.stim_dims, **kwargs)


def _fit_kwargs(stim, dt=0.1):
    rf_time, dim_t, shift, burn_in = get_rf_timing_params(0.5, 0.1, dt)
    x_stimulus = build_design_matrix(stim, n_lag=dim_t, shift=shift, dtype=np.float32)
    glm_kwargs = dict(
        dt=dt, dims=(dim_t,) + stim.shape[1:], rf_time=rf_time, shift=shift, burn_in=burn_in,
        df_ts=(4,), df_ws=(3,) if stim.ndim >= 3 else (), betas=(0.01, 0.1), metric='mse',
        max_iters=60, min_iters=10, n_perm=3)
    return x_stimulus, glm_kwargs


@pytest.mark.parametrize('shape', [(300, 2), (300, 6, 5, 2)])
@pytest.mark.parametrize('kfold', [1, 2])
def test_basis_projections_match_design_matrix(shape, kfold):
    """Fitting to precomputed basis projections must give the same GLM as fitting to the design matrix."""
    pytest.importorskip('rfest')
    rng = np.random.default_rng(5)
    stim = rng.choice([-1., 1.], size=shape).astype(np.float32)
    x_stimulus, glm_kwargs = _fit_kwargs(stim)
    glm_kwargs['kfold'] = kfold
    trace = (x_stimulus @ rng.normal(size=x_stimulus.shape[1]) + rng.normal(size=shape[0])).astype(np.float32)
    trace = trace[:-7]  # traces can be shorter than the design matrix

    projections = ReceptiveFieldGLM(x_stimulus=x_stimulus, trace=None, **glm_kwargs).compute_basis_projections()
    rf_x, quality_x, model_x = ReceptiveFieldGLM(
        x_stimulus=x_stimulus[:trace.size], trace=trace, **glm_kwargs).compute_glm_receptive_field()
    rf_p, quality_p, model_p = ReceptiveFieldGLM(
        x_stimulus=None, trace=trace, basis_projections=projections, **glm_kwargs).compute_glm_receptive_field()

    np.testing.assert_allclose(rf_p, rf_x, rtol=1e-6, atol=1e-9)
    assert model_p['best_iteration'] == model_x['best_iteration']
    for k in ('mse_train', 'mse_test', 'corrcoef_test', 'score_test'):
        np.testing.assert_allclose(quality_p[k], quality_x[k], rtol=1e-6)


def test_basis_projections_must_match_parameters():
    stim = np.random.default_rng(6).normal(size=(100, 6, 5)).astype(np.float32)
    x_stimulus, glm_kwargs = _fit_kwargs(stim)
    trace = np.zeros(100, dtype=np.float32)

    with pytest.raises(ValueError, match="Either"):
        ReceptiveFieldGLM(x_stimulus=None, trace=trace, **glm_kwargs)

    projections = {(4, 4, 3): {'stimulus': np.zeros((100, 48))}}
    with pytest.raises(ValueError, match="dfs"):
        ReceptiveFieldGLM(x_stimulus=None, trace=trace, basis_projections=projections, **{**glm_kwargs, 'df_ts': (3,)})
    with pytest.raises(ValueError, match="fewer rows"):
        ReceptiveFieldGLM(x_stimulus=None, trace=np.zeros(101), basis_projections=projections, **glm_kwargs)


@pytest.mark.parametrize('metric, dev_metrics', [
    ('mse', {3: [0.5, 0.2], 4: [0.4, 0.3], 5: [0.9, 0.8]}),
    ('corrcoef', {3: [0.5, 0.2], 4: [0.4, 0.9], 5: [0.3, 0.3]}),
])
def test_best_df_model_is_returned_without_refit(metric, dev_metrics, monkeypatch):
    """With kfold=1 the model of the best df is returned, not the one fitted last."""
    stim = np.random.default_rng(7).normal(size=(100, 2)).astype(np.float32)
    x_stimulus, glm_kwargs = _fit_kwargs(stim)
    glm_kwargs.update(df_ts=tuple(dev_metrics), betas=(0.01, 0.1), metric=metric, kfold=1)
    glm = ReceptiveFieldGLM(x_stimulus=x_stimulus, trace=np.zeros(100, dtype=np.float32), **glm_kwargs)

    def fake_fit(df, betas, **kwargs):
        metrics = np.array(dev_metrics[df[0]])
        best = metrics.argmax() if metric == 'corrcoef' else metrics.argmin()
        return SimpleNamespace(df=df, beta=betas[best]), metrics

    monkeypatch.setattr(glm, 'create_and_fit_glm', fake_fit)
    model, metrics_dev_opt = glm.compute_best_rf_model(trace=np.zeros(80, dtype=np.float32), rows=slice(0, 80))

    assert metrics_dev_opt.shape == (1, 3, 2)
    assert model.df == ((3,) if metric == 'mse' else (4,))
    assert model.beta == 0.1


def test_clipped_dfs_do_not_warn_again():
    """`RfGlmTemplate.prepare_sta_params` passes the clipped dfs on, so that the clipping warns only once."""
    import warnings

    stim = np.random.default_rng(8).normal(size=(100, 6, 5)).astype(np.float32)
    x_stimulus, glm_kwargs = _fit_kwargs(stim)
    glm_kwargs.update(df_ts=(100,), df_ws=(3, 50))

    with pytest.warns(UserWarning, match="degrees of freedom"):
        glm = ReceptiveFieldGLM(x_stimulus=x_stimulus, trace=None, **glm_kwargs)

    glm_kwargs.update(df_ts=tuple(glm.df_ts), df_ws=tuple(glm.df_ws))
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        ReceptiveFieldGLM(x_stimulus=x_stimulus, trace=np.zeros(100, dtype=np.float32), **glm_kwargs)


@pytest.mark.parametrize('shape, kfold, init_method, betas', [
    ((300, 2), 0, 'random', (0.01,)),
    ((300, 6, 5, 2), 0, 'mle', (0.01,)),
    ((300, 2), 1, 'random', (0.01, 0.1)),
    ((300, 6, 5), 1, 'mle', (0.01,)),
    ((300, 2), 2, 'random', (0.01, 0.1)),
])
def test_batched_fits_match_single_fits(shape, kfold, init_method, betas):
    """Jointly fitting several traces must give the same GLMs as fitting them one by one."""
    pytest.importorskip('rfest')
    from djimaging.utils.receptive_fields.glm_utils import fit_receptive_field_glm, fit_receptive_field_glms

    rng = np.random.default_rng(9)
    stim = rng.choice([-1., 1.], size=shape).astype(np.float32)
    x_stimulus, glm_kwargs = _fit_kwargs(stim)
    glm_kwargs.update(kfold=kfold, init_method=init_method, betas=betas, frac_test=0.2 if kfold else 0.,
                      df_ts=(3, 4) if kfold else (4,))
    projections = ReceptiveFieldGLM(x_stimulus=x_stimulus, trace=None, **glm_kwargs).compute_basis_projections()

    # Different noise levels, so that the fits stop at different iterations; one trace is shorter.
    traces = [(x_stimulus @ rng.normal(size=x_stimulus.shape[1]) + s * rng.normal(size=shape[0])).astype(np.float32)
              for s in (0.1, 1., 3., 10.)]
    traces[2] = traces[2][:-7]

    batched = fit_receptive_field_glms(traces, glm_kwargs=glm_kwargs, basis_projections=projections)
    for trace, (rf_b, quality_b, model_b) in zip(traces, batched):
        rf_s, quality_s, model_s = fit_receptive_field_glm(trace, glm_kwargs=glm_kwargs, basis_projections=projections)
        np.testing.assert_allclose(rf_b, rf_s, rtol=1e-5, atol=1e-8)
        for k in ('best_iteration', 'train_stop', 'beta', 'df'):
            assert model_b[k] == model_s[k], k
        np.testing.assert_allclose(model_b['cost_train'], model_s['cost_train'], rtol=1e-5)
        np.testing.assert_allclose(model_b['metrics_dev_opt'], model_s['metrics_dev_opt'], rtol=1e-5)
        assert quality_b.keys() == quality_s.keys()
        for k, v in quality_s.items():
            if isinstance(v, str):
                assert quality_b[k] == v
            else:
                np.testing.assert_allclose(quality_b[k], v, rtol=1e-5, atol=1e-8)
