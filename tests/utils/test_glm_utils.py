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
