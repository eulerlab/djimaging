# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Receptive field estimation methods
#
# This notebook compares the RF estimation methods of the `FastSta` and `RfGlm` table templates
# (`djimaging/tables/receptivefield/fast_sta.py` and `glm.py`) on a simulated cell with a known RF.
#
# No database is needed. The notebook calls the same functions that the tables call in their `make`:
#
# | Step | Function used by the tables |
# |---|---|
# | Stimulus preprocessing + design matrix | `FastStaParamsTemplate.prepare_stimulus` |
# | Trace preprocessing + alignment to stimulus | `FastStaTemplate._prepare_trace` |
# | STA (optionally whitened) | `FastStaTemplate._compute_entries` (uses `compute_rf_sta`, `compute_whitening_matrix`) |
# | GLM | `RfGlmTemplate._compute_entries` (uses `ReceptiveFieldGLM`) |
#
# Compared methods:
# 1. STA of the raw (z-scored) trace (`fit_kind='trace'`)
# 2. STA of the trace gradient (`fit_kind='gradient'`, the `FastSta` default; counts rises of the calcium signal)
# 3. Whitened STA of the gradient (`whiten=True`)
# 4. GLM with spline-parameterised, regularised RF (`RfGlmTemplate`)

# %%
import time

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d

from djimaging.tables.receptivefield.fast_sta import FastStaParamsTemplate, FastStaTemplate
from djimaging.tables.receptivefield.glm import RfGlmParamsTemplate, RfGlmTemplate
from djimaging.utils.receptive_fields.fit_rf_utils import compute_whitening_matrix
from djimaging.utils.receptive_fields.glm_utils import plot_rf_summary
from djimaging.utils.receptive_fields.plot_rf_utils import plot_rf_frames
from djimaging.utils.receptive_fields.preprocess_rf_utils import preprocess_stimulus

# %% [markdown]
# ## 1. Simulate a cell with known RF
#
# Binary noise stimulus on a small grid, presented at one frame per trigger. The cell is a linear-nonlinear
# unit with a space-time separable RF (Gaussian centre x biphasic temporal filter). Its output is convolved
# with a calcium-indicator kernel and sampled at a lower rate than the stimulus, like a 2P recording.

# %%
rng = np.random.default_rng(42)

n_y, n_x = 6, 8
n_frames = 2400  # triggers
dt_trigger = 0.2  # 5 Hz noise
dt_trace = 0.128  # ~7.8 Hz line scan, must be faster than the stimulus so that each stimulus bin has samples
noise_level = 0.5  # std of noise added to the trace, relative to the signal

stim = rng.choice([-1., 1.], size=(n_frames, n_y, n_x)).astype(np.float32)
triggertimes = np.arange(n_frames) * dt_trigger

# Ground truth RF on the stimulus frame grid. Lag 0 is the most recent frame.
n_lags_true = 6
lags = np.arange(n_lags_true) * dt_trigger
temporal = np.exp(-lags / 0.3) * np.sin(2 * np.pi * lags / 0.9)  # biphasic
temporal /= np.abs(temporal).sum()
yy, xx = np.mgrid[:n_y, :n_x]
spatial = np.exp(-((yy - 2.5) ** 2 + (xx - 4.) ** 2) / (2 * 1.2 ** 2))
spatial /= spatial.sum()
rf_true_lagmajor = temporal[:, None, None] * spatial[None]  # (lag, y, x)

# Generator signal: g[t] = sum_lag sum_xy rf[lag, xy] * stim[t - lag, xy]
g = np.zeros(n_frames)
for lag in range(n_lags_true):
    g[lag:] += (stim[:n_frames - lag] * rf_true_lagmajor[lag]).sum(axis=(1, 2))
g = (g - g.mean()) / g.std()

rate = np.log1p(np.exp(2 * g - 0.5))  # soft-plus nonlinearity
kernel_t = np.arange(0, 2, dt_trigger)
ca = np.convolve(rate, np.exp(-kernel_t / 0.5))[:n_frames]  # calcium decay

# Sample at the (lower) trace rate and add noise
tracetime = np.arange(0, triggertimes[-1] + dt_trigger, dt_trace) + 0.03  # covers the whole stimulus
trace = np.interp(tracetime, triggertimes, ca)
trace = trace + noise_level * trace.std() * rng.standard_normal(trace.size)
trace = gaussian_filter1d(trace, 0.5)

fig, axs = plt.subplots(1, 3, figsize=(12, 2.8))
axs[0].imshow(stim[0], cmap='gray')
axs[0].set_title('Example stimulus frame')
axs[1].plot(lags, temporal)
axs[1].set(title='True temporal filter', xlabel='Lag (s)')
axs[2].plot(tracetime[:200], trace[:200])
axs[2].set(title='Simulated trace', xlabel='Time (s)')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. Build the design matrix (shared by all methods)
#
# In the tables this happens once per stimulus in `FastStaParamsTemplate.add_default` and the result is stored
# as `x_stimulus`. All traces of a stimulus then reuse it.

# %%
filter_dur_s_past = 1.6
filter_dur_s_future = 0.0
fupsample_stim = 1

x_stimulus, rf_time, dims, burn_in, shift, dt, t0 = FastStaParamsTemplate.prepare_stimulus(
    stim=stim, triggertimes=triggertimes, nframes_per_trigger=1, fupsample_stim=fupsample_stim,
    filter_dur_s_past=filter_dur_s_past, filter_dur_s_future=filter_dur_s_future)
x_stimulus = np.ascontiguousarray(x_stimulus.astype(np.float32))  # as in `prepare_sta_params`

# The tables rebuild stimtime from the presentation triggertimes in `make_compute`.
stimtime, _ = preprocess_stimulus(stim, triggertimes, nframes_per_trigger=1, fupsample_stim=fupsample_stim)

print(f"{x_stimulus.shape=}, {dims=}, {dt=:.3f}, {shift=}, {burn_in=}")

# Ground truth in the layout of the estimates: (time, y, x), ordered from past (-) to present (0)
rf_true = np.zeros(dims)
rf_true[-n_lags_true:] = rf_true_lagmajor[::-1]

# %% [markdown]
# ## 3. STA variants
#
# `FastStaTemplate._prepare_trace` preprocesses the trace (`fit_kind`, up-sampling, filtering) and aligns it to the
# stimulus. `FastStaTemplate._compute_entries` then computes `X.T @ y / sum(y)` for all traces of a field at once.
# With a `whitening_matrix` the STA is multiplied by the regularised inverse stimulus covariance.
# For white noise the whitening only corrects finite-sample correlations.

# %%
sta_kw = dict(fupsample_trace=4, lowpass_cutoff=0, pre_blur_sigma_s=0, post_blur_sigma_s=0)

whitening_matrix = compute_whitening_matrix(x_stimulus, reg=1e-3)

sta_variants = {
    'STA (trace)': dict(fit_kind='trace', whitening_matrix=None),
    'STA (gradient)': dict(fit_kind='gradient', whitening_matrix=None),
    'STA (gradient, whitened)': dict(fit_kind='gradient', whitening_matrix=whitening_matrix),
}

rfs = {}
for name, kw in sta_variants.items():
    t_start = time.time()
    aligned_trace = FastStaTemplate._prepare_trace(
        tracetime, trace, stimtime, x_stimulus.shape[0], fit_kind=kw['fit_kind'], **sta_kw)
    # `_compute_entries` only needs the class, so it is called with the class as `self`.
    rfs[name] = FastStaTemplate._compute_entries(
        FastStaTemplate, [aligned_trace],
        dict(x_stimulus=x_stimulus, dims=dims, whitening_matrix=kw['whitening_matrix']), n_jobs=4)[0]['rf']
    print(f"{name}: {time.time() - t_start:.2f} s, contains NaN: {np.any(np.isnan(rfs[name]))}")

# %% [markdown]
# ## 4. GLM
#
# The GLM shares the trace preprocessing with the STA, then `RfGlmTemplate._compute_entries` fits a
# `ReceptiveFieldGLM` to every trace (in parallel processes for several traces).
# The RF is parameterised with splines (temporal `df_ts` x spatial `df_ws`) and regularised (`betas`);
# hyperparameters are selected by held-out score. It is much slower than the STA but gives smoother RFs.
#
# `_compute_entries` only needs the class, so it is called with the class as `self`.

# %%
glm_params = dict(
    df_ts=(5,), df_ws=(5,), betas=(0.001, 0.01, 0.1), kfold=1, metric='mse', output_nonlinearity='none',
    other_params_dict=RfGlmParamsTemplate.get_other_params_dict_default(dict(n_perm=5)))

sta_params = dict(
    x_stimulus=x_stimulus, rf_time=rf_time, burn_in=burn_in, shift=shift, dims=dims, dt=dt,
    fupsample_trace=1, fit_kind='trace', lowpass_cutoff=0, pre_blur_sigma_s=0, post_blur_sigma_s=0,
    glm_params=glm_params)

t_start = time.time()
aligned_trace = FastStaTemplate._prepare_trace(
    tracetime, trace, stimtime, x_stimulus.shape[0], fupsample_trace=1, fit_kind='trace', lowpass_cutoff=0,
    pre_blur_sigma_s=0, post_blur_sigma_s=0)
glm_entry = RfGlmTemplate._compute_entries(RfGlmTemplate, [aligned_trace], sta_params, n_jobs=1)[0]
print(f"GLM: {time.time() - t_start:.1f} s")
rfs['GLM'] = glm_entry['rf']

# %% [markdown]
# ## 5. Compare to the ground truth
#
# For each RF the plot shows the spatial RF at the time of the peak of the true RF (top) and the temporal RF
# at the pixel of the true spatial peak (bottom), each normalised to unit peak. `r` is the correlation with the
# true RF over all space and time.

# %%
def corr_with_truth(rf):
    return np.corrcoef(rf.ravel(), rf_true.ravel())[0, 1]


t_peak = np.argmax(np.abs(rf_true).max(axis=(1, 2)))
y_peak, x_peak = np.unravel_index(np.argmax(np.abs(rf_true[t_peak])), rf_true.shape[1:])

all_rfs = {'Ground truth': rf_true, **rfs}
fig, axs = plt.subplots(2, len(all_rfs), figsize=(3 * len(all_rfs), 5.5))
for i, (name, rf) in enumerate(all_rfs.items()):
    rf = rf / np.abs(rf).max()
    axs[0, i].imshow(rf[t_peak], cmap='bwr', vmin=-1, vmax=1)
    axs[0, i].set_title(name if name == 'Ground truth' else f"{name}\nr={corr_with_truth(rfs[name]):.2f}", fontsize=9)
    axs[0, i].axis('off')
    axs[1, i].plot(rf_time, rf[:, y_peak, x_peak])
    axs[1, i].axhline(0, c='k', lw=0.5)
    axs[1, i].set(xlabel='Lag (s)', ylim=(-1.1, 1.1))
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Full spatio-temporal RFs
#
# `plot1(kind='frames')` of both tables uses `plot_rf_frames`.

# %%
for name, rf in rfs.items():
    print(name)
    plot_rf_frames(rf, rf_time, downsample=2)
    plt.show()

# %% [markdown]
# ## 7. GLM summary and quality
#
# `RfGlmTemplate.plot1` shows the RF together with the training curves, quality metrics and predictions.

# %%
plot_rf_summary(rf=glm_entry['rf'], quality_dict=glm_entry['quality_dict'], model_dict=glm_entry['model_dict'],
                title='GLM')
plt.show()

print({k: v for k, v in glm_entry['quality_dict'].items() if np.ndim(v) == 0})
