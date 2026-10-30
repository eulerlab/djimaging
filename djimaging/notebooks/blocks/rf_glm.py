# %% [markdown]
# # Receptive fields: GLM
#
# RFs estimated with a regularized generalized linear model (GLM) with a spline-parameterized RF.
# It uses the same stimulus design matrix and trace alignment as the STA, but is more robust to noise and
# correlated stimuli. It is slower than the STA. Needs the `noise` stimulus.
#
# **Estimation**
# - `RfGlmParams`: Stimulus/trace preprocessing and GLM hyperparameters (spline sizes, regularization, ...).
# - `RfGlm`: The GLM RF of each ROI, including the fit quality on held-out data.
# - `RfGlmQualityParams` / `RfGlmQuality`: Thresholds and quality index to select ROIs with a reliable GLM RF.
#
# **Decomposition and fits**
# - `SplitRfGlmParams` / `SplitRfGlm`: Splits the GLM RF into a spatial and a temporal RF (SVD).
# - `FitGauss2DRfGlm`: Fits a 2D Gaussian to the spatial RF.
# - `FitDoG2DRfGlm`: Fits a 2D difference of Gaussians (center-surround) to the spatial RF.

# %%
RfGlmParams().add_default(skip_duplicates=True)
RfGlmParams()

# %%
# The ROIs of each field are fit in parallel, `n_jobs` sets the number of CPU cores.
RfGlm().populate(display_progress=True, n_jobs=30)
RfGlm()

# %%
plot_if_data(RfGlm(), key=None)

# %%
RfGlmQualityParams().add_default(skip_duplicates=True)
RfGlmQualityParams()

# %%
RfGlmQuality().populate(processes=20, display_progress=True)
RfGlmQuality()

# %%
plot_if_data(RfGlmQuality(), method='plot')

# %%
SplitRfGlmParams().add_default(skip_duplicates=True)
SplitRfGlmParams()

# %%
SplitRfGlm().populate(processes=20, display_progress=True)
SplitRfGlm()

# %%
plot_if_data(SplitRfGlm(), key=None)

# %%
FitGauss2DRfGlm().populate(processes=20, display_progress=True)
FitGauss2DRfGlm()

# %%
plot_if_data(FitGauss2DRfGlm(), key=None)

# %%
FitDoG2DRfGlm().populate(processes=20, display_progress=True)
FitDoG2DRfGlm()

# %%
plot_if_data(FitDoG2DRfGlm(), key=None)
