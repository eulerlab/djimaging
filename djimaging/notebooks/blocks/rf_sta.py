# %% [markdown]
# # Receptive fields: STA
#
# Spike-triggered average (STA) receptive fields (RFs) from noise responses. The trace (or its gradient) is
# aligned to the noise stimulus and correlated with it, optionally whitened. Needs the `noise` stimulus.
#
# **Estimation**
# - `FastStaParams`: Stimulus and trace preprocessing parameters. Stores the stimulus design matrix once per stimulus.
# - `FastSta`: The STA (spatio-temporal RF) of each ROI.
# - `FastStaQuality`: Signal-to-noise ratio of the STA, to select ROIs with a clear RF.
#
# **Decomposition and fits**
# - `SplitRFParams` / `SplitRF`: Splits the spatio-temporal RF into a spatial and a temporal RF (SVD).
# - `FitGauss2DRF`: Fits a 2D Gaussian to the spatial RF.
# - `FitDoG2DRF`: Fits a 2D difference of Gaussians (center-surround) to the spatial RF.

# %%
FastStaParams().add_default(skip_duplicates=True)
FastStaParams()

# %%
# The STAs of all ROIs of a field are computed at once, `n_jobs` sets the number of CPU cores.
FastSta().populate(display_progress=True, n_jobs=30)
FastSta()

# %%
plot_if_data(FastSta(), key=None)

# %%
FastStaQuality().populate(processes=20, display_progress=True)
FastStaQuality()

# %%
plot_if_data(FastStaQuality(), key=None)

# %%
SplitRFParams().add_default(skip_duplicates=True)
SplitRFParams()

# %%
SplitRF().populate(processes=20, display_progress=True)
SplitRF()

# %%
plot_if_data(SplitRF(), key=None)

# %%
FitGauss2DRF().populate(processes=20, display_progress=True)
FitGauss2DRF()

# %%
plot_if_data(FitGauss2DRF(), key=None)

# %%
FitDoG2DRF().populate(processes=20, display_progress=True)
FitDoG2DRF()

# %%
plot_if_data(FitDoG2DRF(), key=None)
