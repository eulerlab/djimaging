# %% [markdown]
# # Recording location
#
# Where in the retina was each field recorded?
#
# - `OpticDisk`: Position of the optic disk of each experiment, used as reference point.
# - `RelativeFieldLocation`: Position of each field relative to the optic disk.
# - `RetinalFieldLocation`: Position of each field in retinal coordinates (e.g. ventral-dorsal and nasal-temporal),
#   taking the eye and orientation information of the experiment into account.

# %%
OpticDisk().populate(processes=20, display_progress=True)
OpticDisk()

# %%
RelativeFieldLocation().populate(processes=20, display_progress=True)
RelativeFieldLocation()

# %%
plot_if_data(RelativeFieldLocation(), method='plot')

# %%
RetinalFieldLocation().populate(processes=20, display_progress=True)
RetinalFieldLocation()

# %%
plot_if_data(RetinalFieldLocation(), method='plot')
