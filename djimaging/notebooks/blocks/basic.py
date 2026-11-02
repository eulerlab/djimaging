# %% [markdown]
# # Basic response analysis
#
# Simple quality measures and response features computed from the snippets.
#
# - `ChirpQI`: Quality index (signal-to-noise of the response over repetitions) for chirp responses. Use it to
#   select ROIs that respond reliably.
# - `ChirpFeaturesBc` / `ChirpFeaturesRgc`: Features of the chirp response (e.g. transience, polarity),
#   for bipolar cells and retinal ganglion cells, respectively.
# - `OsDsIndexes`: Orientation- and direction-selectivity indexes from moving bar responses, with a shuffle test.

# %%
ChirpQI().populate(display_progress=True, processes=20)
ChirpQI()

# %%
ChirpFeaturesBc().populate(display_progress=True, processes=20)
ChirpFeaturesBc()

# %%
ChirpFeaturesRgc().populate(display_progress=True, processes=20)
ChirpFeaturesRgc()

# %%
OsDsIndexes().populate(display_progress=True, processes=20)
OsDsIndexes()
