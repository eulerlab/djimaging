# %% [markdown]
# # Mouse cam geometry
#
# - `MouseCamGeometry`: orientation of mouse cam movies in the field (`to_field` reorients a QDSpy export movie) and
#   whether the movie version (from the QDSpy log) fits the setup, with a confidence and the reason.

# %%
MouseCamGeometry().populate(display_progress=True)
MouseCamGeometry()
