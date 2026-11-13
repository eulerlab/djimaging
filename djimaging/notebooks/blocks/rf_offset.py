# %% [markdown]
# # Receptive field position relative to the ROI
#
# - `RelativeRoiLocationWrtField`: Position of each ROI relative to the field center (XCoord/YCoord).
# - `RfOffset`: RF center relative to the stimulus center, in stimulus array coordinates.
# - `RfRoiOffset`: RF center minus ROI position in the field, using only the stimulator of the experiment and the
#   `array_to_screen` axes of the noise stimulus. `geometry_confidence` and `geometry_reason` flag uncertain cases.

# %%
RelativeRoiLocationWrtField().populate(display_progress=True, processes=20)
RfOffset().populate(display_progress=True, processes=20)
RfRoiOffset().populate(display_progress=True, processes=20)
RfRoiOffset()

# %%
plot_if_data(RfRoiOffset(), method='plot')
