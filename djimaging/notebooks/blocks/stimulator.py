# %% [markdown]
# ## Setup tables
#
# Setup-dependent parameters, per setup and period, assigned to each experiment from its `setupid` and date. `Field`
# and `Presentation` take the stimulator delay, the scan width (pixel size) and the light artifact rows from here,
# so these tables must be populated before `Field`.
#
# - `Stimulator`: how the QDSpy screen is projected onto the retina (`screen_x_axis`, `screen_y_axis` in
#   XCoord/YCoord), offsets, color mapping, stimulator delay and the confidence of the geometry. `add_defaults` adds
#   setups 1 and 3 (geometry documented 2022 to 05/2026) and setup 2 (assumed to be like setup 1), one row per period
#   of constant stimulator delay. All stimulus geometry (moving bar directions, RF positions relative to ROIs, mouse
#   cam orientation) is taken from this table, see `djimaging.utils.stimulator`.
# - `ImagingSetup`: scan width at zoom 1 and light artifact rows.
# - `ExperimentStimulator`, `ExperimentImagingSetup`: the rows of each experiment, with a confidence ('high',
#   'medium', 'low') and the reason. Unassigned experiments are listed by `report_unassigned`; assign them with
#   `add_manual`, otherwise their fields cannot be added.

# %%
Stimulator().add_defaults()
ImagingSetup().add_defaults()
ExperimentStimulator().populate(display_progress=True)
ExperimentImagingSetup().populate(display_progress=True)
ExperimentStimulator().report()

# %%
ExperimentStimulator().report_unassigned()
