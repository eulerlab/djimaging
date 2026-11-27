# %% [markdown]
# # Inspect recordings (optional, read-only)
#
# Once ROI masks are loaded (`RoiMask`, see AutoROIs above) and `Traces` are populated, you can play back each
# presentation of a field with the ROIs of *its own* mask (`RoiMask.RoiMaskPresentation`, so shifted masks are shown
# where the traces were extracted) and follow the traces in sync with the movie:
#
# - choose the presentation (stimulus and conditions) and the functional channel; keys, paths and diagnostics are
#   in the collapsed *Details* section,
# - **Recording resolution** plays the functional movie; **HighRes** shows the `HighRes` image with the ROIs
#   registered by image correlation (the HighRes scan is not always centred like the functional scan; if the
#   registration cannot be verified, the HighRes view is disabled and the reason is shown),
# - **All eligible ROIs** (heatmap of the population) or **Selected ROIs** (click a cell with `%matplotlib widget`,
#   or use the list),
# - quality filter for chirp (`ChirpQI.qidx`) and moving bar (`OsDsIndexes.d_qi`), if these tables are in your schema
#   and populated; ROIs without a value are counted separately from ROIs below the threshold,
# - *Trial mean ± SD* next to the full trace: repeats cut like `Snippets` (same `_pad_trace`, trigger delay), mean ± SD
#   across valid repeats revealed up to the current trial time; moving bar: only repeats of the current direction,
# - MP4 export of the current view (needs `imageio-ffmpeg`, written to `~/djimaging_inspector_exports` by default).
#
# Nothing is written to the database. Needs `ipywidgets` (`uv pip install -e ".[inspector]"`).
# The same viewer works without database on local files, see `djimaging/notebooks/tutorial_recording_inspector.py`.

# %%
INSPECT_RECORDINGS = False  # Set to True to open the inspector

if INSPECT_RECORDINGS and len(Traces()) > 0:
    from djimaging.inspector import open_recording_inspector

    # Optional: run `%matplotlib widget` (needs ipympl) before this cell to click ROIs in the figure
    # field_key=None takes the first field with traces; set raw_id / preprocess_id if you have several parameter sets
    inspector = open_recording_inspector(globals(), field_key=None)
    display(inspector)
else:
    print("Recording inspector not opened: set INSPECT_RECORDINGS = True (requires populated Traces).")
