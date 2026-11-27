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
# # Recording inspector (without DataJoint)
#
# Play back the presentations of one field with their saved ROI masks (e.g. from AutoROIs) and inspect the traces of
# individual cells in sync with the movie, directly from the ScanM files of one experiment folder:
#
# ```
# <experiment>/            e.g. .../20260916/1
#     *.ini                setup id and date
#     Raw/*.smp, *.smh     (or Pre/*.h5)
#     AutoROIs/*_ROIs.npy  (or .pkl) masks saved by the ROI canvas
# ```
#
# No database connection, schema or AutoROIs model is needed. Traces, triggers and quality indexes are recomputed with
# djimaging's own functions and the default parameters of the example schema; they are labelled as offline values in
# the viewer. Stimulus identity is inferred from the file names (default `Stimulus` aliases), which is reported for
# files that do not match.
#
# Install the optional dependencies first: `uv pip install -e ".[inspector]"` (ipywidgets, imageio-ffmpeg, ipympl).
#
# In the pipeline, use the same viewer on stored results with `open_recording_inspector(globals(), field_key)`
# (section "Inspect recordings" of the generated populate notebook).

# %%
# Optional: clickable figure (needs ipympl); without it, ROIs are selected in the list.
# # %matplotlib widget

# %%
from pathlib import Path

from djimaging.inspector import LocalFieldSource, RecordingInspector

# %% [markdown]
# # Setup

# %%
EXPERIMENT_DIR = Path('/path/to/20260916/1')  # experiment folder (read-only, nothing is written there)
FIELD = 'GCL0'                                # field name as in the file names
EXPORT_DIR = Path.home() / 'djimaging_inspector_exports'  # MP4 exports (must be outside the data folder)

# %%
source = LocalFieldSource(EXPERIMENT_DIR, field=FIELD, roi_mask_dir='AutoROIs')
for info in source.list_presentations():
    print(f"{info.name:40s} channels={info.channels} mask={info.has_mask}", *info.notes, sep='\n    ')

# %% [markdown]
# # Inspect
#
# - Stimulus / Channel: each file of the field is one entry; files without a mask play without ROIs.
# - Recording / HighRes: the HighRes view registers the ROIs onto the HighRes image by image correlation; if it cannot
#   be verified, the view stays at recording resolution (the diagnostics are in the collapsed *Details* section).
# - All eligible ROIs / Selected ROIs. The QI filter (chirp: ChirpQI.qidx, moving bar: OsDsIndexes.d_qi; off at start),
#   its threshold and the number of eligible ROIs sit directly above the ROI list.
# - Bottom row: *Response trace* on the recording clock (~60 %) and *Trial mean ± SD* (~40 %): the repeats are cut
#   like `Snippets` (`split_trace_by_reps`), the mean ± SD across valid repeats is revealed up to the current time
#   within the current trial; for the moving bar only the repeats of the current direction are summarised. In
#   *All eligible ROIs* mode the ROIs are averaged within each repeat first. With one repeat there is no SD band.
# - play / scrub / speed (playback follows the recording clock; frames are skipped if rendering is too slow),
#   MP4 export of the current view (incl. the trial panel) with its own speed.
# - `inspector.playback_stats()` reports the measured per-frame times; `width=` sets the figure width in pixels.

# %%
inspector = RecordingInspector(source, export_dir=EXPORT_DIR)
inspector

# %% [markdown]
# The state can also be set from code, e.g. to reproduce a view or an export:

# %%
# inspector.select_presentation(inspector.pres.info.name, channel='wDataCh1')
# inspector.select_rois([3, 7, 12])
# inspector.set_threshold(0.35)   # None turns the filter off
# inspector.set_view('highres')
# inspector.set_time(60.0)
# inspector.export(t_start=55, t_stop=75, speed=1.0)
