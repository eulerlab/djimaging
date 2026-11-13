# %% [markdown]
# # Motion correction and motion QC
#
# This schema is in **motion mode**: `RoiMask` only stores the field mask (drawn on one presentation, the motion
# reference). The ROI masks of all presentations are computed from the data instead of being drawn or shifted by
# hand, and `Traces` are motion corrected. Method and validation: `docs/motion_qc.md`.
#
# A field has one ROI mask, saved in a **field mask file** `<stem>_FieldROIs.npy` (in `roi_mask_dir`) named after
# the presentation it belongs to. Presentation mask files (`<stem>_ROIs.npy`) and Igor masks are not written in motion
# mode; a copy for a shifted presentation would be wrong if loaded in a schema without motion mode.
#
# Masks without field mask file (e.g. Igor masks copied to all presentations) do not tell which presentation they were
# drawn on, so `RoiMask` chooses it from the data: the presentation whose mean image fits the mask best (presentations
# with a `Stack_ave` and then `mask_alias` win ties), and aligns the mask to it by up to 2 px (`align_dx`, `align_dy`;
# the mask file is not changed). Check `mask_fit` and `mask_selection` ('alias': the mask fits no presentation;
# 'file': from a field mask file, not aligned).
#
# In the ROI canvas (`RoiMask().draw_roi_mask(...)`), the mask is drawn and edited on the presentation it belongs to.
# The other presentations are view only: they show the presentation masks of `MotionRoiMask` (if populated, pass
# `motion_roi_mask_table=MotionRoiMask`, and as long as the mask is not edited), else a preview (the mask moved by the
# global shift between the mean images). 'Save all to file' writes the field mask file; deploy it with
# `deploy_roi_mask`. The presentation masks are computed by `MotionRoiMask` after deploying.
#
# - `MotionParams`: Parameters (defaults in `motion_correction.DEFAULT_MOTION_PARAMS`).
# - `MotionReference`: Reference images of a field, from the presentation the field ROI mask was drawn on.
# - `MotionDetection`: Per presentation: rigid subpixel displacement vs. the reference (only the slow part, `tau_slow_s`,
#   is corrected; faster motion is flagged), a relative z proxy (similarity of the structure after removing xy motion,
#   from the functional and the structural channel, e.g. SR-101), and per-ROI displacements.
# - `MotionRoiMask`: ROI masks of each presentation: the field mask with a per-ROI shift (pooled over the
#   presentations of the same condition, `pooling='cond1'`) and a mild border reshape, as subpixel weights.
#   Flags ROIs with low contrast, inconsistent shifts or shifts at the search limit.
# - `MotionTrust`: Trust of each ROI trace per time bin (~2 s) and per stimulus repeat: the fraction of the ROI that
#   is on its cell (from the residual displacement of the ROI, the uncorrected fast motion and the FOV coverage), and
#   per time bin the approximate z change of the field in um (`dz_um`, structural channel). Exclude ROIs with
#   `trust_median < trust_thresh`, or single repeats with `MotionTrust().valid_repeats(key)`.
#
# Motion detection and QC are **experimental**: validated on a limited set of recordings (docs/motion_qc.md); check
# the plots of your data before relying on them.
#
# Populate `MotionDetection` for all presentations before `MotionRoiMask`, because the per-ROI shifts are pooled over
# the presentations of a condition. If presentations are added later, repopulate `MotionRoiMask` (and downstream) for
# that field and condition.

# %% [markdown]
# Optional: save the prepared field masks (chosen presentation, aligned) to field mask files in a separate directory,
# e.g. to review or edit them in the ROI canvas. The original files are kept. With `update_database=True` the database
# entry references the new file and `align_dx = align_dy = 0`; the mask itself is unchanged, so nothing downstream is
# deleted. Load edited files with `RoiMask().deploy_roi_mask(field_key, roi_table=Roi, roi_mask_dir='PreparedROIs')`.
# A file with a different mask (e.g. edited) is only replaced with `overwrite_files=True`.

# %%
# export = RoiMask().export_field_roi_masks(roi_mask_dir='PreparedROIs', update_database=True, overwrite_files=False)

# %%
# Review / edit a field mask in the ROI canvas (other presentations show the MotionRoiMask masks once populated)
# roi_canvas = RoiMask().draw_roi_mask(field_key=..., roi_mask_dir='PreparedROIs', roi_table=Roi,
#                                      motion_roi_mask_table=MotionRoiMask)
# roi_canvas.start_gui()

# %%
MotionParams().add(motion_params_id=1, skip_duplicates=True)
MotionParams()

# %%
MotionReference().populate(display_progress=True)
MotionReference()

# %%
plot_if_data(MotionReference())

# %%
MotionDetection().populate(display_progress=True, processes=20)
MotionDetection()

# %%
plot_if_data(MotionDetection())

# %%
MotionRoiMask().populate(display_progress=True, processes=20)
MotionRoiMask()

# %%
plot_if_data(MotionRoiMask())

# %%
MotionTrust().populate(display_progress=True, processes=20)
MotionTrust()

# %%
plot_if_data(MotionTrust())

# %%
# Repeats to keep per ROI (thresholds: MotionParams trust_thresh / dz_thresh_um, or pass them explicitly)
# MotionTrust().valid_repeats(key, trust_thresh=0.7, dz_thresh_um=3.)
