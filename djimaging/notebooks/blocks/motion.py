# %% [markdown]
# # Motion correction and motion QC
#
# This schema is in **motion mode**: `RoiMask` only stores the field mask (drawn on one presentation, the motion
# reference). The ROI masks of all presentations are computed from the data instead of being drawn or shifted by
# hand, and `Traces` are motion corrected. Method and validation: `docs/motion_qc.md`.
#
# - `MotionParams`: Parameters (defaults in `motion_correction.DEFAULT_MOTION_PARAMS`).
# - `MotionReference`: Reference images of a field, from the presentation the field ROI mask was drawn on.
# - `MotionDetection`: Per presentation: rigid subpixel displacement vs. the reference (only the slow part, `tau_slow_s`,
#   is corrected; faster motion is flagged), a relative z proxy (similarity of the structure after removing xy motion,
#   from the functional and the structural channel, e.g. SR-101), and per-ROI displacements.
# - `MotionRoiMask`: ROI masks of each presentation: the field mask with a per-ROI shift (pooled over the
#   presentations of the same condition, `pooling='cond1'`) and a mild border reshape, as subpixel weights.
#   Flags ROIs with low contrast, inconsistent shifts or shifts at the search limit.
# - `MotionTrust`: Trust of each ROI trace per time bin (~2 s) and per stimulus repeat, from structure only
#   (local shape similarity, residual displacement, uncorrected fast motion, z drift, coverage).
#
# Populate `MotionDetection` for all presentations before `MotionRoiMask`, because the per-ROI shifts are pooled over
# the presentations of a condition. If presentations are added later, repopulate `MotionRoiMask` (and downstream) for
# that field and condition.

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
