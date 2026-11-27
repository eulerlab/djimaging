# %% [markdown]
# ## Optional: refined field ROI mask
#
# `RefinedRoiMask`: the field mask with its ROI borders refined (by at most `refine_max_px` = 1 px, in or out) on the
# registered mean of all presentations of the reference condition (more signal, no motion blur). ROI ids are kept;
# ROIs are never added, removed or merged. Dim ROIs and ROIs that would shrink too much are kept as drawn (`status`).
# `MotionRoiMask` and `MotionTrust` then start from the refined mask. Populate `MotionDetection` for all presentations
# of the reference condition first. `refine_mask=False` in `MotionParams` passes the drawn mask through.
#
# On the tested data this changed QI and the neuropil correlation of the traces by less than 1.5%
# (`docs/motion_qc.md`), so it is optional.

# %%
RefinedRoiMask().populate(display_progress=True, processes=20)
RefinedRoiMask()

# %%
plot_if_data(RefinedRoiMask())
