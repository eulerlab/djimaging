# %% [markdown]
# # HighRes scans and their alignment to the ROI mask
#
# - `HighRes`: high-resolution scan of each field (e.g. 512 x 512, usually recorded at the end of the stimulus
#   session), with the channel medians.
# - `HighResAlignment`: shift of the HighRes scan relative to the field ROI mask, and how well it matches after the
#   shift (`match_ch0` / `match_ch1`: noise-corrected similarity, 1 = same structure; `dz_um_approx`: rough z offset,
#   < ~2 um is the same plane within the precision). See docs/motion_qc.md ("HighRes alignment").

# %%
HighRes().populate(processes=20, display_progress=True)
HighRes()

# %%
plot_if_data(HighRes())

# %%
HighResAlignment().populate(processes=20, display_progress=True)
HighResAlignment()

# %%
plot_if_data(HighResAlignment())

# %%
plot_if_data(HighResAlignment(), method='plot')
