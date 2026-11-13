# %% [markdown]
# # Moving bar directions on the retina
#
# The moving bar directions of `trial_info` are script angles. Where the bar moves on the retina depends on the moving
# bar version (QDSpy md5), the stimulator and the eye and orientation of the retina (`Experiment.ExpInfo`).
#
# - `MovingBarVersion`: Known moving bar versions and their direction convention.
# - `MbDirectionMap`: Direction of every trial in the field and on the retina (0° = nasal, 90° = dorsal), per
#   presentation, with a confidence and the reason. Unknown versions get their convention from the script name
#   ('medium') or no retinal directions ('low').
# - `MbRetinalPrefDir`: Preferred direction (`OsDsIndexes.pref_dir`) on the retina, nearest cardinal direction, dorsal
#   component (also known without the eye) and, with `RetinalFieldLocation`, the angle relative to the optic disc.

# %%
MovingBarVersion().add_defaults()
MovingBarVersion()

# %%
MbDirectionMap().populate(display_progress=True)
MbDirectionMap().report()

# %%
plot_if_data(MbDirectionMap(), method='plot1')

# %%
MbRetinalPrefDir().populate(display_progress=True, processes=20)
MbRetinalPrefDir()

# %%
plot_if_data(MbRetinalPrefDir(), method='plot')
