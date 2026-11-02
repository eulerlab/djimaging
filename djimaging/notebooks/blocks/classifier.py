# %% [markdown]
# # RGC classifier
#
# Assigns retinal ganglion cell types to ROIs using the classifier of Baden et al. (2016).
# It requires chirp and moving bar recordings. Skip this block if you don't want to use the RGC classifier.
#
# - `ClassifierV2`: The stored (pre-trained) classifier.
# - `Baden16TracesV2`: Chirp and moving bar responses of each ROI, processed to match the format that the classifier expects.
# - `CelltypeAssignmentV2`: Predicted cell type (cluster, group) and probability for each ROI.

# %%
classifier_id = 1
classifier_file = reference_root / 'Resources/Classifier_v2/rgc_classifier_v2_sklearn_1_7.pkl'
if len(ClassifierV2() & {'classifier_id': classifier_id}) == 0:
    ClassifierV2().add(classifier_file=classifier_file, classifier_id=classifier_id)
ClassifierV2()

# %%
Baden16TracesV2().populate(display_progress=True, processes=20)
Baden16TracesV2()

# %%
CelltypeAssignmentV2().populate(display_progress=True)
CelltypeAssignmentV2()

# %%
plot_if_data(
    CelltypeAssignmentV2(), method='plot_group_traces',
    cluster_id=None, cluster_name=None, group_id=5, min_prob=0.5, xlim_chirp=None, xlim_bar=None, plot_baden_data=True)

# %%
plot_if_data(CelltypeAssignmentV2(), method='plot_type_distribution', min_prob=0., level='cluster', plot_baden_data=True)
