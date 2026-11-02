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
# # AutoROIs tutorial (without DataJoint)
#
# This notebook
# 1. loads a single scan field from a ScanM file,
# 2. plots its average images and the ROI mask drawn by a human, and
# 3. runs all available AutoROIs models on the field and compares them to the human ROIs.
#
# It does not need a database. Copy it to your own project folder and adjust the paths below.
#
# Install the optional dependencies first:
# ```bash
# uv pip install -e ".[autorois]"
# uv pip install "cellpose>=3.1.1,<4"  # optional, for the Cellpose3 models
# uv pip install git+https://github.com/sarmueller/MEDIAR.git  # optional, for MEDIAR
# ```
# Models whose dependencies or weights are missing are skipped with a message.
#
# Note: `cellpose` and `mediar` are not installed with djimaging because they conflict with the up-to-date `numpy`
# (`mediar` pins `monai==1.4.0`, which needs `numpy<2`; `cellpose` 3.x needs `numpy<2.1`), see the README.
# `UNet`, `UNet_v1` and `CorrRoiMask` only need the extra `autorois`.

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import gc
import time

import numpy as np
from matplotlib import pyplot as plt

from djimaging.autorois import default_models
from djimaging.utils.mask_format_utils import as_python_format
from djimaging.utils.mask_utils import load_roi_mask_file
from djimaging.utils.plot_utils import plot_field, plot_roi_mask_boundaries
from djimaging.utils.scanm.recording import ScanMRecording

# %% [markdown]
# # Setup

# %%
# ScanM recording (.h5 or .smp) of one field. If the file contains a human ROI mask it is loaded with it.
DATA_FILE = '/path/to/field.h5'

# ID of the setup (1, 2 or 3). It is needed for the pixel size and the number of light-artifact lines.
SETUP_ID = 1

# Only needed if the ROI mask is not part of DATA_FILE, e.g. an Igor ROI file or a .npy/.pkl file.
ROI_MASK_FILE = None

# Folder with the model weights, with the same layout as `Resources/AutoROIs/models` in the reference store,
# see `djimaging.autorois.default_models`.
MODELS_DIR = '/path/to/Resources/AutoROIs/models'

# %% [markdown]
# # Load a field

# %%
rec = ScanMRecording(DATA_FILE, setup_id=SETUP_ID, triggers_ignore_not_found=True)
rec

# %%
ch0_stack = rec.ch_stacks['wDataCh0']
ch1_stack = rec.ch_stacks['wDataCh1']

n_artifact = rec.pix_n_artifact  # Rows affected by the light artifact of the stimulus
pixel_size_um = (rec.pix_dx_um, rec.pix_dx_um) if rec.scan_type == 'xy' else (rec.pix_dx_um, rec.pix_dz_um)

print(f'{ch0_stack.shape=}, {n_artifact=}, {pixel_size_um=}')

# %% [markdown]
# ## Average images and human ROI mask
#
# The AutoROIs models only see the stacks, never the human ROI mask.

# %%
roi_mask_human = rec.roi_mask
if roi_mask_human is None:
    assert ROI_MASK_FILE is not None, 'No ROI mask in DATA_FILE, set ROI_MASK_FILE'
    roi_mask_human = load_roi_mask_file(ROI_MASK_FILE)

ch0_avg = ch0_stack.mean(axis=-1)
ch1_avg = ch1_stack.mean(axis=-1)

# Plotting handles masks in Igor format (ROIs < 0, background = 1) and Python format (ROIs > 0, background = 0)
plot_field(ch0_avg, ch1_avg, scan_type=rec.scan_type, roi_mask=roi_mask_human, npixartifact=n_artifact,
           title=f'{rec.filename}: human ROIs');

# %%
# All models return masks in the Python format (background = 0, ROIs 1..n)
roi_mask_human = as_python_format(roi_mask_human)
print(f'Human ROIs: {len(np.unique(roi_mask_human)) - 1}')

# %% [markdown]
# # AutoROIs models
#
# Every model has the same interface, `model.create_mask_from_data(ch0_stack, ch1_stack, n_artifact, pixel_size_um)`,
# and returns an integer ROI mask.
#
# Large models are loaded one at a time and removed from memory right after they were used (see `run_model`).
# This is important for the neural networks, in particular if you use a GPU.
#
# | Model | Type | Input | Extra |
# |---|---|---|---|
# | `CorrRoiMask` | classical | ch0 stack | - |
# | `UNet_v1` | neural network | ch0 + ch1 averages | `autorois` |
# | `UNet` | neural network | ch0 average | `autorois` |
# | `Cellpose3_cyto2` | neural network | ch0 average | `cellpose` (separate) |
# | `Cellpose3_cyto3` | neural network | ch0 average | `cellpose` (separate) |
# | `MEDIAR` | neural network | ch0 average | `mediar` (separate) |

# %%
# Extent of the images, as in `plot_field`
if rec.scan_type == 'xy':
    EXTENT = (ch0_avg.shape[0] / 2, -ch0_avg.shape[0] / 2, ch0_avg.shape[1] / 2, -ch0_avg.shape[1] / 2)
else:
    EXTENT = (ch0_avg.shape[0] / 2, -ch0_avg.shape[0] / 2, -ch0_avg.shape[1] / 2, ch0_avg.shape[1] / 2)

autorois_masks = dict()  # Predicted masks of all models


def free_memory():
    """Release memory of models that are no longer referenced, also on the GPU."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def run_model(name, load_model):
    """Load a model, create a ROI mask for the field, and free the model again.

    The model only lives inside this function, so its memory can be released as soon as the prediction is done.
    """
    try:
        model = load_model()
    except (ImportError, FileNotFoundError, OSError) as e:
        print(f'Skipping {name}, could not load model: {e!r}')
        return

    try:
        t0 = time.time()
        mask = model.create_mask_from_data(
            ch0_stack=ch0_stack, ch1_stack=ch1_stack, n_artifact=n_artifact, pixel_size_um=pixel_size_um)
        dt = time.time() - t0
    finally:
        del model
        free_memory()

    mask = as_python_format(np.asarray(mask).astype(int))
    autorois_masks[name] = mask
    print(f'{name}: {len(np.unique(mask)) - 1} ROIs in {dt:.1f}s (human: {len(np.unique(roi_mask_human)) - 1})')
    plot_prediction(name, mask)


def plot_prediction(name, mask):
    """Plot the ROI boundaries of a prediction next to the human ROIs on the average image."""
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharex='all', sharey='all')
    for ax, (title, m) in zip(axs, [('human', roi_mask_human), (name, mask)]):
        ax.imshow(ch0_avg.T, origin='lower', extent=EXTENT, cmap='gray')
        plot_roi_mask_boundaries(ax, m, extent=EXTENT)
        ax.set(title=f'{title}: {len(np.unique(m)) - 1} ROIs', xlabel='relY [pixel]')
    axs[0].set(ylabel='relX [pixel]')
    plt.tight_layout()
    plt.show()


# %% [markdown]
# ## Correlation-based ROIs (`CorrRoiMask`)
#
# A classical method without learned weights that follows Zhao (2020). It computes the correlation of each pixel's
# time course with its neighbours. Pixels whose correlation exceeds a threshold are used as seeds, and neighbouring
# pixels are merged into the ROI as long as they stay sufficiently correlated and the ROI is not larger than the
# maximum area (in um^2, which is why it needs the pixel size). The parameters depend on the cell type
# (`default_rgc`, `default_bc`, `default_ac`). It is slow for large fields but needs no GPU and no training data.

# %%
run_model('CorrRoiMask', lambda: default_models.get_corr_roi_mask('default_rgc'))

# %% [markdown]
# ## UNet v1 (`UNet_v1`)
#
# The first U-Net (instance segmentation in the style of Panoptic-DeepLab). The network predicts three maps:
# a foreground probability, the offset of every foreground pixel to the centre of its ROI, and a centre heat map.
# Pixels are assigned to the closest predicted centre, which yields one label per ROI.
# This version takes the normalised average images of both channels (ch0 and ch1) as input.

# %%
run_model('UNet_v1', lambda: default_models.load_unet_v1(MODELS_DIR))

# %% [markdown]
# ## UNet (`UNet`, latest, v0.2.0)
#
# The same architecture, retrained on the larger dataset with the stack of ch0 only.
# This is the default model of `load_default_autorois_models`.

# %%
run_model('UNet', lambda: default_models.load_unet(MODELS_DIR))

# %% [markdown]
# ## Cellpose3 (`Cellpose3_cyto2`, `Cellpose3_cyto3`)
#
# [Cellpose](https://github.com/MouseLand/cellpose) is a general-purpose cell segmentation model that predicts
# flow fields, which point to the centre of each cell, and follows them to group pixels into cells.
# We finetuned Cellpose 3 (from the `cyto2` and `cyto3` base models) on our human ROI masks. It works on the average of ch0 and needs the
# expected cell diameter, which is set in um and converted to pixels using the pixel size.
# Requires `cellpose`, which has to be installed separately (see above).

# %%
run_model('Cellpose3_cyto2', lambda: default_models.load_cellpose3(MODELS_DIR, base='cyto2'))
run_model('Cellpose3_cyto3', lambda: default_models.load_cellpose3(MODELS_DIR, base='cyto3'))

# %% [markdown]
# ## MEDIAR (`MEDIAR`)
#
# [MEDIAR](https://github.com/Lee-Gihun/MEDIAR) is a cell instance segmentation model with a transformer encoder
# (MiT-b5) and a flow-based decoder; it was designed for the NeurIPS 2022 cell segmentation challenge.
# We finetuned it on our human ROI masks. The average image of ch0 is resized to 512x512 for the network, and the
# mask is resized back afterwards. It is the largest model (about 0.5 GB), so it is also the slowest and most memory
# hungry. Requires `mediar`, which has to be installed separately (see above).

# %%
run_model('MEDIAR', lambda: default_models.load_mediar(MODELS_DIR))

# %% [markdown]
# ## Summary

# %%
print(f'{"model":<12} {"n ROIs":>6}')
print(f'{"human":<12} {len(np.unique(roi_mask_human)) - 1:>6}')
for name, mask in autorois_masks.items():
    print(f'{name:<12} {len(np.unique(mask)) - 1:>6}')

# %% [markdown]
# # Use the models in the ROI canvas
#
# With a DataJoint pipeline, the models are used through `RoiMask().draw_roi_mask(autorois_models=...)`.
# By default only the latest UNet (and `CorrRoiMask`) is loaded. Other models have to be requested explicitly,
# which is only recommended if you want to compare them, as every model needs memory:
#
# ```python
# from djimaging.tables.core.roi_mask import load_default_autorois_models
#
# autorois_models = load_default_autorois_models()  # UNet and CorrRoiMask
# autorois_models = load_default_autorois_models(models=['UNet', 'Cellpose3_cyto3', 'MEDIAR'])
# ```
