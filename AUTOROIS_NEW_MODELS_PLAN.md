# Integrate retrained AutoROIs models into djimaging

Goal: djimaging offers the retrained UNet, a finetuned Cellpose3 model and a MEDIAR model as AutoROIs models.

Related repos:
- Development repo: `C:\Users\joesterle\github\jonathanoesterle\autorois`
- Public cleanup version: `C:\Users\joesterle\github\berenslab\autorois`
- Trained weights (READ ONLY for Claude): `Y:\data\autorois` (`database/models/`)

## Decisions

- Checkpoints: the 200-epoch, `train80` models only.
- Inference-only code is copied into djimaging; the public `autorois` package is not a dependency.
- The default model set is only the latest UNet (plus `CorrRoiMask`). Other models must be requested explicitly:
  `load_default_autorois_models(models=['UNet', 'Cellpose3_cyto2', 'Cellpose3_cyto3', 'MEDIAR'])`, or a list of names passed to `draw_roi_mask(autorois_models=[...])`.
  Explicitly requested models raise if they cannot be loaded; the default UNet only warns.
- Model names: `UNet` (latest, v0.2.0), `UNet_v1` (old v0.1.0), `Cellpose3_cyto2`, `Cellpose3_cyto3`, `MEDIAR`, `CorrRoiMask`.

## Known issues (decided to leave as is)

- `cellpose` and `mediar` are deliberately NOT dependencies of djimaging (no extras), the models stay available if the
  user installs the packages. `mediar` pins `monai==1.4.0` (needs `numpy<2`) and `cellpose` 3.x needs `numpy<2.1`
  (locking it downgrades `numpy` to 2.0.2 and `numba` to 0.60.0), which breaks `uv sync` with up-to-date `numpy`.
- Possible later fixes: relax the monai pin in the `sarmueller/MEDIAR` fork (`override-dependencies = ["monai>=1.5.0"]`
  made `uv lock` succeed, untested with the MEDIAR wrapper), and move to Cellpose 4 (check against the finetuned weights).

## What was implemented

- `djimaging/autorois/unet.py`: supports 1-channel (ch0 only) and 2-channel models; `ch1_stack` optional; `multiple_stacks` concatenates in time.
- `djimaging/autorois/cellpose_wrapper.py`: `round()` instead of `int()` when rescaling diameter / min_size by the pixel size.
- `djimaging/autorois/mediar_wrapper.py`: new, inference-only MEDIAR wrapper (resize to 512x512, predict, resize mask back, consecutive IDs).
- `djimaging/tables/core/roi_mask.py`: loaders for all models, `models=` argument, list support in `draw_roi_mask`.
  Cellpose3 uses `diameter=10` um, `min_size=4`, `flow_threshold=0.4`, `nchan=2` (train80: median ROI area 18 px, about 4.8 px diameter at about 2 um per pixel).
- Tests: `tests/autorois/test_new_models.py`, `tests/autorois/test_load_default_models.py` (12 pass, MEDIAR with a fake package).
- Checked: the real `_epoch=199.ckpt` UNet loads and predicts on `field_0001` (129 predicted vs 138 ground-truth ROIs; not a test-set score).

## TODOs

Jonathan:
- [ ] Copy the weights into the `reference` store (Claude has no write access to Y):
  - `Resources/AutoROIs/models/UNET_v0.2.0/config_init.yaml` and `_epoch=199.ckpt`
    (from `model_id-unet_200_epochs_pp_id-1_dataset_id-train80`; the checkpoints are in its `checkpoints/` folder)
  - `Resources/AutoROIs/models/CELLPOSE3_cyto2_v0.2.0/cellpose` and `.../CELLPOSE3_cyto3_v0.2.0/cellpose`
    (from `model_id-cyto3_200_epochs_pp_id-1_dataset_id-train80/models/`)
  - `Resources/AutoROIs/models/MEDIAR_v0.2.0/mediar_best_finetuned_model.pth`
    (from `model_id-mediar_200_epochs_pp_id-1_dataset_id-train80`)
- [ ] Recreate the broken project `.venv` (its base Python is missing), then run the full unit suite and flake8:
  `pytest -v tests --ignore=tests/integration`
- [ ] Test Cellpose3 with the real weights (install `.[autorois]` and `cellpose>=3.1.1,<4` yourself), e.g. in the tutorial notebook, and check the masks.
- [ ] Test MEDIAR with the real weights (install `.[autorois]` and `mediar` yourself). The wrapper mirrors the public repo's `Predictor` usage
  (`_inference`, `_post_process`) but was only tested against a fake package.
- [ ] Check that the 1-channel UNet preprocessing matches training on a few real fields, and compare against the ground truth on a held-out (test split) field.
- [ ] Check the Cellpose `min_size` handling: the wrapper divides `min_size` by the pixel size linearly, although it is an area in pixels.
- [ ] Decide whether `CorrRoiMask` should stay in the default set.
- [ ] Update the tutorial notebook (`djimaging/notebooks/tutorial_my_schema.ipynb`) if the new model selection should be shown.
- [ ] Open a PR against `master`.
