"""Builders for the default AutoROIs models.

These only need a local folder with the model weights (no DataJoint). The folder has the same layout as
``Resources/AutoROIs/models`` in the ``reference`` store:

    <models_dir>/UNET_v0.1.0/{sd_images.yaml, dropout_and_aug_regul.ckpt}
    <models_dir>/UNET_v0.2.0/{config_init.yaml, _epoch=199.ckpt}
    <models_dir>/CELLPOSE3_v0.2.0/cellpose
    <models_dir>/MEDIAR_v0.2.0/mediar_best_finetuned_model.pth
"""
from pathlib import Path

UNET_V1_FILES = ("UNET_v0.1.0/sd_images.yaml", "UNET_v0.1.0/dropout_and_aug_regul.ckpt")
UNET_FILES = ("UNET_v0.2.0/config_init.yaml", "UNET_v0.2.0/_epoch=199.ckpt")
CELLPOSE3_FILE = "CELLPOSE3_v0.2.0/cellpose"
MEDIAR_FILE = "MEDIAR_v0.2.0/mediar_best_finetuned_model.pth"


def load_unet_v1(models_dir: str | Path):
    """Load the first UNet (v0.1.0, two input channels)."""
    from djimaging.autorois.unet import UNet
    config_path, checkpoint_path = (Path(models_dir) / f for f in UNET_V1_FILES)
    return UNet.from_checkpoint(config_path, checkpoint_path)


def load_unet(models_dir: str | Path):
    """Load the latest, retrained UNet (v0.2.0, one input channel)."""
    from djimaging.autorois.unet import UNet
    config_path, checkpoint_path = (Path(models_dir) / f for f in UNET_FILES)
    return UNet.from_checkpoint(config_path, checkpoint_path)


def load_cellpose3(models_dir: str | Path):
    """Load the finetuned Cellpose3 model. Requires the ``autorois-cellpose`` extra."""
    from djimaging.autorois.cellpose_wrapper import CellposeWrapper

    # Diameter and min_size are given in um (and rescaled with the pixel size), the model was
    # trained on data with ~2 um pixels and a median ROI diameter of ~4.8 pixels.
    init_params = dict(pretrained_model=str(Path(models_dir) / CELLPOSE3_FILE), gpu=False, nchan=2)
    eval_params = dict(min_size=4, diameter=10, channels=[0, 0], flow_threshold=0.4)
    return CellposeWrapper(init_kwargs=init_params, eval_kwargs=eval_params)


def load_mediar(models_dir: str | Path, device: str = 'cpu'):
    """Load the finetuned MEDIAR model. Requires the ``autorois-mediar`` extra."""
    from djimaging.autorois.mediar_wrapper import MediarWrapper
    return MediarWrapper(model_path=str(Path(models_dir) / MEDIAR_FILE), device=device)


def get_corr_roi_mask(kind: str = 'default_rgc'):
    """Create the correlation-based ROI mask model with the default parameters for a cell type.

    Args:
        kind: 'default_rgc', 'default_bc' or 'default_ac'.
    """
    from djimaging.autorois.corr_roi_mask_utils import CorrRoiMask

    if kind == 'default_rgc':
        kws = dict(cut_x=(0, 0), cut_z=(0, 0), min_area_um2=10, max_area_um2=400, n_pix_max=None, line_threshold_q=70,
                   use_ch0_stack=True, grow_use_corr_map=False, grow_threshold=0.1,
                   grow_only_local_thresh_pixels=True)
    elif kind == 'default_bc' or kind == 'default_ac':
        kws = dict(cut_x=(4, 2), cut_z=(8, 2), min_area_um2=0.5, max_area_um2=12.6, n_pix_max=None, line_threshold_q=70,
                   use_ch0_stack=True, grow_use_corr_map=False, grow_threshold=None, line_threshold_min=0.1,
                   grow_only_local_thresh_pixels=True)
    else:
        raise NotImplementedError(kind)

    return CorrRoiMask(**kws)
