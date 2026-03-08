import warnings
from typing import List, Dict, Optional

import numpy as np
import pytorch_lightning as pl

try:
    import torch
except ImportError:
    warnings.warn("Could not import torch, AutoROIs will not work.")

try:
    import yaml
except ImportError:
    warnings.warn("Could not import torch, AutoROIs will not work.")

from djimaging.autorois.autorois_utils import create_mask


def _normalize_image(x: np.ndarray, n_artifact: int = 0) -> np.ndarray:
    """Normalize a 2-D image to the range [0, 1] while ignoring artifact rows.

    Parameters
    ----------
    x : np.ndarray
        2-D image array to normalize.
    n_artifact : int, optional
        Number of rows at the top of the image to treat as artifacts; these
        rows are set to 0 after normalization. Default is 0.

    Returns
    -------
    np.ndarray
        Normalized image with values in [0, 1] and artifact rows set to 0.
    """
    x_min = x[n_artifact:, :].min()
    x_max = x[n_artifact:, :].max()
    x_rng = x_max - x_min
    x_normalized = x - x_min
    if x_rng > 0.:
        x_normalized /= x_rng
    x_normalized[:n_artifact, :] = 0.0
    return x_normalized


class DoubleConv(torch.nn.Module):
    """Two consecutive 3x3 convolutions each followed by BatchNorm and ReLU.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    mid_channels : int or None, optional
        Number of channels after the first convolution. Defaults to
        ``out_channels`` when ``None``. Default is ``None``.
    """

    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None) -> None:
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels, mid_channels, kernel_size=3, padding=1, bias=False
            ),
            torch.nn.BatchNorm2d(mid_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(
                mid_channels, out_channels, kernel_size=3, padding=1, bias=False
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Apply the double convolution block.

        Parameters
        ----------
        x : torch.Tensor
            Input feature map.

        Returns
        -------
        torch.Tensor
            Output feature map after two conv-BN-ReLU blocks.
        """
        return self.double_conv(x)


class Down(torch.nn.Module):
    """Downscaling block: 2x2 max-pool followed by a :class:`DoubleConv`.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.maxpool_conv = torch.nn.Sequential(
            torch.nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Apply max-pool and double convolution.

        Parameters
        ----------
        x : torch.Tensor
            Input feature map.

        Returns
        -------
        torch.Tensor
            Downscaled feature map.
        """
        return self.maxpool_conv(x)


class Up(torch.nn.Module):
    """Upscaling block: transposed convolution, skip-connection concatenation, then :class:`DoubleConv`.

    Parameters
    ----------
    in_channels : int
        Number of input channels (from the previous decoder stage).
    out_channels : int
        Number of output channels.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = torch.nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: "torch.Tensor", x2: "torch.Tensor") -> "torch.Tensor":
        """Apply upsampling, concatenate skip connection and apply double conv.

        Parameters
        ----------
        x1 : torch.Tensor
            Feature map from the decoder (to be upsampled).
        x2 : torch.Tensor
            Skip-connection feature map from the corresponding encoder stage.

        Returns
        -------
        torch.Tensor
            Upscaled and merged feature map.
        """
        x1 = self.up(x1)
        # Concatenate at channel dimension.
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(pl.LightningModule):
    """Instance-segmentation UNet with three output heads.

    Predicts a binary cell mask, per-pixel offsets to instance centres and a
    centre-probability map.  Post-processing via :func:`create_mask` converts
    these predictions into an integer ROI mask.

    Parameters
    ----------
    in_channels : int
        Number of input image channels.
    channels : list of int
        Channel counts for each encoder / decoder level.  The encoder has
        ``len(channels) - 1`` downsampling steps.
    dropout_probability : float, optional
        Dropout probability applied at the bottleneck and skip connections.
        Default is 0.0.
    """

    def __init__(
            self,
            in_channels: int,
            channels: List[int],
            dropout_probability: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels

        self.inc = DoubleConv(in_channels, channels[0])
        self.encoder_layers = torch.nn.ModuleList()
        self.decoder_layers = torch.nn.ModuleList()
        self.dropout_module = torch.nn.Dropout(dropout_probability)

        for i in range(len(channels) - 1):
            self.encoder_layers.append(Down(channels[i], channels[i + 1]))
            self.decoder_layers.append(Up(channels[i + 1], channels[i]))
        # Three heads for binary mask, offsets and center predictions.
        in_channels_final = channels[0]
        self.out_binary_mask = torch.nn.Conv2d(in_channels_final, 1, kernel_size=1)
        self.out_offsets = torch.nn.Conv2d(in_channels_final, 2, kernel_size=1)
        self.out_centers = torch.nn.Conv2d(in_channels_final, 1, kernel_size=1)

    def forward(
            self,
            x: "torch.Tensor",
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """Run a forward pass through the UNet.

        Parameters
        ----------
        x : torch.Tensor
            Batch of input images of shape (B, C, H, W).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            A tuple of ``(binary_mask_logits, offsets, center_logits)``, each
            of shape (B, *, H, W).
        """
        x = self.inc(x)
        shortcuts = []
        for encoder_layer in self.encoder_layers:
            shortcuts.append(x)
            x = encoder_layer(x)
        x = self.dropout_module(x)
        for decoder_layer, shortcut in zip(
                reversed(self.decoder_layers), reversed(shortcuts)
        ):
            shortcut = self.dropout_module(shortcut)
            x = decoder_layer(x, shortcut)
        binary_mask_logits_pred = self.out_binary_mask(x)
        offset_pred = self.out_offsets(x)
        center_pred = self.out_centers(x)
        return binary_mask_logits_pred, offset_pred, center_pred

    def create_mask_from_data_dict(self, data_dict: Dict) -> np.ndarray:
        """Create a ROI mask from a data dictionary loaded from a pickle file.

        Parameters
        ----------
        data_dict : dict
            Dictionary with at least the keys ``'ch0_stack'`` and
            ``'ch1_stack'``, and optionally ``'meta'`` containing
            ``'n_artifact'``.

        Returns
        -------
        np.ndarray
            Integer 2-D ROI mask.
        """
        return self.create_mask_from_data(
            ch0_stack=data_dict["ch0_stack"], ch1_stack=data_dict["ch1_stack"],
            n_artifact=data_dict.get("meta", {}).get("n_artifact", 0))

    def create_mask_from_data(
            self,
            ch0_stack: np.ndarray,
            ch1_stack: np.ndarray,
            n_artifact: int = 0,
            **kwargs,
    ) -> np.ndarray:
        """Create a ROI mask from two imaging channel stacks.

        Parameters
        ----------
        ch0_stack : np.ndarray
            Primary channel stack of shape (nx, ny, nt).
        ch1_stack : np.ndarray
            Secondary channel stack of shape (nx, ny, nt).
        n_artifact : int, optional
            Number of artifact rows to exclude. Default is 0.
        **kwargs
            Additional keyword arguments; ``'pixel_size_um'`` is silently
            ignored and any other keys trigger a warning.

        Returns
        -------
        np.ndarray
            Integer 2-D ROI mask.
        """
        kwargs.pop("pixel_size_um", None)
        if len(kwargs) > 0:
            warnings.warn(f'ignoring kwargs: {kwargs}')

        ch0_img = ch0_stack.mean(axis=-1)
        ch1_img = ch1_stack.mean(axis=-1)
        ch0_img_normalized = _normalize_image(ch0_img, n_artifact=n_artifact)
        ch1_img_normalized = _normalize_image(ch1_img, n_artifact=n_artifact)

        neural_input = np.stack([ch0_img_normalized, ch1_img_normalized])
        predicted_mask = self.create_mask_image_stack(neural_input)
        return predicted_mask

    def create_mask_image_stack(self, image_stack: np.ndarray) -> np.ndarray:
        """Run inference on a pre-stacked normalised image array.

        Parameters
        ----------
        image_stack : np.ndarray
            Array of shape (C, H, W) where C is the number of channels.

        Returns
        -------
        np.ndarray
            Integer 2-D ROI mask.
        """
        image_stack_torch = torch.from_numpy(image_stack).unsqueeze(0).to(torch.float32)
        binary_mask_logits_pred, offset_pred, center_pred = self.forward(image_stack_torch)
        binary_mask_pred = torch.sigmoid(binary_mask_logits_pred).squeeze(0)
        predicted_mask = create_mask(binary_mask_pred.squeeze(0), offset_pred.squeeze(0),
                                     center_pred.squeeze(0).squeeze(0))
        return predicted_mask.cpu().numpy()

    @classmethod
    def from_checkpoint(cls, config_path: str, checkpoint_path: str, map_location: str = "cpu") -> "UNet":
        """Load a :class:`UNet` model from a YAML config and a Lightning checkpoint.

        Parameters
        ----------
        config_path : str
            Path to the YAML configuration file containing a ``'network'`` key.
        checkpoint_path : str
            Path to the PyTorch Lightning checkpoint file.
        map_location : str, optional
            Device string passed to ``load_from_checkpoint``. Default is
            ``"cpu"``.

        Returns
        -------
        UNet
            Loaded model instance.
        """
        print(f"Load model weights for {map_location} from checkpoint {checkpoint_path} using config {config_path}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        network_config = config["network"]

        model = UNet.load_from_checkpoint(
            checkpoint_path,
            in_channels=network_config["in_channels"],
            channels=network_config["channels"],
            map_location=map_location,
        )
        return model
