from typing import List, Tuple, Dict, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import yaml

from djimaging.autorois.post_processing import create_mask


def _normalize_image(x: np.array, n_rows_artifacts: int = 0) -> np.array:
    x_min = x[n_rows_artifacts:].min()
    x_max = x[n_rows_artifacts:].max()
    x_normalized = (x - x_min) / (x_max - x_min)
    x_normalized[:n_rows_artifacts] = 0.0
    return x_normalized


class DoubleConv(torch.nn.Module):
    """Two times 3x3 convolution, Batch Normalization and ReLU activation."""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
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

    def forward(self, x):
        return self.double_conv(x)


class Down(torch.nn.Module):
    """Downscaling with maxpool stride 2, then double conv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = torch.nn.Sequential(
            torch.nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(torch.nn.Module):
    """Upscaling, concatenate feature maps from encoder, then double conv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = torch.nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Concatenate at channel dimension.
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(pl.LightningModule):
    def __init__(
            self,
            in_channels: int,
            channels: List[int],
            dropout_probability: float = 0.0,
    ):
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

    def forward(self, x):
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

    def create_mask_from_data_dict(self, data_dict: Dict) -> np.array:
        """
        Create mask from data dict that is loaded from a pickle file

        Args:
            data_dict: Dictionary with the following keys: `ch0_stack`, `ch1_stack`, and optionally `meta.n_artifact`
        """
        n_artifact = data_dict.get("meta", {}).get("n_artifact", 0)
        ch0_img = data_dict["ch0_stack"].mean(axis=-1)
        ch1_img = data_dict["ch1_stack"].mean(axis=-1)
        ch0_img_normalized = _normalize_image(ch0_img, n_rows_artifacts=n_artifact)
        ch1_img_normalized = _normalize_image(ch1_img, n_rows_artifacts=n_artifact)

        neural_input = np.stack([ch0_img_normalized, ch1_img_normalized])
        predicted_mask = self.create_mask_image_stack(neural_input)
        return predicted_mask

    def create_mask_image_stack(self, image_stack: np.array) -> np.array:
        image_stack_torch = torch.from_numpy(image_stack).unsqueeze(0).to(torch.float32)
        binary_mask_logits_pred, offset_pred, center_pred = self.forward(image_stack_torch)
        binary_mask_pred = torch.sigmoid(binary_mask_logits_pred).squeeze(0)
        predicted_mask = create_mask(binary_mask_pred.squeeze(0), offset_pred.squeeze(0),
                                     center_pred.squeeze(0).squeeze(0))
        return predicted_mask.cpu().numpy()

    @classmethod
    def from_checkpoint(cls, config_path: str, checkpoint_path: str, map_location: str = "cpu"):
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
