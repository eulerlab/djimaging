import numpy as np

from djimaging.autorois.autorois_utils import make_mask_ids_consecutive

# Default MEDIARFormer arguments of the finetuned AutoROIs model (see config_init.yaml of the trained model).
DEFAULT_MODEL_ARGS = dict(
    classes=3,
    decoder_channels=[1024, 512, 256, 128, 64],
    decoder_pab_channels=256,
    encoder_name="mit_b5",
    in_channels=3,
)

# The model was trained on images resized to this size.
DEFAULT_IMAGE_SIZE = (512, 512)


class MediarWrapper:
    """Inference-only wrapper around a (finetuned) MEDIAR model for ROI mask creation.

    Requires the ``mediar`` package (https://github.com/sarmueller/MEDIAR), see the ``autorois-mediar`` extra.

    Parameters
    ----------
    model_path : str
        Path to the ``.pth`` file with the model weights.
    device : str, optional
        Torch device used for inference. Default is ``"cpu"``.
    model_args : dict, optional
        Keyword arguments of ``MEDIARFormer``. Defaults to :data:`DEFAULT_MODEL_ARGS`.
    algo_params : dict, optional
        ``algo_params`` of the MEDIAR ``Predictor``. Default is ``{"use_tta": False}``.
    image_size : tuple of int, optional
        Size the average image is resized to before inference. Default is (512, 512).
    """

    def __init__(
            self,
            model_path: str,
            device: str = "cpu",
            model_args: dict | None = None,
            algo_params: dict | None = None,
            image_size: tuple = DEFAULT_IMAGE_SIZE,
    ) -> None:
        import torch
        from mediar.core.MEDIAR import Predictor
        from mediar.train_tools.models import MEDIARFormer

        self.device = device
        self.image_size = tuple(image_size)

        self.model = MEDIARFormer(**(DEFAULT_MODEL_ARGS if model_args is None else model_args))
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)
        self.model.to(device)
        self.model.eval()

        self.predictor = Predictor(
            self.model,
            device=device,
            input_path="./",
            output_path="./",
            algo_params={"use_tta": False} if algo_params is None else algo_params,
        )

    @staticmethod
    def stack_to_image(stack: np.ndarray) -> np.ndarray:
        """Average a (nx, ny, nt) stack over time and normalize it to [0, 1], as done in training."""
        img = stack.mean(axis=-1).astype(np.float32) if stack.ndim > 2 else stack.astype(np.float32)
        img_min, img_max = img.min(), img.max()
        img = img - img_min
        if img_max - img_min > 1e-30:
            img = img / (img_max - img_min)
        return img

    def create_mask_from_data(
            self,
            ch0_stack: np.ndarray,
            ch1_stack: np.ndarray | None = None,
            multiple_stacks: bool = False,
            **kwargs,
    ) -> np.ndarray:
        """Create a ROI mask from the channel 0 stack.

        Parameters
        ----------
        ch0_stack : np.ndarray
            Stack of shape (nx, ny, nt), or a list of such stacks if ``multiple_stacks``.
        ch1_stack : np.ndarray, optional
            Ignored, only for compatibility with the other AutoROIs models.
        multiple_stacks : bool, optional
            If ``True``, ``ch0_stack`` is a list of stacks, which are concatenated in time.
        **kwargs
            Ignored (e.g. ``n_artifact``, ``pixel_size_um``), only for compatibility with the other models.

        Returns
        -------
        np.ndarray
            Integer 2-D ROI mask of shape (nx, ny), where 0 is background and positive integers are ROI IDs.
        """
        import torch
        import torchvision
        from torchvision.transforms import InterpolationMode

        if multiple_stacks:
            ch0_stack = np.concatenate(ch0_stack, axis=-1)

        img = self.stack_to_image(ch0_stack)
        shape = img.shape

        img_t = torch.from_numpy(img)[None, None]  # (1, 1, nx, ny)
        img_t = torchvision.transforms.Resize(
            size=self.image_size, interpolation=InterpolationMode.BILINEAR)(img_t)
        img_t = img_t.repeat(1, 3, 1, 1)  # The network expects three channels

        with torch.no_grad():
            pred = self.predictor._inference(img_t)
        pred = pred.cpu().numpy() if hasattr(pred, 'cpu') else np.asarray(pred)
        mask = np.asarray(self.predictor._post_process(pred[0]), dtype=np.int32)

        # Back to the original resolution
        mask_t = torch.from_numpy(mask.astype(np.int32))[None, None].to(torch.float32)
        mask_t = torchvision.transforms.Resize(size=shape, interpolation=InterpolationMode.NEAREST)(mask_t)
        mask = mask_t[0, 0].numpy().astype(np.int32)

        return make_mask_ids_consecutive(mask)
