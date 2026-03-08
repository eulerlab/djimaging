from typing import Callable

import numpy as np
from matplotlib import pyplot as plt


class CellposeWrapper:
    """Thin wrapper around the Cellpose model for ROI mask creation.

    Parameters
    ----------
    init_kwargs : dict
        Keyword arguments passed to ``cellpose.models.Cellpose.__init__``.
    eval_kwargs : dict
        Keyword arguments passed to ``cellpose.models.Cellpose.eval``.
    """

    def __init__(self, init_kwargs: dict, eval_kwargs: dict) -> None:
        """Initialise the wrapper and load the Cellpose model.

        Parameters
        ----------
        init_kwargs : dict
            Keyword arguments forwarded to ``cellpose.models.Cellpose``.
        eval_kwargs : dict
            Keyword arguments forwarded to ``model.eval``.
        """
        self.init_kwargs = init_kwargs
        self.eval_kwargs = eval_kwargs

        from cellpose import models, io
        io.logger_setup()
        self.model = models.Cellpose(**self.init_kwargs)

    @staticmethod
    def stack_to_image(stack: np.ndarray, n_artifact: int) -> np.ndarray:
        """Convert a stack to a 2-D image by averaging over time and zeroing artifacts.

        Parameters
        ----------
        stack : np.ndarray
            3-D array with shape (nx, ny, nt) or 2-D array with shape (nx, ny).
        n_artifact : int
            Number of artifact rows at the top of the image to set to the
            minimum value.

        Returns
        -------
        np.ndarray
            2-D image of shape (nx, ny).
        """
        img = stack.mean(axis=2) if stack.ndim > 2 else stack.copy()
        img[:n_artifact] = np.min(img)
        return img

    def create_mask_from_data(
            self,
            ch0_stack: np.ndarray,
            ch1_stack: np.ndarray | None = None,
            use_ch0: bool = True,
            do_3D: bool = False,
            n_artifact: int = 0,
            pixel_size_um: tuple = (1., 1.),
            multiple_stacks: bool = False,
            plot: bool = False,
            **kwargs,
    ) -> np.ndarray:
        """Create a ROI mask from imaging stack data using Cellpose.

        Parameters
        ----------
        ch0_stack : np.ndarray
            Primary channel stack of shape (nx, ny, nt) or a list of such
            stacks when ``multiple_stacks=True``.
        ch1_stack : np.ndarray or None, optional
            Secondary channel stack. Used when ``use_ch0=False``. Default is
            ``None``.
        use_ch0 : bool, optional
            If ``True``, use ``ch0_stack``; otherwise use ``ch1_stack``.
            Default is ``True``.
        do_3D : bool, optional
            If ``True``, run 3-D segmentation. Currently not implemented.
            Default is ``False``.
        n_artifact : int, optional
            Number of artifact rows to zero out in the projected image.
            Default is 0.
        pixel_size_um : tuple, optional
            Pixel size in microns ``(dx, dy)`` or ``(dx, dy, dz)`` for 3-D.
            Used to rescale ``diameter`` and ``min_size`` in ``eval_kwargs``.
            Default is ``(1., 1.)``.
        multiple_stacks : bool, optional
            If ``True``, treat ``ch0_stack`` as a list of stacks and stitch
            results. Default is ``False``.
        plot : bool, optional
            If ``True``, display segmentation results via
            :meth:`plot_results`. Default is ``False``.
        **kwargs
            Additional keyword arguments (not forwarded, reserved for API
            compatibility).

        Returns
        -------
        np.ndarray
            Integer 2-D ROI mask where 0 is background and positive integers
            are ROI IDs.

        Raises
        ------
        NotImplementedError
            If ``do_3D=True``.
        ValueError
            If the returned masks list is empty.
        """
        if do_3D:
            raise NotImplementedError('3D not implemented yet')

        stack = ch0_stack if use_ch0 else ch1_stack

        if not multiple_stacks:
            imgs = [self.stack_to_image(stack, n_artifact)[..., np.newaxis]]
        else:
            imgs = [self.stack_to_image(stack_i, n_artifact) for stack_i in stack]
            imgs = np.stack(imgs, axis=0)[..., np.newaxis]

        eval_kwargs = self.eval_kwargs.copy()

        if 'min_size' in eval_kwargs:
            eval_kwargs['min_size'] = int(eval_kwargs['min_size'] / np.mean(pixel_size_um[:2]))
            print(f"min_size: {eval_kwargs['min_size']} [px]")

        if 'diameter' in eval_kwargs:
            eval_kwargs['diameter'] = int(eval_kwargs['diameter'] / np.mean(pixel_size_um[:2]))
            print(f"diameter: {eval_kwargs['diameter']} [px]")

        if len(pixel_size_um) > 2:
            eval_kwargs['anisotropy'] = pixel_size_um[2] / np.mean(pixel_size_um[:2])
            print(f"anisotropy: {eval_kwargs['anisotropy']}")

        if multiple_stacks and 'stitch_threshold' not in eval_kwargs:
            eval_kwargs['stitch_threshold'] = 1e-9  # Stitch everything together

        masks, flows, styles, diams = self.model.eval(imgs, do_3D=do_3D, **eval_kwargs)

        if plot:
            self.plot_results(imgs, masks, flows)

        if isinstance(masks, list):
            if len(masks) == 0:
                raise ValueError("Empty masks")
            elif len(masks) == 1:
                roi_mask = masks[0]
            else:
                roi_mask = np.max(masks, axis=0)
        else:
            roi_mask = masks

        return roi_mask

    def plot_results(self, imgs: list, masks: list, flows: list) -> None:
        """Plot segmentation results using the cellpose built-in visualisation.

        Parameters
        ----------
        imgs : list
            List of input images passed to ``model.eval``.
        masks : list
            List of predicted mask arrays returned by ``model.eval``.
        flows : list
            List of flow arrays returned by ``model.eval``.
        """
        from cellpose import plot

        for imgi, maski, flowi in zip(imgs, masks, flows):
            fig = plt.figure(figsize=(12, 5))
            plot.show_segmentation(fig, imgi, maski, flowi, channels=[0, 0])
            plt.tight_layout()
            plt.show()
