import numpy as np
from matplotlib import pyplot as plt


class CellposeWrapper:
    def __init__(self, init_kwargs: dict, eval_kwargs: dict):
        self.init_kwargs = init_kwargs
        self.eval_kwargs = eval_kwargs

        from cellpose import models, io
        io.logger_setup()
        self.model = models.Cellpose(**self.init_kwargs)

    @staticmethod
    def stack_to_image(stack, n_artifact):
        img = stack.mean(axis=2) if stack.ndim > 2 else stack.copy()
        img[:n_artifact] = np.min(img)
        return img

    def create_mask_from_data(self, ch0_stack, ch1_stack=None, use_ch0=True, do_3D=False, n_artifact=0,
                              pixel_size_um=(1., 1.), multiple_stacks=False, plot=False, **kwargs):

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
                roi_mask = masks[0]
            else:
                roi_mask = np.max(masks, axis=0)
        else:
            roi_mask = masks

        return roi_mask

    def plot_results(self, imgs, masks, flows):
        from cellpose import plot

        for imgi, maski, flowi in zip(imgs, masks, flows):
            fig = plt.figure(figsize=(12, 5))
            plot.show_segmentation(fig, imgi, maski, flowi, channels=[0, 0])
            plt.tight_layout()
            plt.show()
