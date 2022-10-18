import numpy as np
from matplotlib import pyplot as plt


def plot_field(ch0_average, ch1_average, roi_mask=None, title='', figsize=(16, 4), highlight_roi=None):

    if roi_mask is not None and roi_mask.size == 0:
        roi_mask = None

    fig, axs = plt.subplots(1, 2 if roi_mask is None else 4, figsize=figsize, sharex='all', sharey='all')
    fig.suptitle(title)

    extent = (0, ch0_average.shape[1], 0, ch0_average.shape[0])

    ax = axs[0]
    ax.imshow(ch0_average, origin='lower', extent=extent)
    ax.set(title='ch0_average')

    ax = axs[1]
    ax.imshow(ch1_average, origin='lower', extent=extent)
    ax.set(title='ch1_average')

    if roi_mask is not None:
        rois = -roi_mask.astype(float)

        ax = axs[2]
        _rois = rois.copy()
        _rois[_rois <= 0] = np.nan
        roi_mask_im = ax.imshow(_rois, cmap='jet', origin='lower', extent=extent)
        plt.colorbar(roi_mask_im, ax=ax)
        ax.set(title='roi_mask')

        ax = axs[3]

        ax.imshow(ch0_average, cmap='viridis', origin='lower', extent=extent)
        rois_us = np.repeat(np.repeat(rois, 10, axis=0), 10, axis=1)
        vmin = np.min(rois)
        vmax = np.max(rois)

        if highlight_roi is not None:
            rois_to_plot = [highlight_roi]
            ax.set(title=f'roi{highlight_roi} + ch0')
        else:
            rois_to_plot = np.unique(rois[rois>0])
            ax.set(title='roi_mask + ch0')

        for roi in rois_to_plot:
            _rois_us = (rois_us == roi).astype(int) * roi
            ax.contour(_rois_us, extent=extent, vmin=vmin, vmax=vmax, levels=[roi-1e-3], alpha=0.8, cmap='jet')

    plt.tight_layout()
    plt.show()
