import numpy as np
from matplotlib import pyplot as plt


def plot_field(ch0_average, ch1_average, roi_mask=None, title='', figsize=(16, 4), highlight_roi=None,
               fig=None, axs=None):
    if roi_mask is not None and roi_mask.size == 0:
        roi_mask = None

    if (fig is None) or (axs is None):
        fig, axs = plt.subplots(1, 2 if roi_mask is None else 4, figsize=figsize, sharex='all', sharey='all')

    fig.suptitle(title)

    extent = (0, ch0_average.shape[0], 0, ch0_average.shape[1])

    ax = axs[0]
    ax.imshow(ch0_average.T, origin='lower', extent=extent)
    ax.set(title='ch0_average')

    ax = axs[1]
    ax.imshow(ch1_average.T, origin='lower', extent=extent)
    ax.set(title='ch1_average')

    if roi_mask is not None:
        rois = -roi_mask.astype(float).T

        ax = axs[2]
        _rois = rois.copy()
        _rois[_rois <= 0] = np.nan
        roi_mask_im = ax.imshow(_rois, cmap='jet', origin='lower', extent=extent)
        plt.colorbar(roi_mask_im, ax=ax)
        ax.set(title='roi_mask')

        ax = axs[3]

        ax.imshow(ch0_average.T, cmap='viridis', origin='lower', extent=extent)
        rois_us = np.repeat(np.repeat(rois, 10, axis=0), 10, axis=1)
        vmin = np.min(rois)
        vmax = np.max(rois)

        if highlight_roi is not None:
            rois_to_plot = [highlight_roi]
            ax.set(title=f'roi{highlight_roi} + ch0')
        else:
            rois_to_plot = np.unique(rois[rois > 0])
            ax.set(title='roi_mask + ch0')

        for roi in rois_to_plot:
            _rois_us = (rois_us == roi).astype(int) * roi
            ax.contour(_rois_us, extent=extent, vmin=vmin, vmax=vmax, levels=[roi - 1e-3], alpha=0.8, cmap='jet')

    plt.tight_layout()


def plot_field_and_traces(ch0_average, ch1_average, roi_mask, title='', figsize=(16, 8), highlight_roi=None):

    fig = plt.figure(figsize=figsize)
    field_axs = [
        plt.subplot2grid((2, 4), (0, 0)),
        plt.subplot2grid((2, 4), (0, 1)),
        plt.subplot2grid((2, 4), (0, 2)),
        plt.subplot2grid((2, 4), (0, 3)),
    ]

    trace_ax = plt.subplot2grid((2, 4), (1, 0), colspan=4)

    plot_field(ch0_average=ch0_average, ch1_average=ch1_average, roi_mask=roi_mask,
               title=title, highlight_roi=highlight_roi, fig=fig, axs=field_axs)

    trace_ax.plot([0, 1], [0, 1])
    plt.tight_layout()


def plot_trace_and_trigger(time, trace, triggertimes, trace_norm=None, title='', ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 2))

    title = str(title)

    if '\n' not in title and len(title) > 50:
        title = title[:len(title)//2] + '\n' + title[len(title)//2:]

    ax.set(title=title)
    ax.plot(time, trace)
    ax.set(xlabel='time', ylabel='trace')
    ax.vlines(triggertimes, np.min(trace) - 0.1 * (np.max(trace) - np.min(trace)), np.min(trace),
              color='r', label='trigger')
    ax.legend(loc='upper right')

    if trace_norm is not None:
        tax = ax.twinx()
        tax.plot(time, trace_norm)
        tax.set(ylabel='normalized')
