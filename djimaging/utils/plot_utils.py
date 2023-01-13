import numpy as np
from matplotlib import pyplot as plt


def plot_field(ch0_average, ch1_average, roi_mask=None, roi_ch_average=None, npixartifact=0,
               title='', figsize=(16, 4), highlight_roi=None, fig=None, axs=None):
    if roi_mask is not None and roi_mask.size == 0:
        roi_mask = None

    if roi_ch_average is None:
        roi_ch_average = ch0_average

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

        _roi_ch_average = roi_ch_average.copy()
        _roi_ch_average[:npixartifact] = np.nan

        ax.imshow(_roi_ch_average.T, cmap='viridis', origin='lower', extent=extent)
        rois_us = np.repeat(np.repeat(rois, 10, axis=0), 10, axis=1)
        vmin = np.min(rois)
        vmax = np.max(rois)

        if highlight_roi is not None:
            rois_to_plot = [highlight_roi]
            ax.set(title=f'roi{highlight_roi} + data')
        else:
            rois_to_plot = np.unique(rois[rois > 0])
            ax.set(title='roi_mask + data')

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


def plot_traces(time, traces, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 2))

    for trace in traces:
        ax.plot(time, trace, alpha=np.maximum(1. / len(traces), 0.3))


def plot_trace_and_trigger(time, trace, triggertimes, trace_norm=None, title='', ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 2))

    title = str(title)

    if '\n' not in title and len(title) > 50:
        title = title[:len(title) // 2] + '\n' + title[len(title) // 2:]

    ax.set(title=title)
    ax.plot(time, trace)
    ax.set(xlabel='time', ylabel='trace')
    ax.vlines(triggertimes, np.min(trace), np.max(trace), color='r', label='trigger', zorder=-2)
    ax.legend(loc='upper right')

    if trace_norm is not None:
        tax = ax.twinx()
        tax.plot(time, trace_norm, ':')
        tax.vlines(triggertimes, np.min(trace_norm), np.max(trace_norm), color='r', label='trigger', ls=':', zorder=-1)
        tax.set(ylabel='normalized')

    return ax


def plot_srf(srf, ax=None, vabsmax=None, pixelsize=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    if vabsmax is None:
        vabsmax = np.max(np.abs(srf))

    if pixelsize is not None:
        extent = np.array([-srf.shape[1] / 2., srf.shape[1] / 2., -srf.shape[0] / 2., srf.shape[0] / 2.]) * pixelsize
    else:
        extent = None

    ax.set(title='sRF')
    im = ax.imshow(srf.T, vmin=-vabsmax, vmax=vabsmax, cmap='bwr', origin='lower', extent=extent)
    plt.colorbar(im, ax=ax)

    return ax


def plot_trf(trf, t_trf=None, peak_idxs=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    if t_trf is None:
        t_trf = np.arange(-trf.size + 1, 1)

    vabsmax = np.max(np.abs(trf))

    ax.set(title='tRF')
    ax.fill_between(t_trf, trf)
    if peak_idxs is not None:
        for peak_idx in peak_idxs:
            ax.axvline(t_trf[peak_idx], color='r')
    ax.set(ylim=(-1.1 * vabsmax, 1.1 * vabsmax))

    return ax


def plot_signals_heatmap(signals, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 3))

    from sklearn.decomposition import PCA

    vabsmax = np.max(np.abs(signals))
    ax.imshow(signals[np.argsort(PCA(n_components=1).fit_transform(signals).flat)], aspect='auto',
              vmin=-vabsmax, vmax=vabsmax, cmap='coolwarm')
    return ax


def plot_df_cols_as_histograms(df):
    fig, axs = plt.subplots(1, len(df.columns), figsize=(8, 3))
    for ax, col in zip(axs, df.columns):
        df.hist(column=col, ax=ax)
    return axs


def plot_mean_trace_and_std(ax, time, traces, label=None, color='k', color2='gray', lw=0.8, downsample_factor=1):
    """Plot mean and std of trace"""

    trace_mean = np.mean(traces, axis=0)
    trace_std = np.std(traces, axis=0)

    idxs = np.arange(0, time.size, downsample_factor)

    ax.plot(time[idxs], trace_mean[idxs], c=color, label=label, zorder=2, lw=lw, clip_on=False)
    ax.fill_between(time[idxs], trace_mean[idxs] + trace_std[idxs], trace_mean[idxs] - trace_std[idxs], alpha=0.7,
                    color=color2, lw=0, clip_on=False)
