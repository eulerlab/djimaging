import numpy as np
from matplotlib import pyplot as plt


def plot_field(main_ch_average, alt_ch_average, roi_mask=None, roi_ch_average=None, npixartifact=0,
               title='', figsize=(20, 4), highlight_roi=None, fig=None, axs=None):
    if roi_mask is not None and roi_mask.size == 0:
        roi_mask = None

    if roi_ch_average is None:
        roi_ch_average = main_ch_average

    if (fig is None) or (axs is None):
        fig, axs = plt.subplots(1, 2 if roi_mask is None else 4, figsize=figsize, sharex='all', sharey='all')

    fig.suptitle(title)

    extent = (0, main_ch_average.shape[0], 0, main_ch_average.shape[1])

    ax = axs[0]
    ax.imshow(main_ch_average.T, origin='lower', extent=extent)
    ax.set(title='main average')

    ax = axs[1]
    ax.imshow(alt_ch_average.T, origin='lower', extent=extent)
    ax.set(title='alt average')

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


def plot_field_and_traces(main_ch_average, alt_ch_average, roi_mask, title='', figsize=(16, 8), highlight_roi=None):
    fig = plt.figure(figsize=figsize)
    field_axs = [
        plt.subplot2grid((2, 4), (0, 0)),
        plt.subplot2grid((2, 4), (0, 1)),
        plt.subplot2grid((2, 4), (0, 2)),
        plt.subplot2grid((2, 4), (0, 3)),
    ]

    trace_ax = plt.subplot2grid((2, 4), (1, 0), colspan=4)

    plot_field(main_ch_average=main_ch_average, alt_ch_average=alt_ch_average, roi_mask=roi_mask,
               title=title, highlight_roi=highlight_roi, fig=fig, axs=field_axs)

    trace_ax.plot([0, 1], [0, 1])
    plt.tight_layout()


def prep_long_title(title=None):
    if title is None:
        return
    title = str(title)
    if '\n' not in title and len(title) > 50:
        title = title[:len(title) // 2] + '\n' + title[len(title) // 2:]
    return title


def set_long_title(ax=None, fig=None, title=None, **kwargs):
    if ax is not None:
        ax.set_title(prep_long_title(title=title), **kwargs)
    elif fig is not None:
        fig.suptitle(prep_long_title(title=title), **kwargs)


def plot_traces(time, traces, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 2))
    set_long_title(ax, title=title)

    for trace in traces:
        ax.plot(time, trace, alpha=np.maximum(1. / len(traces), 0.3))


def plot_trace_and_trigger(time, trace, triggertimes, trace_norm=None, title=None, ax=None, label=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 2))

    set_long_title(ax, title=title)
    ax.plot(time, trace, label=label)
    ax.set(xlabel='time', ylabel='trace')
    if len(triggertimes) > 0:
        ax.vlines(triggertimes, np.min(trace), np.max(trace), color='r', label='trigger', zorder=-2)
    ax.legend(loc='upper right')

    if trace_norm is not None:
        tax = ax.twinx()
        tax.plot(time, trace_norm, ':')
        if len(triggertimes) > 0:
            tax.vlines(triggertimes, np.min(trace_norm), np.max(trace_norm), color='r', label='trigger', ls=':',
                       zorder=-1)
        tax.set(ylabel='normalized')

    return ax


def plot_srf(srf, ax=None, vabsmax=None, pixelsize=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    if vabsmax is None:
        vabsmax = np.nanmax(np.abs(srf))

    if np.any(srf < 0):
        vmin = -vabsmax
        vmax = vabsmax
        cmap = 'bwr'
    else:
        vmin = 0
        vmax = vabsmax
        cmap = 'viridis'

    if pixelsize is not None:
        extent = np.array([-srf.shape[1] / 2., srf.shape[1] / 2., -srf.shape[0] / 2., srf.shape[0] / 2.]) * pixelsize
    else:
        extent = None

    ax.set(title='sRF')
    im = ax.imshow(srf.T, vmin=vmin, vmax=vmax, cmap=cmap, interpolation='none', origin='lower', extent=extent)
    plt.colorbar(im, ax=ax)

    return ax


def plot_trf(trf, t_trf=None, peak_idxs=None, ax=None, lim_y=True):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    if t_trf is None:
        t_trf = np.arange(-trf.size + 1, 1)

    ax.set(title='tRF')
    ax.fill_between(t_trf, trf)
    if peak_idxs is not None:
        for peak_idx in peak_idxs:
            ax.axvline(t_trf[peak_idx], color='r')

    if lim_y:
        vabsmax = np.nanmax(np.abs(trf))
        ax.set(ylim=(-1.1 * vabsmax, 1.1 * vabsmax))

    return ax


def plot_signals_heatmap(signals, ax=None, cb=True, vabsmax=None, symmetric=True):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 3))

    if vabsmax is None:
        vabsmax = np.nanmax(np.abs(signals))

    if symmetric:
        vmin = -vabsmax
        vmax = vabsmax
        cmap = 'bwr'
    else:
        vmin = np.nanmin(signals)
        vmax = np.nanmax(signals)
        cmap = 'viridis'

    im = ax.imshow(signals, aspect='auto', vmin=vmin, vmax=vmax, cmap=cmap,
                   interpolation='none', origin='lower')
    if cb:
        plt.colorbar(im, ax=ax)

    ax.set_yticks((0, signals.shape[0] - 1))
    ax.set_yticks(np.arange(signals.shape[0]), minor=True)

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


def plot_clusters(traces_list, stim_names, clusters, kind='averages', title=None, min_count=1):
    """Plot traces sorted by clusters"""
    unique_clusters, cluster_counts = np.unique(clusters, return_counts=True)

    if min_count > 1:
        unique_clusters = unique_clusters[cluster_counts >= min_count]
        cluster_counts = cluster_counts[cluster_counts >= min_count]

    n_rows = np.minimum(unique_clusters.size, 15)
    n_cols = len(traces_list)

    if kind == 'averages':
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, 0.8 * (1 + n_rows)), squeeze=False,
                                sharex='col', sharey='row')
    else:
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, 2 * (1 + n_rows)), squeeze=False,
                                sharex='col', sharey='row', gridspec_kw=dict(height_ratios=cluster_counts[:n_rows] + 1))

    set_long_title(fig=fig, title=title, y=1, va='bottom')

    for ax_col, stim_i, traces_i in zip(axs.T, stim_names, traces_list):
        ax_col[0].set(title=f"stim={stim_i}")
        for row_i, (ax, cluster) in enumerate(zip(ax_col, unique_clusters)):
            c_traces_i = traces_i[clusters == cluster, :]

            if kind == 'averages':
                plot_mean_trace_and_std(ax=ax, time=np.arange(c_traces_i.shape[1]), traces=c_traces_i,
                                        color=f'C{row_i}')
            else:
                plot_signals_heatmap(signals=c_traces_i, ax=ax, cb=True, vabsmax=np.max(np.abs(traces_i)))

            ax.set(ylabel=f"cluster={int(cluster)}\nn={c_traces_i.shape[0]}")

    plt.tight_layout()
    plt.show()


def plot_roi_mask_boundaries(ax, roi_mask, extent=None, zorder=1000):
    rois = roi_mask.copy().astype(float).T

    if np.any(roi_mask < 0):
        rois = -rois

    rois_us = np.repeat(np.repeat(rois, 10, axis=0), 10, axis=1)
    vmin, vmax = np.min(rois), np.max(rois)

    roi_idxs = np.unique(rois)
    roi_idxs = roi_idxs[roi_idxs > 0]

    for roi in roi_idxs:
        _rois_us = (rois_us == roi).astype(int) * roi
        ax.contour(_rois_us, extent=extent, vmin=vmin, vmax=vmax, levels=[roi - 1e-3], alpha=0.8, cmap='jet',
                   zorder=zorder)
