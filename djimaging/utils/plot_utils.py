import numpy as np
from matplotlib import pyplot as plt

from djimaging.utils.math_utils import normalize_zero_one


def plot_field(main_ch_average, alt_ch_average, scan_type='xy', roi_mask=None, roi_ch_average=None, npixartifact=0,
               title='', figsize=(18, 4), highlight_roi=None, fig=None, axs=None, gamma=1.):
    """Plot scanning-field averages and, optionally, the ROI mask overlay.

    Parameters
    ----------
    main_ch_average : np.ndarray
        2-D image of the main recording channel average.
    alt_ch_average : np.ndarray
        2-D image of the alternative channel average.
    scan_type : {'xy', 'xz'}, optional
        Orientation of the scan. Controls axis labels and image extent.
        Default is ``'xy'``.
    roi_mask : np.ndarray or None, optional
        2-D integer mask where each unique non-zero value identifies one
        ROI. If None or empty, only the two channel averages are shown.
        Default is None.
    roi_ch_average : np.ndarray or None, optional
        2-D image used as the background behind the ROI contours. Falls
        back to `main_ch_average` when None. Default is None.
    npixartifact : int, optional
        Number of pixel rows at the top of `roi_ch_average` to blank out
        (set to NaN) before display. Default is 0.
    title : str, optional
        Figure-level super-title. Default is ``''``.
    figsize : tuple of float, optional
        Figure size passed to :func:`matplotlib.pyplot.subplots`.
        Default is ``(18, 4)``.
    highlight_roi : int or None, optional
        If provided, only the contour of this ROI index is drawn.
        Default is None.
    fig : matplotlib.figure.Figure or None, optional
        Existing figure to draw into. A new figure is created when None.
        Default is None.
    axs : array-like of matplotlib.axes.Axes or None, optional
        Existing axes to draw into. New axes are created when None.
        Default is None.
    gamma : float, optional
        Gamma correction exponent applied to each normalised image
        (``image ** gamma``). Default is 1.0 (no correction).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the plot.
    axs : np.ndarray of matplotlib.axes.Axes
        Array of axes used in the figure.

    Raises
    ------
    ValueError
        If `scan_type` is not ``'xy'`` or ``'xz'``.
    """
    if roi_mask is not None and roi_mask.size == 0:
        roi_mask = None

    if roi_ch_average is None:
        roi_ch_average = main_ch_average

    # normalize
    main_ch_average = normalize_zero_one(main_ch_average)
    alt_ch_average = normalize_zero_one(alt_ch_average)
    roi_ch_average = normalize_zero_one(roi_ch_average)

    # gamma correction
    main_ch_average = main_ch_average ** gamma
    alt_ch_average = alt_ch_average ** gamma
    roi_ch_average = roi_ch_average ** gamma

    if (fig is None) or (axs is None):
        fig, axs = plt.subplots(1, 2 if roi_mask is None else 4, figsize=figsize, sharex='all', sharey='all')

    fig.suptitle(title)

    if scan_type == 'xy':
        for ax in axs:
            ax.set(xlabel='relY [pixel]')
        axs[0].set(ylabel='relX [pixel]')
        extent = (main_ch_average.shape[0] / 2, -main_ch_average.shape[0] / 2,
                  main_ch_average.shape[1] / 2, -main_ch_average.shape[1] / 2)
    elif scan_type == 'xz':
        for ax in axs:
            ax.set(xlabel='relY [pixel]')
        axs[0].set(ylabel='relZ [pixel]')
        extent = (main_ch_average.shape[0] / 2, -main_ch_average.shape[0] / 2,
                  -main_ch_average.shape[1] / 2, main_ch_average.shape[1] / 2)
    else:
        raise ValueError(f'Unknown scan_type: {scan_type}')

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
        ax.set(title='roi_mask')

        ax = axs[3]

        _roi_ch_average = roi_ch_average.copy()
        _roi_ch_average[:npixartifact] = np.nan

        ax.imshow(_roi_ch_average.T, cmap='viridis', origin='lower', extent=extent)

        if max(rois.shape) < 100:
            fus = 5
        elif max(rois.shape) < 200:
            fus = 2
        else:
            fus = 1

        rois_us = np.repeat(np.repeat(rois, fus, axis=0), fus, axis=1)
        vmin = np.min(rois)
        vmax = np.max(rois)
        plt.colorbar(roi_mask_im, ax=ax)

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
    return fig, axs


def plot_field_and_traces(main_ch_average, alt_ch_average, roi_mask, title='', figsize=(16, 8), highlight_roi=None):
    """Plot scanning-field averages, the ROI mask, and a placeholder trace row.

    Arranges a 2×4 grid: the top row shows the field images (via
    :func:`plot_field`) and the bottom row contains a single wide axes
    for traces.

    Parameters
    ----------
    main_ch_average : np.ndarray
        2-D image of the main recording channel average.
    alt_ch_average : np.ndarray
        2-D image of the alternative channel average.
    roi_mask : np.ndarray
        2-D integer ROI mask.
    title : str, optional
        Figure-level super-title. Default is ``''``.
    figsize : tuple of float, optional
        Figure size. Default is ``(16, 8)``.
    highlight_roi : int or None, optional
        If provided, only the contour of this ROI index is drawn in the
        field panel. Default is None.
    """
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


def prep_long_title(title=None, nmax=50):
    """Insert a newline near the midpoint of a title string that is too long.

    Parameters
    ----------
    title : str or None, optional
        Title string to process. Returns None unchanged. Default is None.
    nmax : int, optional
        Maximum number of characters allowed on a single line before a
        newline is inserted. Default is 50.

    Returns
    -------
    str or None
        Processed title string with a newline inserted if longer than
        `nmax` and not already containing one, or None if `title` was
        None.
    """
    if title is None:
        return
    title = str(title)
    if '\n' not in title and len(title) > nmax:
        title = title[:len(title) // 2] + '\n' + title[len(title) // 2:]
    return title


def set_long_title(ax=None, fig=None, title=None, **kwargs):
    """Set an axes title or figure super-title, wrapping long strings automatically.

    Passes `title` through :func:`prep_long_title` before applying it.
    Either `ax` or `fig` must be provided; `ax` takes precedence.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or None, optional
        Axes on which to call ``set_title``. Default is None.
    fig : matplotlib.figure.Figure or None, optional
        Figure on which to call ``suptitle``. Used only when `ax` is
        None. Default is None.
    title : str or None, optional
        Title string. Default is None.
    **kwargs
        Additional keyword arguments forwarded to ``set_title`` or
        ``suptitle``.
    """
    if ax is not None:
        ax.set_title(prep_long_title(title=title), **kwargs)
    elif fig is not None:
        fig.suptitle(prep_long_title(title=title), **kwargs)


def plot_traces(time, traces, ax=None, title=None):
    """Plot multiple traces on a single axes with alpha-blending.

    Parameters
    ----------
    time : np.ndarray
        1-D array of time values shared by all traces.
    traces : sequence of np.ndarray
        Collection of 1-D trace arrays to plot.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on. A new figure and axes are created when None.
        Default is None.
    title : str or None, optional
        Axes title (processed by :func:`set_long_title`). Default is
        None.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 2))
    set_long_title(ax, title=title)

    for trace in traces:
        ax.plot(time, trace, alpha=np.maximum(1. / len(traces), 0.3))


def plot_trace_and_trigger(time, trace, triggertimes, trace_norm=None, title=None, ax=None, label=None):
    """Plot a trace together with vertical trigger markers.

    Trigger times are shown as short vertical lines below the trace
    baseline. An optional normalised version of the trace can be plotted
    on a twin y-axis.

    Parameters
    ----------
    time : np.ndarray
        1-D array of time values.
    trace : np.ndarray
        1-D array of trace values.
    triggertimes : array-like
        Trigger onset times to mark on the plot.
    trace_norm : np.ndarray or None, optional
        Normalised trace to overlay on a twin y-axis. Default is None.
    title : str or None, optional
        Axes title. Default is None.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on. A new figure and axes are created when None.
        Default is None.
    label : str or None, optional
        Legend label for the main trace. Default is None.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 2))

    set_long_title(ax, title=title)
    ax.plot(time, trace, label=label)
    ax.set(xlabel='time', ylabel='trace')
    if len(triggertimes) > 0:
        vmin, vmax = np.nanmin(trace), np.nanmax(trace)
        vrng = vmax - vmin
        ax.vlines(triggertimes, vmin - 0.22 * vrng, vmin - 0.02 * vrng, color='r', label='trigger', zorder=-2)
    ax.legend(loc='upper right')

    if trace_norm is not None:
        tax = ax.twinx()
        tax.plot(time, trace_norm, ':')
        if len(triggertimes) > 0:
            vmin, vmax = np.nanmin(trace_norm), np.nanmax(trace_norm)
            vrng = vmax - vmin
            tax.vlines(triggertimes, vmin - 0.22 * vrng, vmin - 0.02 * vrng, color='r', label='trigger', ls=':',
                       zorder=-1)
        tax.set(ylabel='normalized')

    return ax


def plot_srf(srf, ax=None, vabsmax=None, pixelsize=None, extent=None):
    """Plot a spatial receptive field (sRF) as a colour-mapped image.

    The colormap and value limits are chosen automatically based on
    whether the data are purely positive, purely negative, or mixed.

    Parameters
    ----------
    srf : np.ndarray
        2-D array containing the spatial receptive field values.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on. A new figure and axes are created when None.
        Default is None.
    vabsmax : float or None, optional
        Absolute maximum for colour scaling. Computed from data when
        None. Default is None.
    pixelsize : float or None, optional
        Physical size of one pixel used to compute `extent` when
        `extent` is None. Default is None.
    extent : array-like or None, optional
        Extent ``[left, right, bottom, top]`` passed to
        :func:`matplotlib.axes.Axes.imshow`. Computed from `pixelsize`
        when None. Default is None.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    if vabsmax is None:
        vabsmax = np.nanmax(np.abs(srf))

    if np.all(srf >= 0):
        vmin = 0
        vmax = vabsmax
        cmap = 'viridis'
    elif np.all(srf <= 0):
        vmin = -vabsmax
        vmax = 0
        cmap = 'viridis_r'
    else:
        vmin = -vabsmax
        vmax = vabsmax
        cmap = 'bwr'

    if extent is None and pixelsize is not None:
        extent = srf_extent(srf_shape=srf.shape, pixelsize=pixelsize)

    ax.set(title='sRF')
    im = ax.imshow(srf, vmin=vmin, vmax=vmax, cmap=cmap, interpolation='none', origin='lower', extent=extent)
    plt.colorbar(im, ax=ax)

    return ax


def srf_extent(srf_shape, pixelsize=None):
    """Compute the image extent for an sRF plot centred at the origin.

    Parameters
    ----------
    srf_shape : tuple of int
        Shape ``(n_rows, n_cols)`` of the sRF array.
    pixelsize : float or None, optional
        Physical size of one pixel. Uses 1 when None. Default is None.

    Returns
    -------
    np.ndarray
        Array ``[left, right, bottom, top]`` suitable for the `extent`
        argument of :func:`matplotlib.axes.Axes.imshow`.
    """
    pixelsize = 1 if pixelsize is None else pixelsize
    return np.array([-srf_shape[1] / 2., srf_shape[1] / 2., -srf_shape[0] / 2., srf_shape[0] / 2.]) * pixelsize


def plot_trf(trf, t_trf=None, peak_idxs=None, ax=None, lim_y=True):
    """Plot a temporal receptive field (tRF) as a filled area curve.

    Parameters
    ----------
    trf : np.ndarray
        1-D array of tRF values.
    t_trf : np.ndarray or None, optional
        1-D array of time lags corresponding to `trf`. Defaults to
        ``np.arange(-len(trf) + 1, 1)`` when None. Default is None.
    peak_idxs : array-like of int or None, optional
        Indices into `trf` (and `t_trf`) at which to draw vertical
        reference lines. Default is None.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on. A new figure and axes are created when None.
        Default is None.
    lim_y : bool, optional
        If True, the y-axis limits are set to ±1.1× the absolute
        maximum of `trf`. Default is True.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes containing the plot.
    """
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
    """Plot a 2-D signal matrix as a heatmap.

    Parameters
    ----------
    signals : np.ndarray
        2-D array of shape ``(n_signals, n_time)`` to display.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on. A new figure and axes are created when None.
        Default is None.
    cb : bool, optional
        If True, add a colorbar next to the image. Default is True.
    vabsmax : float or None, optional
        Absolute maximum for colour scaling. Computed from data when
        None. Default is None.
    symmetric : bool, optional
        If True, use a symmetric ``bwr`` colormap centred at zero with
        limits ``[-vabsmax, vabsmax]``. If False, use ``viridis`` with
        data-driven limits. Default is True.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes containing the plot.
    """
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
    """Plot each column of a DataFrame as a histogram in a separate axes.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame whose columns are to be plotted. One axes is created
        per column.

    Returns
    -------
    axs : np.ndarray of matplotlib.axes.Axes
        Array of axes, one per column of `df`.
    """
    fig, axs = plt.subplots(1, len(df.columns), figsize=(8, 3))
    for ax, col in zip(axs, df.columns):
        df.hist(column=col, ax=ax)
    return axs


def plot_mean_trace_and_std(ax, time, traces, label=None, color='k', color2='gray', lw=0.8, downsample_factor=1):
    """Plot the mean trace with a shaded standard-deviation band.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    time : np.ndarray
        1-D time array of length ``n_time``.
    traces : np.ndarray
        2-D array of shape ``(n_traces, n_time)``.
    label : str or None, optional
        Legend label for the mean line. Default is None.
    color : str or color-like, optional
        Colour of the mean line. Default is ``'k'``.
    color2 : str or color-like, optional
        Fill colour of the standard-deviation band. Default is
        ``'gray'``.
    lw : float, optional
        Line width of the mean trace. Default is 0.8.
    downsample_factor : int, optional
        Plot only every ``downsample_factor``-th sample to reduce
        rendering overhead. Default is 1 (no downsampling).
    """

    trace_mean = np.mean(traces, axis=0)
    trace_std = np.std(traces, axis=0)

    idxs = np.arange(0, time.size, downsample_factor)

    ax.plot(time[idxs], trace_mean[idxs], c=color, label=label, zorder=2, lw=lw, clip_on=False)
    ax.fill_between(time[idxs], trace_mean[idxs] + trace_std[idxs], trace_mean[idxs] - trace_std[idxs], alpha=0.7,
                    color=color2, lw=0, clip_on=False)


def plot_clusters(traces_list, stim_names, clusters, kind='averages', title=None, min_count=1):
    """Plot traces grouped by cluster label, one stimulus per column.

    Parameters
    ----------
    traces_list : list of np.ndarray
        List of 2-D trace arrays, one per stimulus condition. Each array
        has shape ``(n_cells, n_time)``.
    stim_names : list of str
        Stimulus condition names, one per element of `traces_list`.
    clusters : np.ndarray
        1-D integer array of cluster labels, one per cell (row).
    kind : {'averages', 'heatmap'}, optional
        Visualisation style. ``'averages'`` plots the per-cluster mean
        ± std; ``'heatmap'`` shows all individual traces as a heatmap.
        Default is ``'averages'``.
    title : str or None, optional
        Figure-level super-title. Default is None.
    min_count : int, optional
        Clusters with fewer than `min_count` members are excluded.
        Default is 1.
    """
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
    """Draw ROI boundary contours on an existing axes.

    Each unique positive value in `roi_mask` is treated as a separate
    ROI and its boundary is drawn using :func:`matplotlib.axes.Axes.contour`.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes on which to draw the contours.
    roi_mask : np.ndarray
        2-D integer array where each unique non-zero value identifies one
        ROI. Negative values are sign-flipped before processing.
    extent : array-like or None, optional
        Image extent ``[left, right, bottom, top]`` passed to
        ``contour``. Default is None.
    zorder : int or float, optional
        Z-order of the contour artists. Default is 1000.
    """
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
