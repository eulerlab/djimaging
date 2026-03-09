import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse


def flatten_color_channels(rf: np.ndarray) -> np.ndarray:
    """Flatten the color-channel dimension of an RF array by interleaving channels.

    Parameters
    ----------
    rf : np.ndarray
        Receptive field array with shape (n_t, n_y, n_x, n_colors).

    Returns
    -------
    np.ndarray
        Flattened array with shape (n_t, n_y, n_x * n_colors + (n_colors - 1)),
        where channels are separated by NaN columns.
    """
    rf_flat = np.full((rf.shape[0], rf.shape[1], rf.shape[2] * rf.shape[3] + (rf.shape[3] - 1)), np.nan)
    for color_idx in range(rf.shape[3]):
        shift = 0 if color_idx == 0 else 1
        rf_flat[:, :, rf.shape[2] * color_idx + shift:rf.shape[2] * (color_idx + 1) + shift] = (
            rf[:, :, :, color_idx])
    return rf_flat


def plot_rf_frames(rf: np.ndarray, rf_time: np.ndarray, downsample: int = 1) -> tuple:
    """Plot all temporal frames of a receptive field.

    Parameters
    ----------
    rf : np.ndarray
        Receptive field array, shape (n_t, n_y, n_x) or (n_t, n_y, n_x, n_colors)
        or (n_t, n_x) for 1D.
    rf_time : np.ndarray
        Time axis, shape (n_t,).
    downsample : int, optional
        Temporal downsampling factor. Default is 1 (no downsampling).

    Returns
    -------
    tuple
        fig : matplotlib.figure.Figure
            The figure object.
        axs : matplotlib.axes.Axes or np.ndarray
            The axes object(s).

    Raises
    ------
    NotImplementedError
        If rf.ndim is not 2 or 3 (after optional color flattening).
    """
    from matplotlib import pyplot as plt
    from djimaging.utils.plot_utils import plot_srf

    if rf.ndim == 4:
        rf = flatten_color_channels(rf)

    if downsample > 1:
        rf = rf[::downsample]
        rf_time = rf_time[::downsample]

    if rf.ndim == 2:
        fig, axs = plt.subplots(1, 1, figsize=(10, 3))
        axs.plot(rf_time, rf)
    elif rf.ndim == 3:
        n_rows = int(np.ceil(np.sqrt(rf.shape[0])))
        n_cols = int(np.ceil(rf.shape[0] / n_rows))
        aspect_ratio = rf.shape[1] / rf.shape[2]
        fig, axs = plt.subplots(
            n_rows, n_cols, figsize=(np.minimum(n_cols * 3, 15), np.minimum(n_rows * 3 * aspect_ratio, 15)),
            squeeze=False, sharey='all', sharex='all')
        axs = axs.flat
        vabsmax = np.nanmax(np.abs(rf))
        for ax, frame, t in zip(axs, rf, rf_time):
            plot_srf(frame, ax=ax, vabsmax=vabsmax)
            ax.set_title(f"t={t:.3f}")
        for ax in axs[len(rf):]:
            ax.axis('off')
        plt.tight_layout()
    else:
        raise NotImplementedError(rf.shape)

    return fig, axs


def plot_rf_video(rf: np.ndarray, rf_time: np.ndarray, fps: int = 10):
    """Create animation of sRF and display in Jupyter notebook.

    Parameters
    ----------
    rf : np.ndarray
        Receptive field array, shape (n_t, n_y, n_x) or (n_t, n_y, n_x, n_colors).
    rf_time : np.ndarray
        Time axis, shape (n_t,).
    fps : int, optional
        Frames per second for the animation. Default is 10.

    Returns
    -------
    IPython.display.HTML
        HTML5 video animation object for display in a notebook.
    """
    from matplotlib import pyplot as plt
    from matplotlib.animation import FuncAnimation
    from IPython.display import HTML

    if rf.ndim == 4:
        rf = flatten_color_channels(rf)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.set_aspect('equal')

    vabsmax = np.nanmax(np.abs(rf))
    im = ax.imshow(rf[0], cmap='coolwarm', vmin=-vabsmax, vmax=vabsmax, origin='lower')

    def update(frame: int):
        """Update the animation frame.

        Parameters
        ----------
        frame : int
            Current frame index.

        Returns
        -------
        tuple
            Updated image artist.
        """
        if frame < len(rf):
            im.set_data(rf[frame])
            ax.set_title(f"t={rf_time[frame]:.3f}")
        else:
            im.set_data(np.zeros_like(rf[0]))
            ax.set_title("pause")
        return im,

    plt.close(fig)

    anim = FuncAnimation(fig, update, frames=np.arange(int(len(rf) * 1.5)), blit=True, interval=1000 / fps)
    return HTML(anim.to_html5_video())


def plot_srf_gauss_fit(ax, srf: np.ndarray = None, vabsmax: float = None,
                       srf_params: dict = None, n_std: float = 2,
                       color: str = 'k', ms: float = 3,
                       plot_cb: bool = False, **kwargs) -> None:
    """Overlay a Gaussian fit ellipse on an sRF image.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on.
    srf : np.ndarray, optional
        2D spatial receptive field array to display as background image.
    vabsmax : float, optional
        Symmetric color limit for the background image.
        If None and srf is provided, uses data min/max.
    srf_params : dict, optional
        Dictionary with Gaussian fit parameters: 'x_mean', 'y_mean',
        'x_stddev', 'y_stddev', 'theta'.
    n_std : float, optional
        Number of standard deviations for the ellipse radius. Default is 2.
    color : str, optional
        Color for the ellipse and center marker. Default is 'k'.
    ms : float, optional
        Marker size for the center cross. Default is 3.
    plot_cb : bool, optional
        Whether to add a colorbar. Default is False.
    **kwargs
        Additional keyword arguments passed to Ellipse and ax.plot.
    """
    if srf_params is not None:
        ax.plot(srf_params['x_mean'], srf_params['y_mean'], zorder=100, marker='x', ms=ms, c=color, **kwargs)
        ax.add_patch(Ellipse(
            xy=(srf_params['x_mean'], srf_params['y_mean']),
            width=n_std * 2 * srf_params['x_stddev'],
            height=n_std * 2 * srf_params['y_stddev'],
            angle=np.rad2deg(srf_params['theta']), color=color, fill=False, **kwargs))

    if srf is not None:
        if vabsmax is None:
            vmin = np.min(srf)
            vmax = np.max(srf)
            cmap = 'gray'
        else:
            vmin = -vabsmax
            vmax = vabsmax
            cmap = 'bwr'
        im = ax.imshow(srf, vmin=vmin, vmax=vmax, cmap=cmap, zorder=0, origin='lower')
        if plot_cb:
            plt.colorbar(im, ax=ax)
