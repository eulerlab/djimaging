import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse


def flatten_color_channels(rf):
    rf_flat = np.full((rf.shape[0], rf.shape[1], rf.shape[2] * rf.shape[3] + (rf.shape[3] - 1)), np.nan)
    for color_idx in range(rf.shape[3]):
        shift = 0 if color_idx == 0 else 1
        rf_flat[:, :, rf.shape[2] * color_idx + shift:rf.shape[2] * (color_idx + 1) + shift] = (
            rf[:, :, :, color_idx])
    return rf_flat


def plot_rf_frames(rf, rf_time, downsample=1):
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


def plot_rf_video(rf, rf_time, fps=10):
    """Create animation of sRF and display in jupyter notebook"""
    from matplotlib import pyplot as plt
    from matplotlib.animation import FuncAnimation
    from IPython.display import HTML

    if rf.ndim == 4:
        rf = flatten_color_channels(rf)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.set_aspect('equal')

    vabsmax = np.nanmax(np.abs(rf))
    im = ax.imshow(rf[0], cmap='coolwarm', vmin=-vabsmax, vmax=vabsmax, origin='lower')

    def update(frame):
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


def plot_srf_gauss_fit(ax, srf=None, vabsmax=None, srf_params=None, n_std=2, color='k', ms=3, plot_cb=False, **kwargs):
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
