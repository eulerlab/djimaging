import numpy as np


def plot_rf_frames(rf, rf_time, downsample=1):
    from matplotlib import pyplot as plt
    from djimaging.utils.plot_utils import plot_srf

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
            n_rows, n_cols, figsize=(np.minimum(n_cols * 3, 15), np.minimum(n_rows * 2 * aspect_ratio, 15)),
            squeeze=False, sharey='all', sharex='all')
        axs = axs.flat
        vabsmax = np.max(np.abs(rf))
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

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.set_aspect('equal')

    vabsmax = np.max(np.abs(rf))
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
