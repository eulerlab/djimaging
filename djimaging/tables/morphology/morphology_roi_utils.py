import numpy as np
from matplotlib import pyplot as plt


def on_which_path(df_paths, point):
    result_path = df_paths[df_paths.path_stack.apply(lambda x: (x == point).all(1).any())]
    path_id = int(result_path.index[0])
    return path_id


def get_loc_on_path_stack(df_paths, point):
    return [i for i in df_paths.path_stack.apply(lambda x: np.where((x == point).all(1))[0]) if len(i) != 0][0][0]


def compute_dendritic_distance_to_soma(df_paths, path_id, loc_on_path=None):
    """Compute distance to soma_xyz"""
    # Sum all paths lengths from path_id to soma_xyz.
    length_all_paths = sum(df_paths.loc[df_paths.loc[path_id].back_to_soma].real_length)

    # Get piece of path_id segment that should be ignored in the total length?
    if loc_on_path is not None:
        length_to_reduce = compute_segment_length(df_paths.loc[path_id].path[loc_on_path:])
    else:
        length_to_reduce = 0.0

    return length_all_paths - length_to_reduce


def compute_segment_length(arr):
    """Sum of euclidean distances between points"""
    return np.sum(np.sqrt(np.sum((arr[1:] - arr[:-1]) ** 2, 1)))


def compute_euclidean_distance(pt1, pt2):
    """Compute euclidean distance between two points"""
    return np.sqrt(np.sum((pt1 - pt2) ** 2))


def plot_roi_positions_xyz(
        rois_pos_xyz, paths, soma_xyz, c, layer_on_z=None, layer_off_z=None,
        roi_max_distance=250, plot_rois=True, plot_morph=True, plot_soma=True,
        xlim=None, ylim=None, zlim=None):

    fig = plt.figure(figsize=(12, 12))
    plt.suptitle('ROIs on paths')

    ax1 = plt.subplot2grid((4, 4), (0, 1), rowspan=3, colspan=3)
    ax2 = plt.subplot2grid((4, 4), (0, 0), rowspan=3, colspan=1, sharey=ax1)
    ax3 = plt.subplot2grid((4, 4), (3, 1), rowspan=1, colspan=3, sharex=ax1)

    axs = np.array([ax1, ax2, ax3])

    ax1.set(xlabel='x', ylabel='y')
    ax2.set(xlabel='z', ylabel='y')
    ax3.set(xlabel='x', ylabel='z')

    if plot_morph:
        for path in paths:
            x, y, z = path.T
            ax1.plot(x, y, color='black', alpha=0.6)
            ax2.plot(z, y, color='black', alpha=0.6)
            ax3.plot(x, z, color='black', alpha=0.6)

    if plot_soma:
        ax1.scatter(soma_xyz[0], soma_xyz[1], c='grey', s=160, zorder=10, alpha=.3)
        ax2.scatter(soma_xyz[2], soma_xyz[1], c='grey', s=160, zorder=10, alpha=.3)
        ax3.scatter(soma_xyz[0], soma_xyz[2], c='grey', s=160, zorder=10, alpha=.3)

    if plot_rois:
        rois_pos_x, rois_pos_y, rois_pos_z = rois_pos_xyz
        sc = ax1.scatter(rois_pos_x, rois_pos_y, c=c, s=60, marker='o',
                         cmap="viridis", vmin=0, vmax=roi_max_distance, zorder=10)

        cbar = plt.colorbar(sc, ax=ax1, cax=ax1.inset_axes([0.9, 0.5, 0.01, 0.47]), fraction=0.02, pad=.01)
        cbar.outline.set_visible(False)

        ax2.scatter(rois_pos_z, rois_pos_y, c=c, s=60, marker='o',
                    cmap="viridis", vmin=0, vmax=roi_max_distance, zorder=10)
        ax3.scatter(rois_pos_x, rois_pos_z, c=c, s=60, marker='o',
                    cmap="viridis", vmin=0, vmax=roi_max_distance, zorder=10)

    if (layer_on_z is not None) and (layer_off_z is not None):
        ax3.axhline(layer_on_z, color='red', linestyle='dashed')
        ax3.axhline(layer_off_z, color='red', linestyle='dashed')

        ax3.annotate('ON', xy=(ax3.get_xlim()[1], layer_on_z), zorder=10, weight="bold", va='center', ha='left')
        ax3.annotate('OFF', xy=(ax3.get_xlim()[1], layer_off_z), zorder=10, weight="bold", va='center', ha='left')

    ax1.set(xlim=xlim, ylim=ylim)
    if zlim is not None:
        ax2.set_xlim(zlim)
        ax3.set_ylim(zlim)

    plt.tight_layout()

    return fig, axs
