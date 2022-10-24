import numpy as np
from scipy.ndimage.measurements import center_of_mass
import morphkit


def get_linestack(df_paths, stack_shape):
    """Get binary 3D volume with pixels being either zero or belonging to the morphology"""
    coords = np.vstack(df_paths.path_stack)
    coords = np.minimum(coords, np.array(stack_shape) - 1)
    linestack = np.zeros(stack_shape, dtype=np.uint8)

    for c in coords:
        linestack[tuple(c)] = 1

    return linestack


def compute_df_paths_and_density_maps(swc_path, pixel_size_um):
    m = morphkit.Morph(filepath=swc_path)
    m.summarize()

    df_paths = m.df_paths
    if np.unique(np.concatenate(m.df_paths.radius)).size == 1:
        df_paths.drop(labels='radius', axis=1, inplace=True)
    df_paths.drop(labels='connect_to', axis=1, inplace=True)
    df_paths.drop(labels='connect_to_at', axis=1, inplace=True)
    df_paths.back_to_soma = df_paths.back_to_soma.apply(lambda x: np.array(x, dtype=int))

    # turning path from real coordinates to pixel coordinates in stack
    df_paths['path_stack'] = df_paths.path.apply(lambda x: np.round(x / pixel_size_um).astype(int))

    return df_paths, m.density_maps


def compute_density_map_extent(paths, soma):
    """Get density center of density maps in path coordinates"""
    lim = get_morphkit_path_lim(paths=paths, soma=soma)
    return soma[0] - lim, soma[0] + lim, soma[1] - lim, soma[1] + lim


def get_morphkit_path_lim(paths, soma):
    """Compute limits for coordinates as it is done in morphkit"""
    xy = np.vstack(paths)[:, :2] - soma[:2]
    lim_max = int(np.ceil(xy.T.max() / 20) * 20)
    lim_min = int(np.floor(xy.T.min() / 20) * 20)
    lim = max(abs(lim_max), abs(lim_min))
    return lim


def compute_density_center(paths, soma, density_map):
    """Get density center of density maps in path coordinates"""
    lim = get_morphkit_path_lim(paths=paths, soma=soma)
    density_center_map = np.array(center_of_mass(density_map)[::-1])
    density_center_shift = density_center_map * (2 * lim) / density_map.shape[0] - lim
    density_center = density_center_shift + soma[:2]
    return density_center


