from djimaging.autorois.autoshift_utils import compute_corr_map
import numpy as np


def create_stack_with_corr_map_1():
    nx, ny, nt = 3, 6, 100
    stack = np.zeros((nx, ny, nt))

    ts = np.linspace(0, 2 * np.pi, nt)

    stack[:, :3] = np.sin(ts)
    stack[:, 3:] = -np.sin(ts)

    corr_map = np.zeros((nx, ny))
    corr_map[:, :2] = 1
    corr_map[:, 2] = np.array([1 / 5, 2 / 8, 1 / 5])
    corr_map[:, 3] = np.array([1 / 5, 2 / 8, 1 / 5])
    corr_map[:, 4:] = 1

    return stack, corr_map


def create_stack_with_corr_map_2():
    nx, ny, nt = 2, 2, 100
    stack = np.zeros((nx, ny, nt))

    ts = np.linspace(0, 2 * np.pi, nt)

    stack[0, 1, :] = np.sin(ts)
    stack[1, 0, :] = np.sin(ts)
    stack[1, 1, :] = - np.sin(ts)
    stack[0, 0, :] = - np.sin(ts)

    corr_map = np.full((nx, ny), -1 / 3)

    return stack, corr_map


def test_compute_corr_map_1():
    stack, corr_map = create_stack_with_corr_map_1()
    corr_map_obs = compute_corr_map(stack)
    assert corr_map_obs.shape == corr_map.shape
    assert np.allclose(corr_map_obs, corr_map)


def test_compute_corr_map_2():
    stack, corr_map = create_stack_with_corr_map_2()
    corr_map_obs = compute_corr_map(stack)
    assert corr_map_obs.shape == corr_map.shape
    assert np.allclose(corr_map_obs, corr_map)
