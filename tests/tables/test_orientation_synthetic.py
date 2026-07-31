import numpy as np

from djimaging.tables.response.movingbar.orientation_utils_v2 import compute_os_ds_idxs


DIRECTIONS_DEG = np.array([0, 180, 45, 225, 90, 270, 135, 315])
DT = 0.128


def _synthetic_snippets(tuning: str) -> tuple[np.ndarray, np.ndarray]:
    times = np.arange(32) * DT
    time_kernel = (
        np.exp(-((times - 1.7) / 0.45) ** 2)
        + 0.4 * np.exp(-((times - 3.0) / 0.3) ** 2)
    )

    directions_rad = np.deg2rad(DIRECTIONS_DEG)
    if tuning == "direction":
        gains = 1 + 0.8 * np.cos(directions_rad)
    elif tuning == "orientation":
        gains = 1 + 0.8 * np.cos(2 * (directions_rad - np.deg2rad(45)))
    else:
        raise ValueError(tuning)

    snippets = []
    direction_order = []
    for repeat_gain in (0.95, 1.0, 1.05):
        snippets.extend(time_kernel * gain * repeat_gain for gain in gains)
        direction_order.extend(DIRECTIONS_DEG)

    return np.stack(snippets, axis=1), np.asarray(direction_order)


def test_direction_selective_response():
    snippets, direction_order = _synthetic_snippets("direction")
    np.random.seed(42)

    result = compute_os_ds_idxs(
        snippets=snippets,
        dir_order=direction_order,
        dt=DT,
        n_shuffles=256,
    )
    dsi, p_dsi, _, preferred_direction, osi, p_osi = result[:6]

    expected_dsi = 0.5 * (np.pi / 4) / (2 * np.sin(np.pi / 8))
    assert np.isclose(dsi, expected_dsi)
    assert np.isclose(preferred_direction, 0)
    assert np.isclose(osi, 0, atol=1e-12)
    assert p_dsi == 0
    assert p_osi > 0.5


def test_orientation_selective_non_directional_response():
    snippets, direction_order = _synthetic_snippets("orientation")
    np.random.seed(42)

    result = compute_os_ds_idxs(
        snippets=snippets,
        dir_order=direction_order,
        dt=DT,
        n_shuffles=256,
    )
    dsi, p_dsi, _, _, osi, p_osi, _, preferred_orientation = result[:8]

    expected_osi = 0.5 * (np.pi / 2) / (2 * np.sin(np.pi / 4))
    assert np.isclose(dsi, 0, atol=1e-12)
    assert np.isclose(osi, expected_osi)
    assert np.isclose(preferred_orientation, np.deg2rad(45))
    assert p_dsi > 0.5
    assert p_osi == 0
