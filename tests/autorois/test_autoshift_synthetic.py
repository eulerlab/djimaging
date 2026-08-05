from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from djimaging.autorois.autoshift_utils import (
    compute_corr_map,
    compute_corr_map_match_indexes,
    extract_best_shift,
    shift_img,
)
from djimaging.utils.scanm import read_h5_utils


PARAMS_TEST_AUTOSHIFT = [
    (0, -1, 3, 'corr'),
    (-1, -1, 3, 'corr'),
    (-2, 2, 5, 'corr'),
    (0, 0, 3, 'mse'),
    (1, 0, 3, 'mse'),
    (0, 1, 3, 'mse'),
    (1, 1, 3, 'mse'),
    (-1, 0, 3, 'mse'),
    (0, -1, 3, 'mse'),
    (-1, -1, 3, 'mse'),
    (-2, 2, 5, 'mse'),
]


@pytest.fixture(scope="module")
def synthetic_stack(synthetic_recording_path: Callable[[str], Path]) -> np.ndarray:
    ch_stacks, _ = read_h5_utils.load_stacks_and_wparams(
        str(synthetic_recording_path("chirp")), ch_names=('wDataCh0',)
    )
    return ch_stacks['wDataCh0']


@pytest.mark.parametrize("shift_x, shift_y, shift_max, metric", PARAMS_TEST_AUTOSHIFT)
def test_autoshift_synthetic_stack(shift_x, shift_y, shift_max, metric, synthetic_stack):
    ref_corr = compute_corr_map(synthetic_stack)
    shifted_corr = shift_img(ref_corr, shift_x, shift_y)
    match_indexes = compute_corr_map_match_indexes(
        shifted_corr, ref_corr, shift_max, metric=metric
    )

    obs_shift_x, obs_shift_y = extract_best_shift(match_indexes)

    assert obs_shift_x == shift_x and obs_shift_y == shift_y
