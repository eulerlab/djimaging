import os.path

import numpy as np
import pytest

from djimaging.autorois.autoshift_utils import compute_corr_map, compute_corr_map_match_indexes, \
    shift_img, extract_best_shift
from djimaging.utils.scanm import read_h5_utils

params_test_shit_img = [
    (0, 0),
    (1, 0),
    (0, 1),
    (1, 1),
    (-1, 0),
    (0, -1),
    (-1, -1),
    (1, -1),
    (-1, 1),
]


@pytest.mark.parametrize("shift_x, shift_y", params_test_shit_img)
def test_shit_img(shift_x, shift_y):
    img = np.random.randint(0, 100, (3, 5))
    shifted_img = shift_img(img, shift_x=shift_x, shift_y=shift_y)

    assert img.shape == shifted_img.shape

    if shift_x > 0:
        crop = img[:-shift_x, :]
        shifted_crop = shifted_img[shift_x:, :]
    elif shift_x < 0:
        crop = img[-shift_x:, :]
        shifted_crop = shifted_img[:shift_x, :]
    else:
        crop = img
        shifted_crop = shifted_img

    assert crop.shape == shifted_crop.shape

    if shift_y > 0:
        crop = crop[:, :-shift_y]
        shifted_crop = shifted_crop[:, shift_y:]
    elif shift_y < 0:
        crop = crop[:, -shift_y:]
        shifted_crop = shifted_crop[:, :shift_y]
    else:
        pass

    assert crop.shape == shifted_crop.shape
    assert np.all(crop == shifted_crop)


filename_stack1 = '/gpfs01/euler/data/Data/DataJointTestData/xy-RGCs/20220125/2/Pre/SMP_M1_RR_GCL0_chirp_C1.h5'

if os.path.isfile(filename_stack1):
    ch_stacks, wparams = read_h5_utils.load_stacks_and_wparams(filename_stack1, ch_names=('wDataCh0',))
    stack1 = ch_stacks['wDataCh0']
else:
    stack1 = None

params_test_autoshift_real_stacks = [
    (0, -1, 3, stack1, 'corr'),
    (-1, -1, 3, stack1, 'corr'),
    (-2, 2, 5, stack1, 'corr'),
    (0, 0, 3, stack1, 'mse'),
    (1, 0, 3, stack1, 'mse'),
    (0, 1, 3, stack1, 'mse'),
    (1, 1, 3, stack1, 'mse'),
    (-1, 0, 3, stack1, 'mse'),
    (0, -1, 3, stack1, 'mse'),
    (-1, -1, 3, stack1, 'mse'),
    (-2, 2, 5, stack1, 'mse'),
]


@pytest.mark.parametrize("shift_x, shift_y, shift_max, stack, metric", params_test_autoshift_real_stacks)
@pytest.mark.skipif(stack1 is None, reason="No stack1 file found")
def test_autoshift_real_stacks(shift_x, shift_y, shift_max, stack, metric):
    # Calculate local cross-correlation matrix for reference
    ref_corr = compute_corr_map(stack)

    # Introduce shift
    shifted_corr = shift_img(ref_corr, shift_x, shift_y)

    # Calculate the cross-correlation matrix between shifted and reference images
    match_indexes = compute_corr_map_match_indexes(shifted_corr, ref_corr, shift_max)

    obs_shift_x, obs_shift_y = extract_best_shift(match_indexes)

    assert obs_shift_x == shift_x and obs_shift_y == shift_y
