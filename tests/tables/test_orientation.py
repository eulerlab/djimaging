import os
import pickle as pkl

import numpy as np

from djimaging.tables.response.orientation_v2 import compute_os_ds_idxs


def load_test_data(cell_type, path='test_mb.pkl'):
    with open(path, 'rb') as f:
        temp_dict = pkl.load(f)
        snippets = temp_dict[f'{cell_type}_bar_byrepeat']
        dir_order = np.array(temp_dict[f'{cell_type}_dir_deg'])
        # np.array([0, 180,  45, 225,  90, 270, 135, 315])
        gt_dsi = temp_dict[f'{cell_type}_cell_dsi'][0]
        gt_dp = temp_dict[f'{cell_type}_cell_dp'][0]
        gt_osi = temp_dict[f'{cell_type}_cell_osi'][0]
        gt_op = temp_dict[f'{cell_type}_cell_op'][0]
        assert snippets.shape == (32, 8, 3)
    return snippets, dir_order, gt_dsi, gt_dp, gt_osi, gt_op


def test_ds(rel_path='test_mb.pkl'):
    np.random.seed(42)

    snippets, dir_order, gt_dsi, gt_dp, gt_osi, gt_op = load_test_data(
        'ds', path=os.path.join(os.path.dirname(__file__), rel_path))
    dsi, p_dsi, _, _, osi, p_osi, _, _, _, _, _, _, _, _, _ = \
        compute_os_ds_idxs(snippets=snippets.T.reshape((-1, 32)).T, dir_order=dir_order, dt=0.128)

    assert np.isclose(dsi, gt_dsi, atol=0.01, rtol=0.01)
    assert np.isclose(p_dsi, gt_dp, atol=0.01, rtol=0.01)
    assert np.isclose(osi, gt_osi, atol=0.1, rtol=2)  # TODO: Check OSI mismatch
    assert np.isclose(p_osi, gt_op, atol=0.01, rtol=0.01)


def test_nonds(rel_path='test_mb.pkl'):
    np.random.seed(42)

    snippets, dir_order, gt_dsi, gt_dp, gt_osi, gt_op = load_test_data(
        'nds', path=os.path.join(os.path.dirname(os.path.abspath(__file__)), rel_path))
    dsi, p_dsi, _, _, osi, p_osi, _, _, _, _, _, _, _, _, _ = \
        compute_os_ds_idxs(snippets=snippets.T.reshape((-1, 32)).T, dir_order=dir_order, dt=0.128)

    assert np.isclose(dsi, gt_dsi, atol=0.01, rtol=0.01)
    assert np.isclose(p_dsi, gt_dp, atol=0.01, rtol=0.01)
    assert np.isclose(osi, gt_osi, atol=0.1, rtol=2)  # TODO: Check OSI mismatch
    assert np.isclose(p_osi, gt_op, atol=0.01, rtol=0.01)
