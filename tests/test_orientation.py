import pickle as pkl

import numpy as np

from djimaging.tables.optional.orientation import compute_osdsindexes


def load_test_data(cell_type):
    with open('testdata/test_mb.pkl', 'rb') as f:
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


def test_ds():
    # TODO: Find out why gt is different
    np.random.seed(42)

    snippets, dir_order, gt_dsi, gt_dp, gt_osi, gt_op = load_test_data('ds')
    dsi, p_dsi, _, _, osi, p_osi, _, _, _, _, _, _, _, _, _ = \
        compute_osdsindexes(snippets=snippets.T.reshape((-1, 32)).T, dir_order=dir_order)

    assert np.isclose(dsi, 0.4797128907324086, atol=0.01, rtol=0.01)
    assert np.isclose(p_dsi, 0.04, atol=0.01, rtol=0.01)
    assert np.isclose(osi, 0.02462625554525927, atol=0.01, rtol=0.01)
    assert np.isclose(p_osi, 0.996, atol=0.01, rtol=0.01)


def test_nonds():
    # TODO: Find out why gt is different
    np.random.seed(42)

    snippets, dir_order, gt_dsi, gt_dp, gt_osi, gt_op = load_test_data('nds')
    dsi, p_dsi, _, _, osi, p_osi, _, _, _, _, _, _, _, _, _ = \
        compute_osdsindexes(snippets=snippets.T.reshape((-1, 32)).T, dir_order=dir_order)

    assert np.isclose(dsi, 0.05624592170528586, atol=0.01, rtol=0.01)
    assert np.isclose(p_dsi, 0.982, atol=0.01, rtol=0.01)
    assert np.isclose(osi, 0.7038211435585239, atol=0.01, rtol=0.01)
    assert np.isclose(p_osi, 0.002, atol=0.01, rtol=0.01)


