from djimaging.tables.optional.orientation import *
from djimaging.utils.math_utils import normalize_zero_one

import numpy as np
import pickle as pkl
import pytest


@pytest.mark.xfail
def test_ds():
    cell_type = f'ds'
    with open('testdata/test_mb.pkl', 'rb') as f:
        temp_dict = pkl.load(f)
    snippets = temp_dict[f'{cell_type}_bar_byrepeat']
    dir_order = np.array(temp_dict[f'{cell_type}_dir_deg'])  # np.array([0, 180,  45, 225,  90, 270, 135, 315])
    # 'ds_cell_dsi', 'ds_cell_dp', 'ds_cell_osi', 'ds_cell_op'
    gt_dsi = temp_dict[f'{cell_type}_cell_dsi']
    gt_dp = temp_dict[f'{cell_type}_cell_dp']
    gt_osi = temp_dict[f'{cell_type}_cell_osi']
    gt_op = temp_dict[f'{cell_type}_cell_op']
    # assert snippets.shape == (32,8,3)

    # snippets = np.reshape(snippets, (snippets.shape[0],-1))
    # print (snippets.shape)
    sorted_responses, sorted_directions_rad = sort_response_matrix(snippets, dir_order)
    avg_sorted_responses = np.mean(sorted_responses, axis=-1)
    try:
        u, v, s = get_time_dir_kernels(avg_sorted_responses)
    except np.linalg.LinAlgError:
        print(f'ERROR: LinAlgError')
        return

    dsi, pref_dir = get_si(v, sorted_directions_rad, 1)
    osi, pref_or = get_si(v, sorted_directions_rad, 2)

    (t, d, r) = sorted_responses.shape
    # make the result between the original and the shuffled comparable
    projected = np.dot(np.transpose(np.reshape(sorted_responses, (t, d * r))), u)
    projected = np.reshape(projected, (d, r))
    surrogate_v = normalize_zero_one(np.mean(projected, axis=-1))

    dsi_s, pref_dir_s = get_si(surrogate_v, sorted_directions_rad, 1)
    osi_s, pref_or_s = get_si(surrogate_v, sorted_directions_rad, 2)
    null_dist_dsi = compute_null_dist(np.transpose(projected), sorted_directions_rad, 1)
    p_dsi = np.mean(null_dist_dsi > dsi_s)
    null_dist_osi = compute_null_dist(np.transpose(projected), sorted_directions_rad, 2)
    p_osi = np.mean(null_dist_osi > osi_s)
    d_qi = quality_index_ds(sorted_responses)

    print(dsi, p_dsi, osi, p_osi)
    print(gt_dsi[0], gt_dp[0], gt_osi[0], gt_op[0])

    assert (dsi, p_dsi, osi, p_osi) == (gt_dsi[0], gt_dp[0], gt_osi[0], gt_op[0])


@pytest.mark.xfail
def test_nonds():
    cell_type = f'nds'
    with open('testdata/test_mb.pkl', 'rb') as f:
        temp_dict = pkl.load(f)
    snippets = temp_dict[f'{cell_type}_bar_byrepeat']
    dir_order = np.array(temp_dict[f'{cell_type}_dir_deg'])  # np.array([0, 180,  45, 225,  90, 270, 135, 315])
    # 'ds_cell_dsi', 'ds_cell_dp', 'ds_cell_osi', 'ds_cell_op'
    gt_dsi = temp_dict[f'{cell_type}_cell_dsi']
    gt_dp = temp_dict[f'{cell_type}_cell_dp']
    gt_osi = temp_dict[f'{cell_type}_cell_osi']
    gt_op = temp_dict[f'{cell_type}_cell_op']
    # assert snippets.shape == (32,8,3)

    # snippets = np.reshape(snippets, (snippets.shape[0],-1))
    # print (snippets.shape)
    sorted_responses, sorted_directions_rad = sort_response_matrix(snippets, dir_order)
    avg_sorted_responses = np.mean(sorted_responses, axis=-1)
    try:
        u, v, s = get_time_dir_kernels(avg_sorted_responses)
    except np.linalg.LinAlgError:
        print(f'ERROR: LinAlgError')
        return

    dsi, pref_dir = get_si(v, sorted_directions_rad, 1)
    osi, pref_or = get_si(v, sorted_directions_rad, 2)

    (t, d, r) = sorted_responses.shape
    # make the result between the original and the shuffled comparable
    projected = np.dot(np.transpose(np.reshape(sorted_responses, (t, d * r))), u)
    projected = np.reshape(projected, (d, r))
    surrogate_v = normalize_zero_one(np.mean(projected, axis=-1))

    dsi_s, pref_dir_s = get_si(surrogate_v, sorted_directions_rad, 1)
    osi_s, pref_or_s = get_si(surrogate_v, sorted_directions_rad, 2)
    null_dist_dsi = compute_null_dist(np.transpose(projected), sorted_directions_rad, 1)
    p_dsi = np.mean(null_dist_dsi > dsi_s)
    null_dist_osi = compute_null_dist(np.transpose(projected), sorted_directions_rad, 2)
    p_osi = np.mean(null_dist_osi > osi_s)
    d_qi = quality_index_ds(sorted_responses)

    print(dsi, p_dsi, osi, p_osi)
    print(gt_dsi[0], gt_dp[0], gt_osi[0], gt_op[0])

    assert (dsi, p_dsi, osi, p_osi) == (gt_dsi[0], gt_dp[0], gt_osi[0], gt_op[0])
