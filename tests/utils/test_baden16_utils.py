import numpy as np
import pytest

from djimaging.utils import baden16_utils as bu

N_CLUSTERS = 75
N_GROUPS = 46


def test_cluster_info_table_consistent():
    info = bu.BADEN_CLUSTER_INFO
    assert info.shape == (N_CLUSTERS, 4)
    np.testing.assert_array_equal(info[:, 0].astype(int), np.arange(1, N_CLUSTERS + 1))
    group_ids = info[:, 2].astype(int)
    assert np.all(np.diff(group_ids) >= 0)
    assert set(group_ids) == set(range(1, N_GROUPS + 1))
    assert set(bu.BADEN_GROUP_ID_NAMES) == set(range(1, N_GROUPS + 1))
    assert len(set(info[:, 1])) == N_CLUSTERS  # unique cluster names
    # each group belongs to exactly one supergroup
    for g in range(1, N_GROUPS + 1):
        assert len(set(info[group_ids == g, 3])) == 1


@pytest.mark.parametrize('cluster_id', range(1, N_CLUSTERS + 1))
def test_cluster_name_round_trip(cluster_id):
    name = bu.baden_cluster_id_to_cluster_name(cluster_id)
    assert bu.baden_cluster_name_to_cluster_id(name) == cluster_id
    group_id = bu.baden_cluster_id_to_group_id(cluster_id)
    # the cluster name starts with the group number, e.g. 4a -> group 4
    assert int(name.rstrip('abcdef')) == group_id
    assert bu.baden_cluster_id_to_supergroup(cluster_id) == bu.baden_group_id_to_supergroup(group_id)


@pytest.mark.parametrize('cluster_id, name, group_id, supergroup', [
    (1, '1', 1, 'OFF'),
    (5, '4b', 4, 'OFF'),
    (14, '10', 10, 'ON-OFF'),
    (21, '15', 15, 'Fast ON'),
    (39, '28b', 28, 'Slow ON'),
    (41, '30', 30, 'Unc. ON'),
    (46, '31e', 31, 'Unc. SbC'),
    (50, '33', 33, 'dAC'),
    (75, '46b', 46, 'dAC'),
])
def test_known_clusters(cluster_id, name, group_id, supergroup):
    assert bu.baden_cluster_id_to_cluster_name(cluster_id) == name
    assert bu.baden_cluster_id_to_group_id(cluster_id) == group_id
    assert bu.baden_cluster_id_to_supergroup(cluster_id) == supergroup
    assert bu.baden_group_id_to_supergroup(group_id) == supergroup


def test_supergroup_ranges():
    expected = {range(1, 10): 'OFF', range(10, 15): 'ON-OFF', range(15, 21): 'Fast ON', range(21, 29): 'Slow ON',
                range(29, 31): 'Unc. ON', range(31, 33): 'Unc. SbC', range(33, 47): 'dAC'}
    for groups, supergroup in expected.items():
        for g in groups:
            assert bu.baden_group_id_to_supergroup(g) == supergroup


@pytest.mark.parametrize('bad_id', [0, -1, 76, 1000])
def test_out_of_range_cluster_ids(bad_id):
    assert bu.baden_cluster_id_to_cluster_name(bad_id) == 'Unknown'
    assert bu.baden_cluster_id_to_group_id(bad_id) == -1
    assert bu.baden_cluster_id_to_supergroup(bad_id) == 'Unknown'


@pytest.mark.parametrize('bad_id', [0, -1, 47])
def test_out_of_range_group_ids(bad_id):
    assert bu.baden_group_id_to_supergroup(bad_id) == 'Unknown'
    assert bu.baden_group_id_to_group_name(bad_id) == 'Unknown'


def test_unknown_name_and_numpy_inputs():
    assert bu.baden_cluster_name_to_cluster_id('Unknown') == -1
    assert bu.baden_cluster_id_to_group_id(np.int64(5)) == 4
    assert bu.baden_cluster_id_to_group_id(5.0) == 4
    assert bu.baden_cluster_id_to_group_id('5') == 4
    with pytest.raises(IndexError):
        bu.baden_cluster_name_to_cluster_id('99z')


def test_group_names_and_shortening():
    assert bu.baden_group_id_to_group_name(5) == 'Off alpha sustained'
    assert bu.baden_group_id_to_group_name(5, shorten=True) == 'Off alpha sus.'
    assert bu.baden_group_id_to_group_name(34, shorten=True) == 'On high freq. sus. 1'
    assert bu.shorten_baden_name('Off suppressed 1') == 'Off suppr. 1'
    assert bu.shorten_baden_name('On transient') == 'On trans.'
    assert bu.shorten_baden_name('Off') == 'Off'


def _write_baden_mat(path):
    from scipy.io import savemat
    n = 4
    data = {
        'info': {'area2d': np.array([10., 20., 30., 40.]), 'final_idx': np.array([1, 0, 30, 75])},
        'chirp': {'traces': np.arange(5 * n, dtype=float).reshape(5, n), 'qi': np.array([0.5, 0.9, 0.1, 0.1])},
        'ds': {'tc': np.arange(3 * n, dtype=float).reshape(3, n), 'qi': np.array([0.1, 0.9, 0.7, 0.1]),
               'dsi': np.array([.1, .2, .3, .4]), 'dP': np.array([.01, .02, .03, .04])},
    }
    savemat(path, {'data': data})
    return data


def test_load_baden_data_quality_filter(tmp_path):
    path = str(tmp_path / 'baden.mat')
    data = _write_baden_mat(path)
    (cluster, group, superg, chirp, chirp_qi, bar, bar_qi, bar_dsi, bar_dp, area) = bu.load_baden_data(path)
    # cell 0: chirp qi > 0.45; cell 1: unclassified (0); cell 2: bar qi > 0.6; cell 3: low quality
    np.testing.assert_array_equal(cluster, [1, 30])
    np.testing.assert_array_equal(group, [1, 21])
    np.testing.assert_array_equal(superg, ['OFF', 'Slow ON'])
    # traces are stored (time, cells) and returned (cells, time)
    np.testing.assert_array_equal(chirp, data['chirp']['traces'].T[[0, 2]])
    np.testing.assert_array_equal(bar, data['ds']['tc'].T[[0, 2]])
    np.testing.assert_array_equal(chirp_qi, [0.5, 0.1])
    np.testing.assert_array_equal(bar_qi, [0.1, 0.7])
    np.testing.assert_array_equal(bar_dsi, [0.1, 0.3])
    np.testing.assert_array_equal(bar_dp, [0.01, 0.03])
    np.testing.assert_array_equal(area, [10., 30.])


def test_load_baden_data_without_filter(tmp_path):
    path = str(tmp_path / 'baden_all.mat')
    _write_baden_mat(path)
    out = bu.load_baden_data(path, quality_filter=False)
    np.testing.assert_array_equal(out[0], [1, 0, 30, 75])
    np.testing.assert_array_equal(out[1], [1, -1, 21, 46])
    np.testing.assert_array_equal(out[2], ['OFF', 'Unknown', 'Slow ON', 'dAC'])
    assert out[3].shape == (4, 5) and out[5].shape == (4, 3)
    assert all(len(o) == 4 for o in out)
