from functools import lru_cache

import numpy as np

# Cluster ID, Cluster Name, Group ID, Supergroup
# Don't change this array, it is used in the classifier.
BADEN_CLUSTER_INFO = np.array([
    [1, '1', 1, 'OFF', ],
    [2, '2', 2, 'OFF', ],
    [3, '3', 3, 'OFF', ],
    [4, '4a', 4, 'OFF', ],
    [5, '4b', 4, 'OFF', ],
    [6, '5a', 5, 'OFF', ],
    [7, '5b', 5, 'OFF', ],
    [8, '5c', 5, 'OFF', ],
    [9, '6', 6, 'OFF', ],
    [10, '7', 7, 'OFF', ],
    [11, '8a', 8, 'OFF', ],
    [12, '8b', 8, 'OFF', ],
    [13, '9', 9, 'OFF', ],
    [14, '10', 10, 'ON-OFF', ],
    [15, '11a', 11, 'ON-OFF', ],
    [16, '11b', 11, 'ON-OFF', ],
    [17, '12a', 12, 'ON-OFF', ],
    [18, '12b', 12, 'ON-OFF', ],
    [19, '13', 13, 'ON-OFF', ],
    [20, '14', 14, 'ON-OFF', ],
    [21, '15', 15, 'Fast ON', ],
    [22, '16', 16, 'Fast ON', ],
    [23, '17a', 17, 'Fast ON', ],
    [24, '17b', 17, 'Fast ON', ],
    [25, '17c', 17, 'Fast ON', ],
    [26, '18a', 18, 'Fast ON', ],
    [27, '18b', 18, 'Fast ON', ],
    [28, '19', 19, 'Fast ON', ],
    [29, '20', 20, 'Fast ON', ],
    [30, '21', 21, 'Slow ON', ],
    [31, '22a', 22, 'Slow ON', ],
    [32, '22b', 22, 'Slow ON', ],
    [33, '23', 23, 'Slow ON', ],
    [34, '24', 24, 'Slow ON', ],
    [35, '25', 25, 'Slow ON', ],
    [36, '26', 26, 'Slow ON', ],
    [37, '27', 27, 'Slow ON', ],
    [38, '28a', 28, 'Slow ON', ],
    [39, '28b', 28, 'Slow ON', ],
    [40, '29', 29, 'Unc. ON', ],
    [41, '30', 30, 'Unc. ON', ],
    [42, '31a', 31, 'Unc. SbC', ],
    [43, '31b', 31, 'Unc. SbC', ],
    [44, '31c', 31, 'Unc. SbC', ],
    [45, '31d', 31, 'Unc. SbC', ],
    [46, '31e', 31, 'Unc. SbC', ],
    [47, '32a', 32, 'Unc. SbC', ],
    [48, '32b', 32, 'Unc. SbC', ],
    [49, '32c', 32, 'Unc. SbC', ],
    [50, '33', 33, 'dAC', ],
    [51, '34a', 34, 'dAC', ],
    [52, '34b', 34, 'dAC', ],
    [53, '35a', 35, 'dAC', ],
    [54, '35b', 35, 'dAC', ],
    [55, '36', 36, 'dAC', ],
    [56, '37a', 37, 'dAC', ],
    [57, '37b', 37, 'dAC', ],
    [58, '38a', 38, 'dAC', ],
    [59, '38b', 38, 'dAC', ],
    [60, '38c', 38, 'dAC', ],
    [61, '39', 39, 'dAC', ],
    [62, '40a', 40, 'dAC', ],
    [63, '40b', 40, 'dAC', ],
    [64, '41', 41, 'dAC', ],
    [65, '42a', 42, 'dAC', ],
    [66, '42b', 42, 'dAC', ],
    [67, '42c', 42, 'dAC', ],
    [68, '42d', 42, 'dAC', ],
    [69, '42e', 42, 'dAC', ],
    [70, '42f', 42, 'dAC', ],
    [71, '43', 43, 'dAC', ],
    [72, '44', 44, 'dAC', ],
    [73, '45', 45, 'dAC', ],
    [74, '46a', 46, 'dAC', ],
    [75, '46b', 46, 'dAC', ],
])


@lru_cache(maxsize=None)
def load_baden_data(baden_data_file, quality_filter=True):
    from scipy.io import loadmat
    baden_data = loadmat(baden_data_file, struct_as_record=True, matlab_compatible=False,
                         squeeze_me=True, simplify_cells=True)['data']

    roi_size_um2 = baden_data['info']['area2d']

    chirp_traces = baden_data['chirp']['traces'].T
    chirp_qi = baden_data['chirp']['qi']

    bar_traces = baden_data['ds']['tc'].T
    bar_qi = baden_data['ds']['qi']
    bar_dsi = baden_data['ds']['dsi']
    bar_dp = baden_data['ds']['dP']

    cluster_labels = np.asarray(baden_data['info']['final_idx']).flatten()
    group_labels = np.array([baden_cluster_id_to_group_id(c_label) for c_label in cluster_labels])
    super_labels = np.array([baden_group_id_to_supergroup(g_label) for g_label in group_labels])

    if quality_filter:
        qidx = (cluster_labels > 0) & ((bar_qi > 0.6) | (chirp_qi > 0.45))
    else:
        qidx = np.ones(cluster_labels.size, dtype=bool)

    return (
        cluster_labels[qidx], group_labels[qidx], super_labels[qidx],
        chirp_traces[qidx], chirp_qi[qidx], bar_traces[qidx], bar_qi[qidx], bar_dsi[qidx], bar_dp[qidx],
        roi_size_um2[qidx],
    )


def baden_cluster_id_to_cluster_name(cluster_id):
    cluster_id = int(cluster_id)
    if cluster_id < 1 or cluster_id > 75:
        return 'Unknown'
    cluster_ids = BADEN_CLUSTER_INFO[:, 0].astype(int)
    cluster_names = BADEN_CLUSTER_INFO[:, 1].astype(str)
    i = np.where(cluster_ids == cluster_id)[0][0]
    return cluster_names[i]


def baden_cluster_name_to_cluster_id(cluster_name):
    cluster_name = str(cluster_name)
    if cluster_name == 'Unknown':
        return -1
    cluster_ids = BADEN_CLUSTER_INFO[:, 0].astype(int)
    cluster_names = BADEN_CLUSTER_INFO[:, 1].astype(str)
    i = np.where(cluster_names == cluster_name)[0][0]
    return cluster_ids[i]


def baden_cluster_id_to_group_id(cluster_id):
    cluster_id = int(cluster_id)
    if cluster_id < 1 or cluster_id > 75:
        return -1
    cluster_ids = BADEN_CLUSTER_INFO[:, 0].astype(int)
    group_ids = BADEN_CLUSTER_INFO[:, 2].astype(int)
    i = np.where(cluster_ids == cluster_id)[0][0]
    return group_ids[i]


def baden_cluster_id_to_supergroup(cluster_id):
    cluster_id = int(cluster_id)
    if cluster_id < 1 or cluster_id > 75:
        return 'Unknown'
    cluster_ids = BADEN_CLUSTER_INFO[:, 0].astype(int)
    supergroups = BADEN_CLUSTER_INFO[:, 3].astype(str)
    i = np.where(cluster_ids == cluster_id)[0][0]
    return supergroups[i]


def baden_group_id_to_supergroup(group_id):
    group_id = int(group_id)
    if group_id < 1 or group_id > 46:
        return 'Unknown'
    group_ids = BADEN_CLUSTER_INFO[:, 2].astype(int)
    supergroups = BADEN_CLUSTER_INFO[:, 3].astype(str)
    i = np.where(group_ids == group_id)[0][0]
    return supergroups[i]
