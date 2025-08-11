from functools import lru_cache

import numpy as np


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
    group_labels = np.array([baden_cluster_to_group(c_label) for c_label in cluster_labels])
    super_labels = np.array([baden_group_to_supergroup(g_label) for g_label in group_labels])

    if quality_filter:
        qidx = (cluster_labels > 0) & ((bar_qi > 0.6) | (chirp_qi > 0.45))
    else:
        qidx = np.ones(cluster_labels.size, dtype=bool)

    return (
        cluster_labels[qidx], group_labels[qidx], super_labels[qidx],
        chirp_traces[qidx], chirp_qi[qidx], bar_traces[qidx], bar_qi[qidx], bar_dsi[qidx], bar_dp[qidx],
        roi_size_um2[qidx],
    )


def baden_cluster_to_cluster_name(cluster_label):
    _baden_cluster_to_cluster_name = {
        1: "1",
        2: "2",
        3: "3",
        4: "4a", 5: "4b",
        6: "5a", 7: "5b", 8: "5c",
        9: "6",
        10: "7",
        11: "8a", 12: "8b",
        13: "9",
        14: "10",
        15: "11a", 16: "11b",
        17: "12a", 18: "12b",
        19: "13",
        20: "14",
        21: "15",
        22: "16",
        23: "17a", 24: "17b", 25: "17c",
        26: "18a", 27: "18b",
        28: "19",
        29: "20",
        30: "21",
        31: "22a", 32: "22b",
        33: "23",
        34: "24",
        35: "25",
        36: "26",
        37: "27",
        38: "28a", 39: "28b",
        40: "29",
        41: "30",
        42: "31a", 43: "31b", 44: "31c", 45: "31d", 46: "31e",
        47: "32a", 48: "32b", 49: "32c",
        50: "33",
        51: "34a", 52: "34b",
        53: "35a", 54: "35b",
        55: "36",
        56: "37a", 57: "37b",
        58: "38a", 59: "38b", 60: "38c",
        61: "39",
        62: "40a", 63: "40b",
        64: "41",
        65: "42a", 66: "42b", 67: "42c", 68: "42d", 69: "42e", 70: "42f",
        71: "43",
        72: "44",
        73: "45",
        74: "46a", 75: "46b",
    }

    return _baden_cluster_to_cluster_name.get(cluster_label, -1)


def baden_cluster_to_group(cluster_label):
    _baden_cluster_to_group = {
        1: 1,
        2: 2,
        3: 3,
        4: 4, 5: 4,
        6: 5, 7: 5, 8: 5,
        9: 6,
        10: 7,
        11: 8, 12: 8,
        13: 9,
        14: 10,
        15: 11, 16: 11,
        17: 12, 18: 12,
        19: 13,
        20: 14,
        21: 15,
        22: 16,
        23: 17, 24: 17, 25: 17,
        26: 18, 27: 18,
        28: 19,
        29: 20,
        30: 21,
        31: 22, 32: 22,
        33: 23,
        34: 24,
        35: 25,
        36: 26,
        37: 27,
        38: 28, 39: 28,
        40: 29,
        41: 30,
        42: 31, 43: 31, 44: 31, 45: 31, 46: 31,
        47: 32, 48: 32, 49: 32,
        50: 33,
        51: 34, 52: 34,
        53: 35, 54: 35,
        55: 36,
        56: 37, 57: 37,
        58: 38, 59: 38, 60: 38,
        61: 39,
        62: 40, 63: 40,
        64: 41,
        65: 42, 66: 42, 67: 42, 68: 42, 69: 42, 70: 42,
        71: 43,
        72: 44,
        73: 45,
        74: 46, 75: 46,
    }

    return _baden_cluster_to_group.get(cluster_label, -1)


def baden_group_to_supergroup(group_label):
    if group_label <= 0:
        return -1
    elif group_label <= 9:
        return 1  # Off
    elif group_label <= 14:
        return 2  # On Off
    elif group_label <= 20:
        return 3  # Fast On
    elif group_label <= 28:
        return 4  # Slow On
    elif group_label <= 32:
        return 5  # Uncertain
    elif group_label <= 46:
        return 6  # dAC
    else:
        return -1


def supergroup2str(supergroup_label):
    if supergroup_label <= 0:
        return 'none'
    elif supergroup_label == 1:
        return 'Off'
    elif supergroup_label == 2:
        return 'On Off'
    elif supergroup_label == 3:
        return 'Fast On'
    elif supergroup_label == 4:
        return 'Slow On'
    elif supergroup_label == 5:
        return 'Uncertain'
    elif supergroup_label == 6:
        return 'dAC'
    else:
        return 'Unknown'
