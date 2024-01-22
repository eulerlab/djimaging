from djimaging.utils.scanm import read_smp_utils, read_h5_utils


def load_stacks(filepath, from_raw_data, ch_names=('wDataCh0', 'wDataCh1')) -> (dict, dict):
    if from_raw_data:
        ch_stacks, wparams = read_smp_utils.load_stacks_and_wparams(filepath, ch_names=ch_names)
    else:
        ch_stacks, wparams = read_h5_utils.load_stacks_and_wparams(filepath, ch_names=ch_names)
    return ch_stacks, wparams
