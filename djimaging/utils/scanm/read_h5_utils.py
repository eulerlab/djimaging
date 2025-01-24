from configparser import ConfigParser

import h5py
import numpy as np

from djimaging.utils.scanm.wparams_utils import check_dims_ch_stack_wparams
from djimaging.utils.scanm.roi_utils import get_roi2trace


def load_stacks_and_wparams(filepath, ch_names=('wDataCh0', 'wDataCh1')) -> (dict, dict):
    """Load stacks from h5 file"""
    try:
        with h5py.File(filepath, 'r', driver="stdio") as h5_file:
            ch_stacks = extract_stacks(h5_file, ch_names=ch_names)
            wparams = extract_wparams(h5_file)
    except OSError as e:
        raise OSError(f"Error loading file {filepath}: {e}")

    for name, stack in ch_stacks.items():
        check_dims_ch_stack_wparams(ch_stack=stack, wparams=wparams)

    return ch_stacks, wparams


def load_traces(filepath):
    """Extract traces from ScanM h5 file"""
    try:
        with h5py.File(filepath, "r", driver="stdio") as h5_file:
            traces, traces_times = extract_traces(h5_file)
    except OSError as e:
        raise OSError(f"Error loading file {filepath}: {e}")
    return traces, traces_times


def load_triggers(filepath):
    try:
        with h5py.File(filepath, 'r', driver="stdio") as h5_file:
            triggertimes, triggervalues = extract_triggers(h5_file)
    except OSError as e:
        raise OSError(f"Error loading file {filepath}: {e}")
    return triggertimes, triggervalues


def load_roi_mask(filepath, ignore_not_found=False):
    try:
        with h5py.File(filepath, 'r', driver="stdio") as h5_file:
            roi_mask = extract_roi_mask(h5_file, ignore_not_found=ignore_not_found)
    except OSError as e:
        raise OSError(f"Error loading file {filepath}: {e}")
    return roi_mask


def load_roi2trace(filepath: str, roi_ids: np.ndarray):
    try:
        with h5py.File(filepath, "r", driver="stdio") as h5_file:
            traces, traces_times = extract_traces(h5_file)
    except OSError as e:
        raise OSError(f"Error loading file {filepath}: {e}")
    roi2trace = get_roi2trace(traces=traces, traces_times=traces_times, roi_ids=roi_ids)
    frame_dt = np.mean(np.diff(traces_times, axis=0))
    return roi2trace, frame_dt


def extract_stacks(h5_file_open, ch_names=('wDataCh0', 'wDataCh1')) -> dict:
    ch_stacks = {ch_name: np.copy(h5_file_open[ch_name]) for ch_name in ch_names}
    for name, stack in ch_stacks.items():
        if stack.ndim != 3:
            raise ValueError(f"stack must be 3d but ndim={stack.ndim}")
        if stack.shape != ch_stacks[ch_names[0]].shape:
            raise ValueError('Stacks must be of equal size')
    return ch_stacks


def extract_all_stack_from_h5(h5_file_open) -> dict:
    ch_names = [ch_name for ch_name in h5_file_open.keys() if ch_name.startswith('wDataCh')]
    ch_stacks = extract_stacks(h5_file_open, ch_names=ch_names)
    return ch_stacks


def extract_roi_mask(h5_file_open, ignore_not_found=False):
    roi_keys = [k for k in h5_file_open.keys() if 'rois' in k.lower()]

    roi_mask = None
    if len(roi_keys) > 1:
        raise KeyError('Multiple ROI masks found in single file')
    elif len(roi_keys) == 1:
        roi_mask = np.copy(h5_file_open[roi_keys[0]])
        if np.all(roi_mask.astype(int) == roi_mask):
            roi_mask = roi_mask.astype(int)

        if roi_mask.size == 0:
            roi_mask = None

    if (roi_mask is None) and (not ignore_not_found):
        raise KeyError('No ROI mask found in single file')

    return roi_mask


def extract_wparams(h5_file_open, lower_keys=True):
    wparams = dict()
    wparamsstr_key = [k for k in h5_file_open.keys() if k.lower() == 'wparamsstr']
    if len(wparamsstr_key) > 0:
        wparams.update(extract_h5_table(wparamsstr_key[0], open_file=h5_file_open, lower_keys=lower_keys))
    wparamsnum_key = [k for k in h5_file_open.keys() if k.lower() == 'wparamsnum']
    if len(wparamsnum_key) > 0:
        wparams.update(extract_h5_table(wparamsnum_key[0], open_file=h5_file_open, lower_keys=lower_keys))
    return wparams


def extract_os_params(h5_file_open, lower_keys=True) -> dict:
    os_params = dict()
    os_params_key = [k for k in h5_file_open.keys() if k.lower() == 'os_parameters']
    if len(os_params_key) > 0:
        os_params.update(extract_h5_table(os_params_key[0], open_file=h5_file_open, lower_keys=lower_keys))
    return os_params


def extract_triggers(h5_file_open, check_triggervalues=False, ignore_not_found=True):
    key_triggertimes = [k for k in h5_file_open.keys() if k.lower() == 'triggertimes']

    if len(key_triggertimes) == 1:
        triggertimes = h5_file_open[key_triggertimes[0]][()]
    elif len(key_triggertimes) == 0:
        if not ignore_not_found:
            raise KeyError('No triggertimes found')
        triggertimes = np.zeros(0)
    else:
        raise ValueError('Multiple triggertimes found')

    if triggertimes.size > 0:
        if "Tracetimes0" in h5_file_open.keys():  # Correct if triggertimes are in frames and not in time (old version)
            traces_times = np.asarray(h5_file_open["Tracetimes0"][()])
            correct_triggertimes = triggertimes[-1] > 2 * np.max(traces_times)
        else:
            correct_triggertimes = triggertimes[0] > 250 and triggertimes[-1] > 1000

        if correct_triggertimes:
            triggertimes = triggertimes / 500.

    key_triggervalues = [k for k in h5_file_open.keys() if k.lower() == 'triggervalues']

    if len(key_triggervalues) == 1:
        triggervalues = h5_file_open[key_triggervalues[0]][()]
    elif len(key_triggervalues) == 0:
        triggervalues = np.zeros(0)
    else:
        raise ValueError('Multiple triggervalues found')

    if check_triggervalues:
        assert len(triggertimes) == len(triggervalues), f'Mismatch: {len(triggertimes)} != {len(triggervalues)}'

    return triggertimes, triggervalues


def extract_traces(h5_file_open):
    """Read all traces and their times from file"""
    if "Traces0_raw" in h5_file_open.keys() and "Tracetimes0" in h5_file_open.keys():
        traces = np.asarray(h5_file_open["Traces0_raw"][()])
        traces_times = np.asarray(h5_file_open["Tracetimes0"][()])
    else:
        raise ValueError('Traces not found in h5 file.')

    assert traces.shape == traces_times.shape, 'Inconsistent traces and tracetimes shapes'

    return traces, traces_times


def load_h5_data(filename, lower_keys=False):
    """Helper function to load h5 file."""
    with h5py.File(filename, 'r') as f:
        return {key.lower() if lower_keys else key: f[key][:] for key in list(f.keys())}


def load_h5_table(*tablename, filename, lower_keys=False):
    """Load h5 tables from a filename"""
    with h5py.File(filename, 'r', driver="stdio") as f:
        data = extract_h5_table(*tablename, open_file=f, lower_keys=lower_keys)
    return data


def extract_h5_table(*tablename, open_file, lower_keys=False):
    """Load h5 table from an open h5 file"""
    data_dict = dict()
    for name in tablename:
        keys = [v[0] for v in list(open_file[name].attrs.values())[0][1:]]
        values = open_file[name][:]

        assert len(keys) == len(values), 'Lengths do not match'

        for key, value in zip(keys, values):
            if len(key) > 0:
                if type(key) == bytes:
                    key = key.decode('utf-8')
                assert type(key) == str, type(key)

                if lower_keys:
                    key = key.lower()
                data_dict[key] = value
    return data_dict


def read_config_dict(filename):
    config_dict = dict()
    parser = ConfigParser()
    parser.read(filename)
    for key1 in parser.keys():
        for key2 in parser[key1].keys():
            config_dict[key2[key2.find("_") + 1:]] = str(parser[key1][key2])
    return config_dict
