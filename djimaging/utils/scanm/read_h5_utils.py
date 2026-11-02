from configparser import ConfigParser
from typing import Optional, Tuple

import h5py
import numpy as np

from djimaging.utils.scanm.wparams_utils import check_dims_ch_stack_wparams
from djimaging.utils.scanm.roi_utils import get_roi2trace, extract_roi_ids


def load_stacks_and_wparams(filepath: str, ch_names: tuple = ('wDataCh0', 'wDataCh1')) -> Tuple[dict, dict]:
    """Load channel stacks and scan parameters from an h5 file.

    Parameters
    ----------
    filepath : str
        Path to the .h5 recording file.
    ch_names : tuple, optional
        Names of the channel datasets to load. Default is
        ``('wDataCh0', 'wDataCh1')``.

    Returns
    -------
    ch_stacks : dict
        Mapping from channel name to 3-D numpy array ``(x, y, frames)``.
    wparams : dict
        Scan parameters extracted from the file.

    Raises
    ------
    OSError
        If the file cannot be opened.
    ValueError
        If a stack is not 3-D or stacks have inconsistent shapes.
    """
    try:
        with h5py.File(filepath, 'r', driver="stdio") as h5_file:
            ch_stacks = extract_stacks(h5_file, ch_names=ch_names)
            wparams = extract_wparams(h5_file)
    except OSError as e:
        raise OSError(f"Error loading file {filepath}: {e}")

    for name, stack in ch_stacks.items():
        check_dims_ch_stack_wparams(ch_stack=stack, wparams=wparams)

    return ch_stacks, wparams


def load_traces(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Extract fluorescence traces and their timestamps from a ScanM h5 file.

    Parameters
    ----------
    filepath : str
        Path to the .h5 recording file.

    Returns
    -------
    traces : np.ndarray
        Array of shape ``(frames, rois)`` containing trace values.
    traces_times : np.ndarray
        Array of the same shape as ``traces`` with per-sample timestamps in
        seconds.

    Raises
    ------
    OSError
        If the file cannot be opened.
    ValueError
        If the expected trace datasets are absent or shapes are inconsistent.
    """
    try:
        with h5py.File(filepath, "r", driver="stdio") as h5_file:
            traces, traces_times = extract_traces(h5_file)
    except OSError as e:
        raise OSError(f"Error loading file {filepath}: {e}")
    return traces, traces_times


def load_triggers(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load trigger times and values from a ScanM h5 file.

    Parameters
    ----------
    filepath : str
        Path to the .h5 recording file.

    Returns
    -------
    triggertimes : np.ndarray
        1-D array of trigger onset times in seconds.
    triggervalues : np.ndarray
        1-D array of signal amplitudes at each trigger onset.

    Raises
    ------
    OSError
        If the file cannot be opened.
    """
    try:
        with h5py.File(filepath, 'r', driver="stdio") as h5_file:
            triggertimes, triggervalues = extract_triggers(h5_file)
    except OSError as e:
        raise OSError(f"Error loading file {filepath}: {e}")
    return triggertimes, triggervalues


def load_roi_mask(filepath: str, ignore_not_found: bool = False) -> Optional[np.ndarray]:
    """Load the ROI mask from a ScanM h5 file.

    Parameters
    ----------
    filepath : str
        Path to the .h5 recording file.
    ignore_not_found : bool, optional
        If True, return None when no ROI mask is found instead of raising.
        Default is False.

    Returns
    -------
    roi_mask : np.ndarray or None
        2-D integer array encoding ROI membership, or None if not present
        and ``ignore_not_found`` is True.

    Raises
    ------
    OSError
        If the file cannot be opened.
    KeyError
        If no ROI mask is found and ``ignore_not_found`` is False.
    """
    try:
        with h5py.File(filepath, 'r', driver="stdio") as h5_file:
            roi_mask = extract_roi_mask(h5_file, ignore_not_found=ignore_not_found)
    except OSError as e:
        raise OSError(f"Error loading file {filepath}: {e}")
    return roi_mask


def load_roi2trace(filepath: str, roi_ids: np.ndarray) -> Tuple[dict, float]:
    """Load traces for a subset of ROIs from a ScanM h5 file.

    Parameters
    ----------
    filepath : str
        Path to the .h5 recording file.
    roi_ids : np.ndarray
        1-D array of positive integer ROI IDs to extract.

    Returns
    -------
    roi2trace : dict
        Mapping from ROI ID (int) to a dict with keys ``'trace'``,
        ``'trace_times'``, and ``'trace_valid'``.
    frame_dt : float
        Mean frame duration in seconds estimated from the trace timestamps.

    Raises
    ------
    OSError
        If the file cannot be opened.
    ValueError
        If ``roi_ids`` are not a subset of the ROI IDs present in the file,
        or if the number of ROI IDs does not match the trace array.
    """
    try:
        with h5py.File(filepath, "r", driver="stdio") as h5_file:
            traces, traces_times = extract_traces(h5_file)
            roi_ids_traces = extract_roi_ids(extract_roi_mask(h5_file, ignore_not_found=False), npixartifact=0)
    except OSError as e:
        raise OSError(f"Error loading file {filepath}: {e}")

    if not set(roi_ids).issubset(set(roi_ids_traces)):
        raise ValueError(f"roi_ids {roi_ids} do not match roi_ids in traces {roi_ids_traces}")
    if traces.shape[-1] != len(roi_ids_traces):
        raise ValueError(f"Number of roi_ids {len(roi_ids_traces)} does not match traces shape {traces.shape[-1]}.")

    roi2trace = get_roi2trace(traces=traces, traces_times=traces_times,
                              roi_ids_traces=roi_ids_traces, roi_ids_subset=roi_ids)
    frame_dt = np.mean(np.diff(traces_times, axis=0))
    return roi2trace, frame_dt


def extract_stacks(h5_file_open: h5py.File, ch_names: tuple = ('wDataCh0', 'wDataCh1')) -> dict:
    """Extract named channel stacks from an open h5 file.

    Parameters
    ----------
    h5_file_open : h5py.File
        An already-opened h5py file object.
    ch_names : tuple, optional
        Dataset names to read. Default is ``('wDataCh0', 'wDataCh1')``.

    Returns
    -------
    dict
        Mapping from channel name to 3-D numpy array ``(x, y, frames)``.

    Raises
    ------
    ValueError
        If any stack is not 3-D or if stacks have differing shapes.
    """
    ch_stacks = {ch_name: np.copy(h5_file_open[ch_name]) for ch_name in ch_names}
    for name, stack in ch_stacks.items():
        if stack.ndim != 3:
            raise ValueError(f"stack must be 3d but ndim={stack.ndim}")
        if stack.shape != ch_stacks[ch_names[0]].shape:
            raise ValueError('Stacks must be of equal size')
    return ch_stacks


def extract_all_stack_from_h5(h5_file_open: h5py.File) -> dict:
    """Extract all ``wDataCh*`` channel stacks from an open h5 file.

    Parameters
    ----------
    h5_file_open : h5py.File
        An already-opened h5py file object.

    Returns
    -------
    dict
        Mapping from channel name to 3-D numpy array ``(x, y, frames)``
        for every dataset whose name starts with ``'wDataCh'``.
    """
    ch_names = [ch_name for ch_name in h5_file_open.keys() if ch_name.startswith('wDataCh')]
    ch_stacks = extract_stacks(h5_file_open, ch_names=ch_names)
    return ch_stacks


def extract_roi_mask(h5_file_open: h5py.File, ignore_not_found: bool = False) -> Optional[np.ndarray]:
    """Extract the ROI mask from an open h5 file.

    Parameters
    ----------
    h5_file_open : h5py.File
        An already-opened h5py file object.
    ignore_not_found : bool, optional
        If True, return None when no ROI mask dataset is found.
        Default is False.

    Returns
    -------
    roi_mask : np.ndarray or None
        2-D integer array encoding ROI membership, or None if absent and
        ``ignore_not_found`` is True.

    Raises
    ------
    KeyError
        If multiple ROI datasets are found, or if none are found and
        ``ignore_not_found`` is False.
    """
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


def extract_wparams(h5_file_open: h5py.File, lower_keys: bool = True) -> dict:
    """Extract scan parameters (wParamsStr and wParamsNum) from an open h5 file.

    Parameters
    ----------
    h5_file_open : h5py.File
        An already-opened h5py file object.
    lower_keys : bool, optional
        If True, convert all parameter keys to lower-case. Default is True.

    Returns
    -------
    dict
        Combined dictionary of string and numeric scan parameters.
    """
    wparams = dict()
    wparamsstr_key = [k for k in h5_file_open.keys() if k.lower() == 'wparamsstr']
    if len(wparamsstr_key) > 0:
        wparams.update(extract_h5_table(wparamsstr_key[0], open_file=h5_file_open, lower_keys=lower_keys))
    wparamsnum_key = [k for k in h5_file_open.keys() if k.lower() == 'wparamsnum']
    if len(wparamsnum_key) > 0:
        wparams.update(extract_h5_table(wparamsnum_key[0], open_file=h5_file_open, lower_keys=lower_keys))
    return wparams


def extract_os_params(h5_file_open: h5py.File, lower_keys: bool = True) -> dict:
    """Extract OS parameters from an open h5 file.

    Parameters
    ----------
    h5_file_open : h5py.File
        An already-opened h5py file object.
    lower_keys : bool, optional
        If True, convert all parameter keys to lower-case. Default is True.

    Returns
    -------
    dict
        Dictionary of OS parameter key-value pairs.  Empty if the
        ``'os_parameters'`` dataset is absent.
    """
    os_params = dict()
    os_params_key = [k for k in h5_file_open.keys() if k.lower() == 'os_parameters']
    if len(os_params_key) > 0:
        os_params.update(extract_h5_table(os_params_key[0], open_file=h5_file_open, lower_keys=lower_keys))
    return os_params


def extract_triggers(
        h5_file_open: h5py.File,
        check_triggervalues: bool = False,
        ignore_not_found: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract trigger times and values from an open h5 file.

    Automatically converts frame-index-based trigger times (old file format)
    to seconds.

    Parameters
    ----------
    h5_file_open : h5py.File
        An already-opened h5py file object.
    check_triggervalues : bool, optional
        If True, assert that the number of trigger times equals the number of
        trigger values. Default is False.
    ignore_not_found : bool, optional
        If True, return empty arrays when the ``'triggertimes'`` dataset is
        absent. Default is True.

    Returns
    -------
    triggertimes : np.ndarray
        1-D array of trigger onset times in seconds.
    triggervalues : np.ndarray
        1-D array of signal amplitudes at each trigger onset.

    Raises
    ------
    KeyError
        If ``ignore_not_found`` is False and no trigger times dataset is found.
    ValueError
        If multiple ``'triggertimes'`` or ``'triggervalues'`` datasets are found.
    """
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


def extract_traces(h5_file_open: h5py.File) -> Tuple[np.ndarray, np.ndarray]:
    """Read all fluorescence traces and their timestamps from an open h5 file.

    Parameters
    ----------
    h5_file_open : h5py.File
        An already-opened h5py file object containing ``'Traces0_raw'`` and
        ``'Tracetimes0'`` datasets.

    Returns
    -------
    traces : np.ndarray
        Array of shape ``(frames, rois)`` with fluorescence trace values.
    traces_times : np.ndarray
        Array of the same shape as ``traces`` with per-sample timestamps in
        seconds.

    Raises
    ------
    ValueError
        If the expected datasets are absent or have inconsistent shapes.
    """
    if "Traces0_raw" in h5_file_open.keys() and "Tracetimes0" in h5_file_open.keys():
        traces = np.asarray(h5_file_open["Traces0_raw"][()])
        traces_times = np.asarray(h5_file_open["Tracetimes0"][()])
    else:
        raise ValueError('Traces not found in h5 file.')

    assert traces.shape == traces_times.shape, 'Inconsistent traces and tracetimes shapes'

    return traces, traces_times


def load_h5_data(filename: str, lower_keys: bool = False) -> dict:
    """Load all datasets from an h5 file into a dictionary.

    Parameters
    ----------
    filename : str
        Path to the .h5 file.
    lower_keys : bool, optional
        If True, convert dataset keys to lower-case. Default is False.

    Returns
    -------
    dict
        Mapping from dataset name to numpy array.
    """
    with h5py.File(filename, 'r') as f:
        return {key.lower() if lower_keys else key: f[key][:] for key in list(f.keys())}


def load_h5_table(*tablename: str, filename: str, lower_keys: bool = False) -> dict:
    """Load one or more h5 attribute tables from a file by name.

    Parameters
    ----------
    *tablename : str
        One or more dataset names whose attribute tables should be read.
    filename : str
        Path to the .h5 file.
    lower_keys : bool, optional
        If True, convert all parameter keys to lower-case. Default is False.

    Returns
    -------
    dict
        Combined mapping of parameter names to values from all requested
        tables.
    """
    with h5py.File(filename, 'r', driver="stdio") as f:
        data = extract_h5_table(*tablename, open_file=f, lower_keys=lower_keys)
    return data


def extract_h5_table(*tablename: str, open_file: h5py.File, lower_keys: bool = False) -> dict:
    """Load one or more attribute tables from an already-open h5 file.

    Parameters
    ----------
    *tablename : str
        One or more dataset names whose attribute tables should be read.
    open_file : h5py.File
        An already-opened h5py file object.
    lower_keys : bool, optional
        If True, convert all parameter keys to lower-case. Default is False.

    Returns
    -------
    dict
        Combined mapping of parameter names to values from all requested
        tables.
    """
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


def read_config_dict(filename: str) -> dict:
    """Read a configuration file into a flat dictionary.

    Parameters
    ----------
    filename : str
        Path to the configuration file in INI/ConfigParser format.

    Returns
    -------
    dict
        Flat mapping of parameter names (with the leading ``<section>_``
        prefix stripped) to their string values.
    """
    config_dict = dict()
    parser = ConfigParser()
    parser.read(filename)
    for key1 in parser.keys():
        for key2 in parser[key1].keys():
            config_dict[key2[key2.find("_") + 1:]] = str(parser[key1][key2])
    return config_dict
