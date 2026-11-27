import warnings

import numpy as np

from scanmsupport.scanm.scanm_smp import SMP
from djimaging.utils.misc_utils import NoPrints


def load_smp_file(raw_file_path):
    """Load an SMP file from disk.

    Parameters
    ----------
    raw_file_path : str or Path
        Path to the SMP file to load.

    Returns
    -------
    SMP
        Loaded SMP file object.
    """
    raw_file = SMP()

    with NoPrints():
        raw_file.loadSMH(raw_file_path, verbose=False)
        raw_file.loadSMP(raw_file_path)

    return raw_file


# Plausible pixel durations in µs; values outside indicate a wrong unit or a corrupt header
_REAL_PIX_DUR_RANGE_US = (0.01, 1000.)
# Suffixes that clearly name a unit other than µs
_NON_MICRO_SECOND_SUFFIXES = ('s', 'ms', 'ns', 'ps')


def _normalize_real_pixel_duration_key(wparams: dict) -> None:
    """Rename the RealPixelDuration key to 'RealPixelDuration_µs' in place.

    The micro sign in this key is read with varying encodings (µ, �, ...), see read_header_utils._clean_key.
    Any suffix that is not clearly another unit is therefore taken as µs, and the value is checked for a
    plausible pixel duration.
    """
    keys = [k for k in wparams if k.startswith('RealPixelDuration_')]
    if len(keys) == 0:
        return
    if len(keys) > 1:
        raise ValueError(f"Multiple RealPixelDuration keys in header: {keys}")
    key = keys[0]
    suffix = key[len('RealPixelDuration_'):]
    if suffix.lower() in _NON_MICRO_SECOND_SUFFIXES:
        raise ValueError(f"Header key {key!r}: unit {suffix!r} is not µs, refusing to guess the conversion.")
    value = wparams.pop(key)
    vmin, vmax = _REAL_PIX_DUR_RANGE_US
    if not (vmin <= float(value) <= vmax):
        raise ValueError(f"Header key {key!r}: pixel duration {value} is outside [{vmin}, {vmax}] µs.")
    wparams['RealPixelDuration_µs'] = value


def load_wparams(raw_file, return_file=True, lower_keys=True):
    """Load wparams (scan parameters) from a raw SMP file.

    Parameters
    ----------
    raw_file : str, Path, or SMP
        Path to the SMP file, or an already-loaded SMP object.
    return_file : bool, optional
        If True, also return the loaded SMP file object. Default is True.
    lower_keys : bool, optional
        If True, all parameter keys are lowercased. Default is True.

    Returns
    -------
    wparams : dict
        Dictionary of scan parameters.
    raw_file : SMP
        Loaded SMP file object. Only returned if ``return_file=True``.
    """
    if not isinstance(raw_file, SMP):
        raw_file = load_smp_file(raw_file)

    wparams = dict()
    for k, v in raw_file._kvPairDict.items():
        wparams[k] = v[2]

    wparams['User_dxPix'] = raw_file.dxFr_pix
    wparams['User_dyPix'] = raw_file.dyFr_pix
    wparams['User_dzPix'] = raw_file.dzFr_pix or 0
    wparams['User_nPixRetrace'] = raw_file.dxRetrace_pix
    wparams['User_nXPixLineOffs'] = raw_file.dxOffs_pix
    wparams['User_ScanType'] = raw_file.scanType

    _normalize_real_pixel_duration_key(wparams)

    # Handle different naming conventions
    renaming_dict = {
        "RealPixelDuration_µs": 'RealPixDur',
        "AspectRatioFrame": "User_AspectRatioFr",
    }

    for k, v in renaming_dict.items():
        if k in wparams.keys():
            wparams[v] = wparams.pop(k)

    if lower_keys:
        wparams = {k.lower(): v for k, v in wparams.items()}

    if return_file:
        return wparams, raw_file
    else:
        return wparams


def load_all_stacks_and_wparams(raw_file, ch_name_base='wDataCh', ch_max=5, crop=True, lower_keys=True):
    """Load all available channel stacks and scan parameters from a raw SMP file.

    Attempts to load up to ``ch_max`` channels. Channels that cannot be loaded
    due to missing data are silently skipped; other errors produce a warning.

    Parameters
    ----------
    raw_file : str, Path, or SMP
        Path to the SMP file, or an already-loaded SMP object.
    ch_name_base : str, optional
        Base name for channel keys in the returned dict. Default is ``'wDataCh'``.
    ch_max : int, optional
        Maximum number of channels to attempt loading. Default is 5.
    crop : bool, optional
        If True, crop the data when loading. Default is True.
    lower_keys : bool, optional
        If True, all wparams keys are lowercased. Default is True.

    Returns
    -------
    ch_stacks : dict of str -> np.ndarray
        Mapping from channel name (e.g. ``'wDataCh0'``) to the transposed data array.
    wparams : dict
        Dictionary of scan parameters.
    """
    wparams, raw_file = load_wparams(raw_file, lower_keys=lower_keys)

    ch_stacks = dict()
    for i in range(ch_max):
        try:
            ch_stacks[f"{ch_name_base}{i}"] = raw_file.getData(ch=i, crop=crop).T
        except (IndexError, AttributeError):
            continue
        except Exception as e:
            warnings.warn(f'Failed to load channel={ch_name_base}{i} for {raw_file} with error: {e}')

    return ch_stacks, wparams


def load_stacks_and_wparams(raw_file, ch_names=('wDataCh0', 'wDataCh1'), lower_keys=True):
    """Load specified channel stacks and scan parameters from a raw SMP file.

    The first channel in ``ch_names`` is treated as the primary channel and
    must load successfully. Subsequent channels fall back to zero arrays of the
    same shape if loading fails.

    Parameters
    ----------
    raw_file : str, Path, or SMP
        Path to the SMP file, or an already-loaded SMP object.
    ch_names : tuple of str, optional
        Names of the channels to load. Channel index is inferred from the last
        character of each name (e.g. ``'wDataCh1'`` -> channel 1).
        Default is ``('wDataCh0', 'wDataCh1')``.
    lower_keys : bool, optional
        If True, all wparams keys are lowercased. Default is True.

    Returns
    -------
    ch_stacks : dict of str -> np.ndarray
        Mapping from channel name to the transposed data array. Missing
        secondary channels are replaced with zero arrays matching the primary
        channel shape.
    wparams : dict
        Dictionary of scan parameters.
    """
    wparams, raw_file = load_wparams(raw_file, lower_keys=lower_keys)

    ch_stacks = dict()
    ch_name_main = ch_names[0]
    ch_stacks[ch_name_main] = raw_file.getData(ch=int(ch_name_main[-1]), crop=True).T
    for ch_name in ch_names[1:]:
        try:
            ch_stacks[ch_name] = raw_file.getData(ch=int(ch_name[-1]), crop=True).T
        except Exception as e:
            warnings.warn(f'Failed to load channel={ch_name} for {raw_file} with error: {e}')
            ch_stacks[ch_name] = np.zeros_like(ch_stacks[ch_name_main])

    return ch_stacks, wparams
