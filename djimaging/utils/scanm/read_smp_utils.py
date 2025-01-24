import warnings

import numpy as np

from djimaging.utils.misc_utils import CapturePrints


def load_smp_file(raw_file_path):
    """Load smp file"""
    try:
        from scanmsupport.scanm.scanm_smp import SMP
    except ImportError:
        raise ImportError('Custom package `scanmsupport is not installed. Cannot load SMP files.')

    raw_file = SMP()

    with CapturePrints():
        raw_file.loadSMH(raw_file_path, verbose=False)
        raw_file.loadSMP(raw_file_path)

    return raw_file


def load_wparams(raw_file, return_file=True, lower_keys=True):
    """Load wparams from raw file"""

    try:
        from scanmsupport.scanm.scanm_smp import SMP
    except ImportError:
        raise ImportError('Custom package `scanmsupport` is not installed. Cannot load SMP files.')

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
    """Load defined channels stacks from raw file"""
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
