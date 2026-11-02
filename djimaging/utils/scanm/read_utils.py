from typing import Tuple

from djimaging.utils.scanm import read_smp_utils, read_h5_utils


def load_stacks(filepath: str, from_raw_data: bool, ch_names: tuple = ('wDataCh0', 'wDataCh1')) -> Tuple[dict, dict]:
    """Load channel stacks and wparams from a ScanM recording file.

    Parameters
    ----------
    filepath : str
        Path to the recording file (.h5 or .smp).
    from_raw_data : bool
        If True, load from raw .smp file; otherwise load from .h5 file.
    ch_names : tuple, optional
        Names of channels to load. Default is ('wDataCh0', 'wDataCh1').

    Returns
    -------
    ch_stacks : dict
        Dictionary mapping channel name to 3-D numpy array (x, y, frames).
    wparams : dict
        Dictionary of scan parameters.
    """
    if from_raw_data:
        ch_stacks, wparams = read_smp_utils.load_stacks_and_wparams(filepath, ch_names=ch_names)
    else:
        ch_stacks, wparams = read_h5_utils.load_stacks_and_wparams(filepath, ch_names=ch_names)
    return ch_stacks, wparams
