import os
import warnings
from pathlib import Path
import pandas as pd
import numpy as np


def get_file_info_row(folder: str, filename: str, loc_mapper: dict, od_aliases: list, hr_aliases: list,
                      ol_aliases: list, rm_aliases: list) -> dict:
    """Parse a single filename and return a dictionary of extracted metadata.

    The function determines the file's ``kind`` (``'od'``, ``'hr'``,
    ``'outline'``, or ``'response'``) and computes a ``mask_order`` score
    that controls which file is preferred when loading ROI masks.

    Parameters
    ----------
    folder : str
        Directory containing the file.
    filename : str
        Filename (not the full path) to parse.
    loc_mapper : dict
        Mapping of metadata field names to their positional index within the
        underscore-separated filename tokens.
    od_aliases : list
        List of alias strings identifying optic-disk files.
    hr_aliases : list
        List of alias strings identifying high-resolution files.
    ol_aliases : list
        List of alias strings identifying outline files.
    rm_aliases : list
        List of alias strings identifying ROI-mask reference files, ordered
        by priority.

    Returns
    -------
    dict
        Dictionary with keys derived from `loc_mapper` plus ``'kind'``,
        ``'filepath'``, and ``'mask_order'``.

    Raises
    ------
    ValueError
        If `filename` does not contain a valid stimulus field and cannot be
        assigned to any known kind.
    """
    file_info = str(Path(filename).with_suffix('')).split('_')
    file_info_dict = {name: get_file_info_at_loc(file_info, loc) for name, loc in loc_mapper.items()}
    # Field or Stim are valid locations to place special field info like optic-disk
    if is_file_info_alias_match(file_info_dict, aliases=od_aliases, key_list=["region", "field", "stimulus"]):
        kind = 'od'
        mask_order = len(rm_aliases) + 1
    elif is_file_info_alias_match(file_info_dict, aliases=hr_aliases, key_list=["region", "field", "stimulus"]):
        kind = 'hr'
        mask_order = len(rm_aliases) + 1
    elif is_file_info_alias_match(
            file_info_dict, aliases=ol_aliases, key_list=["region", "field", "stimulus"], allow_num_suffix=True):
        kind = 'outline'
        mask_order = len(rm_aliases) + 1
    else:
        if file_info_dict.get('stimulus', None) is not None:
            kind = 'response'
            is_alias_match = [file_info_dict["stimulus"].lower() == rm_alias for rm_alias in rm_aliases]
            mask_order = np.argmax(is_alias_match) if np.any(is_alias_match) else len(rm_aliases)
        else:
            raise ValueError(f"File {filename} does not contain a valid stimulus field: {file_info_dict}")

    def get_cond(i: int) -> str:
        cond = file_info_dict.get(f'cond{i + 1}', 'none')
        if cond is None:
            cond = 'none'
        return cond.lower()

    # Add penalty for non-control experiments
    mask_order = float(mask_order) + sum([0. if get_cond(i) in ['control', 'none', 'c1', 'contr', 'cntr'] else 0.1
                                          for i in range(3)])

    file_info_dict["kind"] = kind
    file_info_dict["filepath"] = os.path.join(folder, filename)
    file_info_dict["mask_order"] = mask_order

    return file_info_dict


def get_file_info_df(folder: str, user_dict: dict, from_raw_data: bool) -> pd.DataFrame:
    """Build a DataFrame of file metadata for all data files in a folder.

    Parameters
    ----------
    folder : str
        Directory containing the data files.
    user_dict : dict
        User configuration dictionary containing alias and location settings.
        Expected keys include ``'opticdisk_alias'``, ``'highres_alias'``,
        ``'outline_alias'``, ``'mask_alias'``, and any ``'*_loc'`` keys.
    from_raw_data : bool
        If True, look for ``.smp`` raw files; otherwise look for ``.h5`` files.

    Returns
    -------
    pd.DataFrame
        DataFrame where each row corresponds to one successfully parsed file,
        with columns for each metadata field plus ``'kind'``, ``'filepath'``,
        and ``'mask_order'``.
    """
    # Prepare intput
    filenames = [f for f in os.listdir(folder)
                 if f.lower().endswith('.smp' if from_raw_data else '.h5') and not f.startswith('.')]

    loc_mapper = {k[:-4]: v - int(from_raw_data) for k, v in user_dict.items() if k.endswith('_loc') and v is not None}
    od_aliases = user_dict['opticdisk_alias'].lower().split('_')
    hr_aliases = user_dict['highres_alias'].lower().split('_')
    ol_aliases = user_dict['outline_alias'].lower().split('_')
    rm_aliases = user_dict['mask_alias'].lower().split('_')

    # Extract file information
    file_info_dicts = []
    for filename in filenames:
        try:
            file_info_dict = get_file_info_row(
                folder, filename, loc_mapper, od_aliases, hr_aliases, ol_aliases, rm_aliases)
        except Exception as e:
            warnings.warn(f"Error in file {filename}: {e}")
            continue

        file_info_dicts.append(file_info_dict)

    file_info_df = pd.DataFrame(file_info_dicts)
    return file_info_df


def print_tree(startpath: str, include_types: list | None = None,
               exclude_types: list | None = None, nmax: int = 200) -> None:
    """Print a directory tree, optionally filtering by file type.

    Parameters
    ----------
    startpath : str
        Root directory from which to start the tree.
    include_types : list | None, optional
        If provided, only files with these extensions are shown.
    exclude_types : list | None, optional
        If provided, files with these extensions are hidden.
    nmax : int, optional
        Maximum number of directories to display. Default is 200.

    Returns
    -------
    None
    """
    if exclude_types is not None:
        exclude_types = [t.lower().strip('.') for t in exclude_types]

    if include_types is not None:
        include_types = [t.lower().strip('.') for t in include_types]

    paths = sorted(os.walk(startpath, followlinks=True))

    for root, dirs, files in paths[:nmax]:
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f'{indent}{os.path.basename(root)}/:  [{len(files)} files]')
        subindent = ' ' * 4 * (level + 1)

        for f in sorted(files):
            f_type = f.split('.')[1].lower()

            if exclude_types is not None and f_type in exclude_types:
                continue

            if include_types is None or f_type in include_types:
                print(f'{subindent}{f}')

    if len(paths) > nmax:
        print('...')


def find_folders_with_file_of_type(data_dir: str, ending: str = '.ini', ignore_hidden: bool = False) -> list:
    """Search recursively for folders that contain at least one file with a given extension.

    Parameters
    ----------
    data_dir : str
        Root folder to search.
    ending : str, optional
        File extension (including leading dot) to look for. Default is ``'.ini'``.
    ignore_hidden : bool, optional
        If True, files whose names start with ``'.'`` are ignored.
        Default is False.

    Returns
    -------
    list
        List of folder paths (strings) that contain at least one matching file.
    """
    os_walk_output = []
    for folder, subfolders, files in os.walk(data_dir, followlinks=True):
        if np.any([f.endswith(ending) and not (ignore_hidden and f.startswith('.')) for f in files]):
            os_walk_output.append(folder)
    return os_walk_output


def is_alias_number_match(name: str, alias: str) -> bool:
    """Check whether `name` equals `alias` or equals `alias` followed by an integer suffix.

    Parameters
    ----------
    name : str
        Name string to test (e.g. ``'outline3'``).
    alias : str
        Base alias string to match against (e.g. ``'outline'``).

    Returns
    -------
    bool
        True if `name` exactly equals `alias` or equals `alias` concatenated
        with an integer (e.g. ``'outline3'`` matches alias ``'outline'``).
    """
    if name == alias:
        return True
    elif name.startswith(alias):
        try:
            # is a number? e.g. outline3
            int(name[len(alias):])
            is_number = True
        except ValueError:
            is_number = False
        if is_number:
            return True
    return False


def get_file_info_at_loc(file_info: list, loc: int | None) -> str | None:
    """Return the token at position `loc` in a list of filename parts.

    Parameters
    ----------
    file_info : list
        List of filename tokens (typically from splitting on ``'_'``).
    loc : int | None
        Index of the desired token. Returns ``None`` if ``loc`` is ``None``
        or negative.

    Returns
    -------
    str | None
        The token at `loc`, or ``None`` if `loc` is out of range or invalid.
    """
    if loc is None or loc < 0:
        return None
    else:
        return file_info[loc] if len(file_info) > loc else None


def is_file_info_alias_match(file_info_dict: dict, aliases: list,
                              key_list: list | None = None, allow_num_suffix: bool = False) -> bool:
    """Check whether any value in `file_info_dict` matches one of the given aliases.

    Parameters
    ----------
    file_info_dict : dict
        Dictionary of filename metadata field names to their values.
    aliases : list
        List of alias strings to match against.
    key_list : list | None, optional
        Subset of keys in `file_info_dict` to inspect. If None, all keys are used.
    allow_num_suffix : bool, optional
        If True, values that equal an alias followed by an integer are also
        accepted (e.g. ``'outline3'`` matches alias ``'outline'``).
        Default is False.

    Returns
    -------
    bool
        True if at least one inspected value matches an alias.
    """
    key_list = file_info_dict.keys() if key_list is None else key_list
    for key in key_list:
        value = file_info_dict.get(key, None)

        if value is None:
            continue

        if not allow_num_suffix:
            if value.lower() in aliases:
                return True
        else:
            for alias in aliases:
                if is_alias_number_match(value.lower(), alias):
                    return True

    return False


def as_pre_filepath(filepath: str, raw_data_dir: str = 'Raw', pre_data_dir: str = 'Pre',
                    raw_suffix: str = '.smp', pre_suffix: str = '.h5',
                    pre_prefix: str = 'SMP_') -> str:
    """Convert a raw or pre-processed file path to its canonical pre-processed path.

    Parameters
    ----------
    filepath : str
        Path to a raw (``.smp``) or pre-processed (``.h5``) file.
    raw_data_dir : str, optional
        Name of the raw-data subdirectory. Default is ``'Raw'``.
    pre_data_dir : str, optional
        Name of the pre-processed-data subdirectory. Default is ``'Pre'``.
    raw_suffix : str, optional
        Extension of raw files. Default is ``'.smp'``.
    pre_suffix : str, optional
        Extension of pre-processed files. Default is ``'.h5'``.
    pre_prefix : str, optional
        Prefix prepended to the filename when constructing the pre-processed
        path from a raw path. Default is ``'SMP_'``.

    Returns
    -------
    str
        Canonical path to the corresponding pre-processed file.

    Raises
    ------
    ValueError
        If `filepath` does not end with `raw_suffix` or `pre_suffix`, or if
        a pre-processed file is not located in `pre_data_dir`, or if a raw
        file is not located in `raw_data_dir`.
    """
    fpath, fname = os.path.split(filepath)
    froot, fdir = os.path.split(fpath)

    if fname.endswith(pre_suffix):
        if fdir != pre_data_dir:
            raise ValueError(f"Pre file {filepath} is not in pre dir {pre_data_dir} but in {fdir}")
        return filepath
    elif fname.endswith(raw_suffix):
        if fdir != raw_data_dir:
            raise ValueError(f"Raw file {filepath} is not in raw dir {raw_data_dir} but in {fdir}")
        return os.path.join(froot, pre_data_dir, f"{pre_prefix}{fname.replace(raw_suffix, pre_suffix)}")
    else:
        raise ValueError(f"File {filepath} does not end with {raw_suffix} or {pre_suffix}")
