import os
from pathlib import Path
import pandas as pd
import numpy as np


def get_file_info_df(folder, user_dict, from_raw_data):
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
            kind = 'response'
            is_alias_match = [file_info_dict["stimulus"].lower() == rm_alias for rm_alias in rm_aliases]
            mask_order = np.argmax(is_alias_match) if np.any(is_alias_match) else len(rm_aliases)

        def get_cond(i):
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

        file_info_dicts.append(file_info_dict)

    file_info_df = pd.DataFrame(file_info_dicts)
    return file_info_df


def print_tree(startpath, include_types=None, exclude_types=None, nmax=200):
    if exclude_types is not None:
        exclude_types = [t.lower().strip('.') for t in exclude_types]

    if include_types is not None:
        include_types = [t.lower().strip('.') for t in include_types]

    paths = sorted(os.walk(startpath))

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


def find_folders_with_file_of_type(data_dir: str, ending: str = '.ini', ignore_hidden=False) -> list:
    """
    Search for header files in folder in given path.
    :param data_dir: Root folder.
    :param ending: File ending to search.
    :param ignore_hidden: Ignore hidden files?
    :return: List of header files.
    """
    os_walk_output = []
    for folder, subfolders, files in os.walk(data_dir):
        if np.any([f.endswith(ending) and not (ignore_hidden and f.startswith('.')) for f in files]):
            os_walk_output.append(folder)
    return os_walk_output


def is_alias_number_match(name: str, alias: str):
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


def get_file_info_at_loc(file_info, loc):
    if loc is None or loc < 0:
        return None
    else:
        return file_info[loc] if len(file_info) > loc else None


def is_file_info_alias_match(file_info_dict, aliases, key_list=None, allow_num_suffix=False):
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


def as_pre_filepath(filepath, raw_data_dir='Raw', pre_data_dir='Pre',
                    raw_suffix='.smp', pre_suffix='.h5', pre_prefix='SMP_'):
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
