import os
import warnings
from pathlib import Path

import numpy as np

from djimaging.utils import scanm_utils
from djimaging.utils.alias_utils import check_shared_alias_str


def get_filename_info(filename, datatype_loc, animal_loc, region_loc, field_loc, stimulus_loc, condition_loc,
                      from_raw_data=False):
    """Extract information from filename"""
    file_info = str(Path(filename).with_suffix('')).split('_')

    if from_raw_data:
        file_info = ['SMP'] + file_info

    datatype = file_info[datatype_loc] if len(file_info) > datatype_loc else None
    animal = file_info[animal_loc] if len(file_info) > animal_loc else None
    region = file_info[region_loc] if len(file_info) > region_loc else None
    field = file_info[field_loc] if len(file_info) > field_loc else None
    stimulus = file_info[stimulus_loc] if len(file_info) > stimulus_loc else None
    condition = file_info[condition_loc] if len(file_info) > condition_loc else None

    return datatype, animal, region, field, stimulus, condition


def print_tree(startpath, include_types=None, exclude_types=None, nmax=200):
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


def find_folders_with_file_of_type(data_dir: str, ending: str = '.ini', ignore_hidden=False) -> list:
    """
    Search for header files in folder in given path.
    :param data_dir: Root folder.
    :param ending: File ending to search.
    :param ignore_hidden: Ignore hidden files?
    :return: List of header files.
    """
    os_walk_output = []
    for folder, subfolders, files in os.walk(data_dir, followlinks=True):
        if np.any([f.endswith(ending) and not (ignore_hidden and f.startswith('.')) for f in files]):
            os_walk_output.append(folder)
    return os_walk_output


def get_info_from_loc(data_file, loc, fill_value, suffix='auto'):
    data_file = os.path.split(data_file)[1]

    if suffix == 'auto':
        suffix = data_file.split('.')[-1]

    if not suffix.startswith('.'):
        suffix = '.' + suffix

    split_string = data_file[:data_file.find(suffix)].split("_")
    info = split_string[loc] if loc < len(split_string) else fill_value
    return info


def get_condition(data_file, loc, fill_value='control', suffix='auto'):
    return get_info_from_loc(data_file, loc, fill_value, suffix)


def get_stim(data_file, loc, fill_value='nostim', suffix='auto'):
    return get_info_from_loc(data_file, loc, fill_value, suffix)


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


def load_field_roi_mask_from_h5(pre_data_path, files, mask_alias='', highres_alias=''):
    """Load ROI mask for field"""
    files = np.array(files)
    penalties = np.full(files.size, len(mask_alias.split('_')))

    for i, file in enumerate(files):
        if check_shared_alias_str(highres_alias, file.lower().replace('.h5', '')):
            penalties[i] = len(mask_alias.split('_')) + 1

        else:
            for penalty, alias in enumerate(mask_alias.split('_')):
                if alias.lower() in file.lower().replace('.h5', '').split('_'):
                    penalties[i] = penalty

    sorted_files = files[np.argsort(penalties)]

    for file in sorted_files:
        roi_mask = scanm_utils.load_roi_mask_from_h5(filepath=os.path.join(pre_data_path, file), ignore_not_found=True)
        if roi_mask is not None:
            return roi_mask, file
    else:
        raise ValueError(f'No ROI mask found in any file in {pre_data_path}: {files}')


def scan_region_field_file_dicts(folder: str, user_dict: dict, verbose: bool = False, suffix='.h5') -> dict:
    """Return a dictionary that maps (region, field) to their respective files"""

    loc_mapper = {k: v for k, v in user_dict.items() if k.endswith('loc')}

    file_dicts = dict()

    files = sorted(os.listdir(folder))

    for file in files:
        if file.startswith('.') or not file.endswith(suffix):
            continue

        datatype, animal, region, field, stimulus, condition = get_filename_info(
            file if suffix == '.h5' else 'SMP_' + file, **loc_mapper)  # locs refer to h5 files, which start with SMP

        if field is None:
            if verbose:
                print(f"\tSkipping file with unrecognized field={field}: {file}")
            continue

        # Add file to list
        if (region, field) not in file_dicts:
            file_dicts[(region, field)] = dict(files=[])
        file_dicts[(region, field)]['files'].append(file)

    return file_dicts


def scan_field_file_dicts(folder: str, from_raw_data: bool, user_dict: dict, incl_condition=False,
                          verbose: bool = False) -> dict:
    """Return a dictionary that maps fields to their respective files"""

    loc_mapper = {k: v for k, v in user_dict.items() if k.endswith('loc')}

    file_dicts = dict()

    for file in sorted(os.listdir(folder)):
        if file.startswith('.') or not file.endswith('.smp' if from_raw_data else '.h5'):
            continue

        datatype, animal, region, field, stimulus, condition = get_filename_info(
            filename=file, from_raw_data=from_raw_data, **loc_mapper)

        if condition is None and incl_condition:
            condition = 'control'
            warnings.warn(f"File {file} does not have a condition incl_condition=True. Set to `{condition}`.")

        if field is None:
            if verbose:
                print(f"\tSkipping file with unrecognized field={field}: {file}")
            continue

        key = field if not incl_condition else (field, condition)

        # Create new or check for inconsistencies
        if key not in file_dicts:
            file_dicts[key] = dict(files=[], region=region)
        else:
            assert file_dicts[key]['region'] == region, \
                f"Found multiple regions for single field: {file_dicts[key]['region']} vs. {region}"

        # Add file
        file_dicts[key]['files'].append(file)
    return file_dicts


def clean_field_file_dicts(field_dicts, user_dict):
    """Remove optic-disk and outline recordings from dict with fields."""
    remove_field_keys = []
    remove_aliases = user_dict['opticdisk_alias'].split('_') + user_dict['outline_alias'].split('_')
    for field_key in field_dicts.keys():
        field = field_key[0] if isinstance(field_key, tuple) else field_key
        removed = False
        i = 0
        while not removed and i < len(remove_aliases):
            if is_alias_number_match(name=field.lower(), alias=remove_aliases[i]):
                remove_field_keys.append(field_key)
                removed = True
            i += 1

    for remove_field in remove_field_keys:
        field_dicts.pop(remove_field)

    return field_dicts


def clean_region_field_file_dicts(field_dicts, user_dict):
    """Remove optic-disk and outline recordings"""
    remove_fields = []
    remove_aliases = user_dict['opticdisk_alias'].split('_') + user_dict['outline_alias'].split('_')
    for region, field in field_dicts.keys():
        removed = False
        i = 0
        while not removed and i < len(remove_aliases):
            if is_alias_number_match(name=field.lower(), alias=remove_aliases[i]):
                remove_fields.append((region, field))
                removed = True
            i += 1

    for remove_field in remove_fields:
        field_dicts.pop(remove_field)

    fields, field_counts = np.unique([field for region, field in field_dicts.keys()], return_counts=True)

    for m_field in fields[field_counts > 1]:
        regions = [region for region, field in field_dicts.keys() if m_field == field]

        raise ValueError(f"Field `{m_field}` occurs multiple times\n" +
                         f"You have the same field name for different regions={regions}\n"
                         "This is currently not supported." +
                         "Please rename the fields to unique names per experiment.")

    return field_dicts


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
