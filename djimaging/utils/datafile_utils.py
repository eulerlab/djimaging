import os

import numpy as np


def get_filename_info(filename, datatype_loc, animal_loc, region_loc, field_loc, stimulus_loc, condition_loc):
    """Extract information from filename"""
    file_info = filename.replace('.h5', '').split('_')
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


def find_folders_with_file_of_type(data_dir: str, ending: str = '.ini') -> list:
    """
    Search for header files in folder in given path.
    :param data_dir: Root folder.
    :param ending. File ending to search.
    :return: List of header files.
    """
    os_walk_output = []
    for folder, subfolders, files in os.walk(data_dir):
        if np.any([f.endswith(ending) for f in files]):
            os_walk_output.append(folder)
    return os_walk_output


def get_condition(data_file, loc, fill_value='control'):
    data_file = os.path.split(data_file)[1]
    split_string = data_file[:data_file.find(".h5")].split("_")
    condition = split_string[loc] if loc < len(split_string) else fill_value
    return condition


def get_stim(data_file, loc, fill_value='nostim'):
    data_file = os.path.split(data_file)[1]
    split_string = data_file[:data_file.find(".h5")].split("_")
    condition = split_string[loc] if loc < len(split_string) else fill_value
    return condition
