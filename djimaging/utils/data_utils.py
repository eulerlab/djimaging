import h5py
import os
from configparser import ConfigParser


def load_h5_data(filename):
    """Helper function to load h5 file."""
    with h5py.File(filename, 'r') as f:
        return {key: f[key][:] for key in list(f.keys())}


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


def list_data_files(folder, hidden=False, field=None, field_loc=None, ftype='h5'):
    data_files = []
    for file in os.listdir(folder):
        if not file.endswith(f'.{ftype}'):
            continue
        if file.startswith('.') and not hidden:
            continue
        if field is not None:
            assert field_loc is not None
            fileinfo = file.replace(f'.{ftype}', '').split("_")

            if len(fileinfo) <= field_loc or field != fileinfo[field_loc]:
                continue

        data_files.append(file)
    return data_files


def read_config_dict(filename):
    config_dict = dict()
    parser = ConfigParser()
    parser.read(filename)
    for key1 in parser.keys():
        for key2 in parser[key1].keys():
            config_dict[key2[key2.find("_") + 1:]] = str(parser[key1][key2])
    return config_dict


def check_shared_alias_str(str1, str2, case_sensitive=False):

    if not case_sensitive:
        str1 = str1.lower()
        str2 = str2.lower()

    list1 = str1.split('_')
    list2 = str2.split('_')

    return check_shared_alias_list(list1, list2)


def check_shared_alias_list(list1, list2):
    set1 = set(list1)
    set2 = set(list2)

    if '' in set1:
        set1.remove('')
    if '' in set2:
        set2.remove('')

    return bool(set1 & set2)
