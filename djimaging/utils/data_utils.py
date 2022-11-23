from configparser import ConfigParser

import h5py


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
