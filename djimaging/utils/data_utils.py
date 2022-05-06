import h5py
import os


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

        assert len(keys) == len(values), 'Lenghts do not match'

        for key, value in zip(keys, values):
            if len(key) > 0:
                if type(key) == bytes:
                    key = key.decode('utf-8')
                assert type(key) == str, type(key)

                if lower_keys:
                    key = key.lower()
                data_dict[key] = value
    return data_dict


def list_h5_files(folder, hidden=False, field=None, field_loc=None):
    h5_files = []
    for file in os.listdir(folder):
        if not file.endswith('.h5'):
            continue
        if file.startswith('.') and not hidden:
            continue
        if field is not None:
            assert field_loc is not None
            if field != file.replace('.h5', '').split("_")[field_loc]:
                continue
        h5_files.append(file)
    return h5_files

