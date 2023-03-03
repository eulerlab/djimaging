import hashlib
import random
from collections import OrderedDict
from collections.abc import Iterable, Mapping

import datajoint as dj


def get_class_attributes(class_):
    class_attrs = [attr for attr in class_.__dict__.keys() if attr[:2] != '__']
    return class_attrs


def get_input_tables(definition):
    all_lines = definition.replace(' ', '').split('\n')
    table_lines = [line for line in all_lines if line.startswith('->')]
    tables = [line.replace('->', '').replace('self.', '').replace('self().', '') for line in table_lines]
    return tables


def activate_schema(schema, schema_name=None, create_schema=True, create_tables=True):
    # Based on https://github.com/datajoint/element-lab
    """
    activate(schema, schema_name=None, create_schema=True, create_tables=True)
        :param schema: schema objec
        :param schema_name: schema name on the database server to activate the
                            `lab` element
        :param create_schema: when True (default), create schema in the
                              database if it does not yet exist
        :param create_tables: when True (default), create tables in the
                              database if they do not yet exist
    """
    if schema_name is None:
        schema_name = dj.config.get('schema_name', '')
        assert len(schema_name) > 0, 'Set schema name as parameter or in config file'
    else:
        config_schema_name = dj.config.get('schema_name', '')
        assert len(config_schema_name) == 0 or schema_name == config_schema_name, \
            'Trying to set two different schema names'

    schema.activate(schema_name, create_schema=create_schema, create_tables=create_tables)


def make_hash(obj: object) -> str:
    """
    Given a Python object, returns a 32 character hash string to uniquely identify
    the content of the object. The object can be arbitrary nested (i.e. dictionary
    of dictionary of list etc), and hashing is applied recursively to uniquely
    identify the content.

    For dictionaries (at any level), the key order is ignored when hashing
    so that {"a":5, "b": 3, "c": 4} and {"b": 3, "a": 5, "c": 4} will both
    give rise to the same hash. Exception to this rule is when an OrderedDict
    is passed, in which case difference in key order is respected. To keep
    compatible with previous versions of Python and the assumed general
    intentions, key order will be ignored even in Python 3.7+ where the
    default dictionary is officially an ordered dictionary.

    :param obj: A (potentially nested) Python object
    :return: hash: str - a 32 charcter long hash string to uniquely identify the object.
    """
    hashed = hashlib.md5()

    if isinstance(obj, str):
        hashed.update(obj.encode())
    elif isinstance(obj, OrderedDict):
        for k, v in obj.items():
            hashed.update(str(k).encode())
            hashed.update(make_hash(v).encode())
    elif isinstance(obj, Mapping):
        for k in sorted(obj, key=str):
            hashed.update(str(k).encode())
            hashed.update(make_hash(obj[k]).encode())
    elif isinstance(obj, Iterable):
        for v in obj:
            hashed.update(make_hash(v).encode())
    else:
        hashed.update(str(obj).encode())

    return hashed.hexdigest()


def get_primary_key(table, key=None):
    if key is not None:
        key = {k: v for k, v in key.items() if k in table.primary_key}
    else:
        key = random.choice(table.proj().fetch(as_dict=True))
    return key


def get_secondary_keys(table):
    """Get all secondary keys of table"""
    key = get_primary_key(table)
    row = (table & key).fetch1()
    secondary_keys = list(set(row.keys()) - set(key.keys()))
    return secondary_keys
