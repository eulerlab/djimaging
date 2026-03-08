import hashlib
import json
import os
import random
import sys
from contextlib import contextmanager
from typing import Any, Generator

import datajoint as dj
import numpy as np


def get_class_attributes(class_: type) -> list:
    """Return the names of non-dunder attributes defined directly on a class.

    Parameters
    ----------
    class_ : type
        Class object to inspect.

    Returns
    -------
    list
        List of attribute name strings that do not start with ``'__'``.
    """
    class_attrs = [attr for attr in class_.__dict__.keys() if attr[:2] != '__']
    return class_attrs


def get_input_tables(definition: str) -> list:
    """Parse a DataJoint table definition string and return upstream table names.

    Parameters
    ----------
    definition : str
        DataJoint table definition string (the ``definition`` class attribute).

    Returns
    -------
    list
        List of table name strings referenced via ``->`` dependency arrows.
    """
    all_lines = definition.replace(' ', '').split('\n')
    table_lines = [line for line in all_lines if line.startswith('->')]
    tables = [line.replace('->', '').replace('self.', '').replace('self().', '') for line in table_lines]
    return tables


def activate_schema(schema: dj.Schema, schema_name: str | None = None,
                    create_schema: bool = True, create_tables: bool = True) -> None:
    """Activate a DataJoint schema, resolving the name from config or parameter.

    Based on: https://github.com/datajoint/element-lab

    Parameters
    ----------
    schema : dj.Schema
        The DataJoint schema object to activate.
    schema_name : str | None, optional
        Schema name to use. If None, the name is read from ``dj.config['schema_name']``.
    create_schema : bool, optional
        Whether to create the schema in the database if it does not exist. Default is True.
    create_tables : bool, optional
        Whether to create tables in the database if they do not exist. Default is True.

    Raises
    ------
    ValueError
        If no schema name can be resolved, or if the provided name conflicts with the
        one stored in ``dj.config``.
    """
    config_schema_name = dj.config.get('schema_name', None)

    if not schema_name and not config_schema_name:
        raise ValueError('Must provide schema_name in config or as parameter')
    elif schema_name and config_schema_name and schema_name != config_schema_name:
        raise ValueError('Schema name in config must match schema_name parameter')

    schema.activate(schema_name or config_schema_name, create_schema=create_schema, create_tables=create_tables)


def make_hash(obj: Any) -> str:
    """
    Creates a 32-character hash that uniquely identifies the content of a Python object.

    This function handles arbitrarily nested objects (dictionaries, lists, etc.) by
    recursively processing the structure and generating a consistent hash based on content.

    Parameters
    ----------
    obj : Any
        The Python object to hash. Can be a primitive type or a complex nested structure
        like dictionaries, lists, tuples, sets, etc.

    Returns
    -------
    str
        A 32-character hexadecimal hash string that uniquely identifies the object content.

    Notes
    -----
    - The function handles common Python data types including:
      - Primitives (int, float, str, bool, None)
      - Collections (dict, list, tuple, set)
      - Nested combinations of the above
    - Objects that aren't directly serializable will be converted to their string representation.
    - Dictionary keys are sorted to ensure consistent hashing regardless of key order.
    """

    def _prepare_for_hashing(value: Any) -> Any:
        """Recursively prepare an object for hashing by converting to serializable form.

        Parameters
        ----------
        value : Any
            Object to prepare.

        Returns
        -------
        Any
            JSON-serializable representation of `value`.
        """
        if isinstance(value, dict):
            # Sort dictionary items by key for consistent ordering
            return {
                k: _prepare_for_hashing(v)
                for k, v in sorted(value.items())
            }
        elif isinstance(value, (list, tuple)):
            return [_prepare_for_hashing(item) for item in value]
        elif isinstance(value, set):
            # Convert set to sorted list for consistent ordering
            return [_prepare_for_hashing(item) for item in sorted(value)]
        elif isinstance(value, (int, float, str, bool)) or value is None:
            # Primitive types can be used directly
            return value
        else:
            # For other types, use their string representation
            return str(value)

    # Prepare the object by handling nested structures and ensuring consistent representation
    prepared_obj = _prepare_for_hashing(obj)

    # Convert to a JSON string with sorted keys for consistent serialization
    json_str = json.dumps(prepared_obj, sort_keys=True)

    # Generate MD5 hash (32 characters) of the JSON string
    hash_obj = hashlib.md5(json_str.encode('utf-8'))

    return hash_obj.hexdigest()


def get_primary_key(table: dj.Table, key: dict | None = None) -> dict:
    """Return a primary-key dict for a DataJoint table.

    Parameters
    ----------
    table : dj.Table
        DataJoint table to query.
    key : dict | None, optional
        If provided, it is filtered to contain only the primary-key attributes.
        If None, a random row is chosen from the table.

    Returns
    -------
    dict
        Dictionary containing only the primary-key attributes and their values.
    """
    if key is not None:
        key = {k: v for k, v in key.items() if k in table.primary_key}
    else:
        key = random.choice(table.proj().fetch(as_dict=True))
    return key


def get_secondary_keys(table: dj.Table) -> list:
    """Get all secondary (non-primary) key names of a DataJoint table.

    Parameters
    ----------
    table : dj.Table
        DataJoint table to inspect.

    Returns
    -------
    list
        List of attribute name strings that are not part of the primary key.
    """
    key = get_primary_key(table)
    row = (table & key).fetch1()
    secondary_keys = list(set(row.keys()) - set(key.keys()))
    return secondary_keys


def merge_keys(key1: dict, key2: dict) -> dict:
    """Merge two DataJoint key dictionaries, raising on conflicts.

    Parameters
    ----------
    key1 : dict
        First key dictionary (used as the base).
    key2 : dict
        Second key dictionary to merge into `key1`.

    Returns
    -------
    dict
        Merged dictionary containing all entries from both keys.

    Raises
    ------
    ValueError
        If a key appears in both dictionaries with inconsistent values.
    """
    merged_key = key1.copy()
    for k, v2 in key2.items():
        if k in key1.keys():
            v1 = key1[k]
            if not is_equal(v1, v2):
                raise ValueError(f'Keys are inconsistent for key {k} with values {v2} and {v1}.')
        else:
            merged_key[k] = v2
    return merged_key


def is_equal(v1: Any, v2: Any) -> bool:
    """Check equality of two values, falling back to numpy comparison.

    Parameters
    ----------
    v1 : Any
        First value.
    v2 : Any
        Second value.

    Returns
    -------
    bool
        True if `v1` and `v2` are considered equal, False otherwise.
    """
    try:
        if v1 == v2:
            return True
        elif np.asarray(v1) == np.asarray(v2):
            return True
    except:
        pass
    return False


def check_unique_one(values: Any, name: str = 'values') -> Any:
    """Assert that all elements in `values` are identical and return that value.

    Parameters
    ----------
    values : array-like
        Sequence of values expected to be all equal.
    name : str, optional
        Label used in the error message. Default is ``'values'``.

    Returns
    -------
    Any
        The single unique value shared by all elements.

    Raises
    ------
    ValueError
        If the elements are not all equal.
    """
    values = np.asarray(values)
    if 'float' in str(type(values[0])):
        all_equal = np.allclose(values, values[0])
    else:
        all_equal = np.all(values == values[0])

    if not all_equal:
        raise ValueError(f'{name} are not unique: {values}')

    return values[0]


@contextmanager
def suppress_output(condition: bool = True) -> Generator[None, None, None]:
    """Context manager that suppresses stdout and stderr when `condition` is True.

    Parameters
    ----------
    condition : bool, optional
        If True (default), stdout and stderr are redirected to ``/dev/null``
        for the duration of the ``with`` block. If False, output is unchanged.

    Yields
    ------
    None
    """
    if condition:
        with open(os.devnull, 'w') as devnull:
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = devnull
            sys.stderr = devnull
            try:
                yield
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
    else:
        yield
