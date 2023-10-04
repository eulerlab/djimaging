import os

import datajoint as dj


def _connect_test_schema():
    home = os.path.expanduser("~")
    user = os.path.split(home)[1]
    config_file = f'{home}/datajoint/dj_{user}_conf.json'
    schema_name = f"ageuler_{user}_pytest"
    dj.config.load(config_file)
    dj.config['schema_name'] = schema_name
    dj.conn()
    return schema_name


def _activate_test_schema(test_schema):
    from djimaging.utils.dj_utils import activate_schema
    activate_schema(schema=test_schema, create_schema=True, create_tables=True)
    return test_schema


def _drop_test_schema(schema):
    schema.drop(force=True)


def _reset_test_schema(test_schema):
    _activate_test_schema(test_schema)
    _drop_test_schema(test_schema)


def is_not_connected():
    try:
        _connect_test_schema()
    except (FileNotFoundError, dj.DataJointError):
        return True
    return False
