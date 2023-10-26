import os
import random

import datajoint as dj


def _connect_test_schema(use_rgc_classifier=True):
    home = os.path.expanduser("~")
    user = os.path.split(home)[1]
    config_file = f'{home}/datajoint/dj_{user}_conf.json'
    schema_name = f"ageuler_{user}_pytest_{random.randint(0, 999):03}"

    dj.config.load(config_file)
    dj.config['schema_name'] = schema_name

    if use_rgc_classifier:
        from djimaging.tables.classifier.rgc_classifier import prepare_dj_config_rgc_classifier
        output_dir = f'{home}/djimaging/tests'
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        prepare_dj_config_rgc_classifier(output_dir)

    connection = dj.conn()
    return connection


def _activate_test_schema(test_schema):
    from djimaging.utils.dj_utils import activate_schema
    activate_schema(schema=test_schema, create_schema=True, create_tables=True)
    return test_schema


def _drop_test_schema(schema):
    schema.drop(force=True)


def _reset_test_schema(test_schema):
    _activate_test_schema(test_schema)
    _drop_test_schema(test_schema)


def is_connected(connection):
    if connection is None:
        return False
    else:
        return connection.is_connected
