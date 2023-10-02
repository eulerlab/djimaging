import datajoint as dj
import numpy as np

from tests.database.database_test_utils import _connect_test_schema, _populate_user, _populate_raw_data_params, \
    _populate_experiment, _populate_field, _populate_stimulus, _populate_presentation, _populate_roi, _populate_traces, \
    _populate_preprocessparams, _populate_preprocesstraces, _populate_snippets, _populate_averages, _drop_test_schema, \
    _activate_test_schema, _reset_test_schema


def _populate_test_schema(processes=20, pop_traces=True):
    _populate_user()
    _populate_raw_data_params()
    _populate_experiment()
    _populate_field()
    _populate_stimulus()
    _populate_presentation(processes=processes)
    _populate_roi(processes=processes)
    if pop_traces:
        _populate_traces(processes=processes)
        _populate_preprocessparams()
        _populate_preprocesstraces(processes=processes)
        _populate_snippets(processes=processes)
        _populate_averages(processes=processes)


def test_populate_core_schema():
    from djimaging.schemas.core_schema import schema as test_schema

    try:
        test_schema_name = _connect_test_schema()
    except FileNotFoundError:
        return
    except dj.DataJointError:
        return

    try:
        _reset_test_schema(test_schema)
    except dj.DataJointError:
        return

    try:
        test_schema = _activate_test_schema(test_schema)
    except dj.DataJointError:
        return

    try:
        _populate_test_schema(pop_traces=False)
    except Exception as e:
        _drop_test_schema(test_schema)
        raise e

    assert test_schema_name in dj.list_schemas()

    from djimaging.schemas.core_schema import Experiment
    prepwmorients = Experiment().ExpInfo().fetch('prepwmorient')
    assert np.any(prepwmorients == -1)
    assert np.any(prepwmorients == 0)

    _drop_test_schema(test_schema)
    assert test_schema_name not in dj.list_schemas()
