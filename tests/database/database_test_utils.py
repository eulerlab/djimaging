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


def _populate_user():
    from djimaging.schemas.core_schema import UserInfo
    userinfo = {
        'experimenter': 'DataJointTestData',
        'data_dir': '/gpfs01/euler/data/Data/DataJointTestData/xy-RGCs/',
        'datatype_loc': 0,
        'animal_loc': 1,
        'region_loc': 2,
        'field_loc': 3,
        'stimulus_loc': 4,
        'condition_loc': 5,
    }
    UserInfo().upload_user(userinfo)


def _populate_raw_data_params():
    from djimaging.schemas.core_schema import RawDataParams
    RawDataParams().add_default()


def _populate_experiment():
    from djimaging.schemas.core_schema import Experiment
    Experiment().rescan_filesystem(verboselvl=0)


def _populate_field():
    from djimaging.schemas.core_schema import Field
    Field().rescan_filesystem(verboselvl=0)


def _populate_stimulus():
    from djimaging.schemas.core_schema import Stimulus
    Stimulus().add_nostim(skip_duplicates=True)
    Stimulus().add_chirp(spatialextent=1000, stim_name='gChirp', alias="chirp_gchirp_globalchirp", skip_duplicates=True)
    Stimulus().add_chirp(spatialextent=300, stim_name='lChirp', alias="lchirp_localchirp", skip_duplicates=True)
    Stimulus().add_movingbar(skip_duplicates=True)


def _populate_presentation(processes):
    from djimaging.schemas.core_schema import Presentation
    Presentation().populate(processes=processes)


def _populate_roi(processes):
    from djimaging.schemas.core_schema import Roi
    Roi().populate(processes=processes)


def _populate_traces(processes):
    from djimaging.schemas.core_schema import Traces
    Traces().populate(processes=processes)


def _populate_preprocessparams():
    from djimaging.schemas.core_schema import PreprocessParams
    PreprocessParams().add_default(skip_duplicates=True)


def _populate_preprocesstraces(processes):
    from djimaging.schemas.core_schema import PreprocessTraces
    PreprocessTraces().populate(processes=processes)


def _populate_snippets(processes):
    from djimaging.schemas.core_schema import Snippets
    Snippets().populate(processes=processes)


def _populate_averages(processes):
    from djimaging.schemas.core_schema import Averages
    Averages().populate(processes=processes)


def _drop_test_schema(schema):
    schema.drop(force=True)


def _activate_test_schema(test_schema):
    from djimaging.utils.dj_utils import activate_schema
    activate_schema(schema=test_schema, create_schema=True, create_tables=True)
    return test_schema
