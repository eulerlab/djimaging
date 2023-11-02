import time

import datajoint as dj
import pytest

from tests.database.database_test_utils import _connect_test_schema, _drop_test_schema, \
    _activate_test_schema, is_connected


@pytest.fixture(autouse=True, scope='module')
def database_fixture():
    # Run before
    from djimaging.schemas.core_autorois_schema import schema as test_schema

    try:
        time.sleep(0.5)
        connection = _connect_test_schema(use_rgc_classifier=False)
        time.sleep(0.5)
        _activate_test_schema(test_schema)
        time.sleep(0.5)
    except (FileNotFoundError, dj.DataJointError):
        connection = None

    # Run tests
    yield connection

    # Run after
    if is_connected(connection):
        _drop_test_schema(test_schema)


@pytest.mark.dependency()
def test_create_schema(database_fixture):
    connection = database_fixture
    if not is_connected(connection):
        pytest.skip("Not connected to database")

    from djimaging.schemas.core_autorois_schema import schema as test_schema
    assert test_schema.database in dj.list_schemas()


@pytest.mark.dependency(depends=['test_create_schema'])
def test_populate_user(database_fixture):
    connection = database_fixture
    if not is_connected(connection):
        pytest.skip("Not connected to database")
    from djimaging.schemas.core_autorois_schema import UserInfo
    userinfo = {
        'experimenter': 'DataJointTestData',
        'data_dir': '/gpfs01/euler/data/Data/DataJointTestData/xy-RGCs-minimal/',
        'datatype_loc': 0,
        'animal_loc': 1,
        'region_loc': 2,
        'field_loc': 3,
        'stimulus_loc': 4,
        'condition_loc': 5,
    }

    UserInfo().upload_user(userinfo)
    assert len(UserInfo()) == 1


@pytest.mark.dependency()
def test_populate_raw_data_params(database_fixture):
    connection = database_fixture
    if not is_connected(connection):
        pytest.skip("Not connected to database")
    from djimaging.schemas.core_autorois_schema import RawDataParams
    RawDataParams().add_default(
        from_raw_data=True,
    )
    assert len(RawDataParams()) == 1


@pytest.mark.dependency(depends=['test_populate_user'])
def test_populate_experiment(database_fixture):
    connection = database_fixture
    if not is_connected(connection):
        pytest.skip("Not connected to database")
    from djimaging.schemas.core_autorois_schema import Experiment
    Experiment().rescan_filesystem(verboselvl=0)
    assert len(Experiment()) > 0


@pytest.mark.dependency(depends=['test_populate_experiment'])
def test_populate_field(database_fixture):
    connection = database_fixture
    if not is_connected(connection):
        pytest.skip("Not connected to database")
    from djimaging.schemas.core_autorois_schema import Field
    Field().rescan_filesystem(verboselvl=0)
    assert len(Field()) > 0


@pytest.mark.dependency()
def test_populate_stimulus(database_fixture):
    connection = database_fixture
    if not is_connected(connection):
        pytest.skip("Not connected to database")

    from djimaging.schemas.core_autorois_schema import Stimulus
    Stimulus().add_nostim(skip_duplicates=True)
    Stimulus().add_chirp(spatialextent=1000, stim_name='gChirp', alias="chirp_gchirp_globalchirp", skip_duplicates=True)
    Stimulus().add_chirp(spatialextent=300, stim_name='lChirp', alias="lchirp_localchirp", skip_duplicates=True)
    Stimulus().add_movingbar(skip_duplicates=True)
    assert len(Stimulus()) == 4


@pytest.mark.dependency(depends=['test_populate_stimulus'])
def test_populate_presentation(database_fixture):
    connection = database_fixture
    if not is_connected(connection):
        pytest.skip("Not connected to database")
    from djimaging.schemas.core_autorois_schema import Presentation
    Presentation().populate()
    assert len(Presentation()) > 0


@pytest.mark.dependency(depends=['test_populate_presentation'])
def test_populate_roi_mask(database_fixture):
    connection = database_fixture
    if not is_connected(connection):
        pytest.skip("Not connected to database")
    from djimaging.schemas.core_autorois_schema import RoiMask

    missing_keys = RoiMask().list_missing_field()

    from djimaging.tables.core_autorois.roi_mask import load_default_autorois_models
    autorois_models = load_default_autorois_models()

    for missing_key in missing_keys:
        roi_mask_gui = RoiMask().draw_roi_mask(field_key=missing_key, autorois_models=autorois_models)
        roi_mask_gui.exec_autorois_all()
        roi_mask_gui.insert_database(RoiMask, missing_key)

    assert len(RoiMask()) > 0


@pytest.mark.dependency(depends=['test_populate_roi_mask'])
def test_populate_roi(database_fixture, processes=20):
    connection = database_fixture
    if not is_connected(connection):
        pytest.skip("Not connected to database")
    from djimaging.schemas.core_autorois_schema import Roi
    Roi().populate(processes=processes)
    assert len(Roi()) > 0


@pytest.mark.dependency(depends=['test_populate_roi'])
def test_populate_traces(database_fixture, processes=20):
    connection = database_fixture
    if not is_connected(connection):
        pytest.skip("Not connected to database")
    from djimaging.schemas.core_autorois_schema import Traces
    Traces().populate(processes=processes)
    assert len(Traces()) > 0
