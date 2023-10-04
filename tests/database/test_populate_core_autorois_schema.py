import datajoint as dj
import pytest

from tests.database.database_test_utils import _connect_test_schema, _drop_test_schema, \
    _activate_test_schema, _reset_test_schema, is_not_connected


@pytest.fixture(autouse=True, scope='module')
def create_test_drop():
    # Run before
    from djimaging.schemas.core_autorois_schema import schema as test_schema

    try:
        _connect_test_schema()
        _reset_test_schema(test_schema)
        _activate_test_schema(test_schema)
        pass
    except (FileNotFoundError, dj.DataJointError):
        pass

    # Run tests
    yield

    # Run after
    _drop_test_schema(test_schema)


@pytest.mark.dependency()
@pytest.mark.skipif(is_not_connected(), reason="Not connected to database")
def test_create_schema():
    from djimaging.schemas.core_autorois_schema import schema as test_schema
    assert test_schema.database in dj.list_schemas()


@pytest.mark.dependency(depends=['test_create_schema'])
@pytest.mark.skipif(is_not_connected(), reason="Not connected to database")
def test_populate_user():
    from djimaging.schemas.core_autorois_schema import UserInfo
    userinfo = {
        'experimenter': 'DataJointTestData',
        'data_dir': '/gpfs01/euler/data/Data/DataJointTestData/autorois/',
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
@pytest.mark.skipif(is_not_connected(), reason="Not connected to database")
def test_populate_raw_data_params():
    from djimaging.schemas.core_autorois_schema import RawDataParams
    RawDataParams().add_default()
    assert len(RawDataParams()) == 1


@pytest.mark.dependency(depends=['test_populate_user'])
@pytest.mark.skipif(is_not_connected(), reason="Not connected to database")
def test_populate_experiment():
    from djimaging.schemas.core_autorois_schema import Experiment
    Experiment().rescan_filesystem(verboselvl=0)
    assert len(Experiment()) > 0


@pytest.mark.dependency(depends=['test_populate_experiment'])
@pytest.mark.skipif(is_not_connected(), reason="Not connected to database")
def test_populate_field():
    from djimaging.schemas.core_autorois_schema import Field
    Field().rescan_filesystem(verboselvl=0)
    assert len(Field()) > 0


@pytest.mark.dependency()
@pytest.mark.skipif(is_not_connected(), reason="Not connected to database")
def test_populate_stimulus():
    from djimaging.schemas.core_autorois_schema import Stimulus
    Stimulus().add_nostim(skip_duplicates=True)
    Stimulus().add_chirp(spatialextent=1000, stim_name='gChirp', alias="chirp_gchirp_globalchirp", skip_duplicates=True)
    Stimulus().add_chirp(spatialextent=300, stim_name='lChirp', alias="lchirp_localchirp", skip_duplicates=True)
    Stimulus().add_movingbar(skip_duplicates=True)
    assert len(Stimulus()) == 4


@pytest.mark.dependency(depends=['test_populate_stimulus'])
@pytest.mark.skipif(is_not_connected(), reason="Not connected to database")
def test_populate_presentation():
    from djimaging.schemas.core_autorois_schema import Presentation
    Presentation().populate()
    assert len(Presentation()) > 0


@pytest.mark.dependency(depends=['test_populate_presentation'])
@pytest.mark.skipif(is_not_connected(), reason="Not connected to database")
def test_populate_roi_mask():
    from djimaging.schemas.core_autorois_schema import RoiMask

    missing_keys = RoiMask().list_missing_field()

    for missing_key in missing_keys:
        roi_mask_gui = RoiMask().draw_roi_mask(field_key=missing_key)
        roi_mask_gui.exec_autorois()
        roi_mask_gui.insert_database(RoiMask, missing_key)

    assert len(RoiMask()) > 0


@pytest.mark.dependency(depends=['test_populate_roi_mask'])
@pytest.mark.skipif(is_not_connected(), reason="Not connected to database")
def test_populate_roi(processes=20):
    from djimaging.schemas.core_autorois_schema import Roi
    Roi().populate(processes=processes)
    assert len(Roi()) > 0


@pytest.mark.dependency(depends=['test_populate_roi'])
@pytest.mark.skipif(is_not_connected(), reason="Not connected to database")
def test_populate_traces(processes=20):
    from djimaging.schemas.core_autorois_schema import Traces
    Traces().populate(processes=processes)
    assert len(Traces()) > 0


@pytest.mark.dependency()
@pytest.mark.skipif(is_not_connected(), reason="Not connected to database")
def test_populate_preprocessparams():
    from djimaging.schemas.core_autorois_schema import PreprocessParams
    PreprocessParams().add_default(skip_duplicates=True)
    assert len(PreprocessParams()) == 1


@pytest.mark.dependency(depends=['test_populate_preprocessparams', 'test_populate_traces'])
@pytest.mark.skipif(is_not_connected(), reason="Not connected to database")
def test_populate_preprocesstraces(processes=20):
    from djimaging.schemas.core_autorois_schema import PreprocessTraces
    PreprocessTraces().populate(processes=processes)
    assert len(PreprocessTraces()) > 0


@pytest.mark.dependency(depends=['test_populate_preprocesstraces'])
@pytest.mark.skipif(is_not_connected(), reason="Not connected to database")
def test_populate_snippets(processes=20):
    from djimaging.schemas.core_autorois_schema import Snippets
    Snippets().populate(processes=processes)
    assert len(Snippets()) > 0


@pytest.mark.dependency(depends=['test_populate_snippets'])
@pytest.mark.skipif(is_not_connected(), reason="Not connected to database")
def test_populate_averages(processes=20):
    from djimaging.schemas.core_autorois_schema import Averages
    Averages().populate(processes=processes)
    assert len(Averages()) > 0
