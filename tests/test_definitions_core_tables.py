from djimaging.core_tables import experiment
from djimaging.core_tables import field
from djimaging.core_tables import location
from djimaging.core_tables import traces
from djimaging.core_tables import stimulus
from djimaging.utils.dj_utils import get_class_attributes, get_input_tables


def _test_definition(class_):
    tables = get_input_tables(class_().definition)
    class_attrs = get_class_attributes(class_)

    for table in tables:
        assert table == 'master' or table in class_attrs, table


# experiment
def test_definition_experiment():
    _test_definition(experiment.Experiment)


def test_definition_expinfo():
    _test_definition(experiment.Experiment.ExpInfo)


def test_definition_animal():
    _test_definition(experiment.Experiment.Animal)


def test_definition_indicator():
    _test_definition(experiment.Experiment.Indicator)


# field
def test_definition_field():
    _test_definition(field.Field)


def test_definition_roi():
    _test_definition(field.Roi)


# location
def test_definition_relativefieldlocation():
    _test_definition(location.RelativeFieldLocation)


def test_definition_retinalfieldlocation():
    _test_definition(location.RetinalFieldLocation)


# traces
def test_definition_traces():
    _test_definition(traces.Traces)


def test_definition_detrendtraces():
    _test_definition(traces.DetrendTraces)


def test_definition_detrendsnippets():
    _test_definition(traces.DetrendSnippets)


# stimulus
def test_definition_presentation():
    _test_definition(stimulus.Presentation)



