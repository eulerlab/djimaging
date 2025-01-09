from djimaging.tables import core
from djimaging.utils.dj_utils import get_input_tables, get_class_attributes


def _test_definition(djclass):
    tables = get_input_tables(djclass().definition)
    class_attrs = get_class_attributes(djclass)

    for table in tables:
        assert table == 'master' or table in class_attrs, table

    for class_attr in class_attrs:
        if class_attr.endswith('_table'):
            assert class_attr.lower() == class_attr, 'Use lower case'


def test_definition_experiment():
    _test_definition(core.ExperimentTemplate)


def test_definition_expinfo():
    _test_definition(core.ExperimentTemplate.ExpInfo)


def test_definition_animal():
    _test_definition(core.ExperimentTemplate.Animal)


def test_definition_indicator():
    _test_definition(core.ExperimentTemplate.Indicator)


# field
def test_definition_field():
    _test_definition(core.FieldTemplate)


def test_definition_roi():
    _test_definition(core.RoiTemplate)


# traces
def test_definition_traces():
    _test_definition(core.TracesTemplate)


def test_definition_preprocesstraces():
    _test_definition(core.PreprocessTracesTemplate)


def test_definition_snippets():
    _test_definition(core.SnippetsTemplate)
