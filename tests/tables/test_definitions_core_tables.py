from djimaging.tables import core

from tests.utils import _test_definition


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


# presentation
def test_definition_presentation():
    _test_definition(core.PresentationTemplate)
