from djimaging.tables.core import *
from djimaging.tables.core import SnippetsTemplate

from .utils import _test_definition


def test_definition_experiment():
    _test_definition(ExperimentTemplate)


def test_definition_expinfo():
    _test_definition(ExperimentTemplate.ExpInfo)


def test_definition_animal():
    _test_definition(ExperimentTemplate.Animal)


def test_definition_indicator():
    _test_definition(ExperimentTemplate.Indicator)


# field
def test_definition_field():
    _test_definition(FieldTemplate)


def test_definition_roi():
    _test_definition(RoiTemplate)


# traces
def test_definition_traces():
    _test_definition(TracesTemplate)


def test_definition_preprocesstraces():
    _test_definition(PreprocessTracesTemplate)


def test_definition_snippets():
    _test_definition(SnippetsTemplate)


# presentation
def test_definition_presentation():
    _test_definition(PresentationTemplate)
