from djimaging.tables import optional, response, location
from .utils import _test_definition


# location
def test_definition_RelativeFieldLocationTemplate():
    _test_definition(location.RelativeFieldLocationTemplate)


def test_definition_RetinalFieldLocationTemplate():
    _test_definition(location.RetinalFieldLocationTemplate)


# chirp
def test_definition_ChirpQITemplate():
    _test_definition(response.ChirpQITemplate)


# orientation
def test_definition_OsDsIndexesTemplate():
    _test_definition(response.OsDsIndexesTemplate)


# rgc_classifier
def test_definition_CellFilterParametersTemplate():
    _test_definition(optional.CellFilterParamsTemplate)


def test_definition_ClassifierTemplate():
    _test_definition(optional.ClassifierTemplate)


def test_definition_ClassifierTrainingDataTemplate():
    _test_definition(optional.ClassifierTrainingDataTemplate)


def test_definition_CelltypeAssignmentTemplate():
    _test_definition(optional.CelltypeAssignmentTemplate)


def test_definition_ClassifierMethodTemplate():
    _test_definition(optional.ClassifierMethodTemplate)


def test_definition_HighResTemplate():
    _test_definition(optional.HighResTemplate)
