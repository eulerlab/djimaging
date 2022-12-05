from djimaging.tables.optional import *
from .utils import _test_definition


# location
def test_definition_RelativeFieldLocationTemplate():
    _test_definition(RelativeFieldLocationTemplate)


def test_definition_RetinalFieldLocationTemplate():
    _test_definition(RetinalFieldLocationTemplate)


# chirp
def test_definition_ChirpQITemplate():
    _test_definition(ChirpQITemplate)


# orientation
def test_definition_OsDsIndexesTemplate():
    _test_definition(OsDsIndexesTemplate)


# rgc_classifier
def test_definition_CellFilterParametersTemplate():
    _test_definition(CellFilterParamsTemplate)


def test_definition_ClassifierTemplate():
    _test_definition(ClassifierTemplate)


def test_definition_ClassifierTrainingDataTemplate():
    _test_definition(ClassifierTrainingDataTemplate)


def test_definition_CelltypeAssignmentTemplate():
    _test_definition(CelltypeAssignmentTemplate)


def test_definition_ClassifierMethodTemplate():
    _test_definition(ClassifierMethodTemplate)


def test_definition_HighResTemplate():
    _test_definition(HighResTemplate)
