from djimaging.tables.optional import location, chirp, orientation, rgc_classifier

from .utils import _test_definition


# location
def test_definition_RelativeFieldLocationTemplate():
    _test_definition(location.RelativeFieldLocationTemplate)


def test_definition_RetinalFieldLocationTemplate():
    _test_definition(location.RetinalFieldLocationTemplate)


# chirp
def test_definition_ChirpQITemplate():
    _test_definition(chirp.ChirpQITemplate)


# orientation
def test_definition_OsDsIndexesTemplate():
    _test_definition(orientation.OsDsIndexesTemplate)


# rgc_classifier
def test_definition_CellFilterParametersTemplate():
    _test_definition(rgc_classifier.CellFilterParamsTemplate)


def test_definition_ClassifierTemplate():
    _test_definition(rgc_classifier.ClassifierTemplate)


def test_definition_ClassifierTrainingDataTemplate():
    _test_definition(rgc_classifier.ClassifierTrainingDataTemplate)


def test_definition_CelltypeAssignmentTemplate():
    _test_definition(rgc_classifier.CelltypeAssignmentTemplate)


def test_definition_ClassifierMethodTemplate():
    _test_definition(rgc_classifier.ClassifierMethodTemplate)

