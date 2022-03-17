from djimaging.tables.core import location, traces, presentation, experiment, field

from .utils import _test_definition


# experiment
def test_definition_experiment():
    _test_definition(experiment.ExperimentTemplate)


def test_definition_expinfo():
    _test_definition(experiment.ExperimentTemplate.ExpInfo)


def test_definition_animal():
    _test_definition(experiment.ExperimentTemplate.Animal)


def test_definition_indicator():
    _test_definition(experiment.ExperimentTemplate.Indicator)


# field
def test_definition_field():
    _test_definition(field.FieldTemplate)


def test_definition_roi():
    _test_definition(field.RoiTemplate)


# location
def test_definition_relativefieldlocation():
    _test_definition(location.RelativeFieldLocationTemplate)


def test_definition_retinalfieldlocation():
    _test_definition(location.RetinalFieldLocationTemplate)


# traces
def test_definition_traces():
    _test_definition(traces.TracesTemplate)


def test_definition_detrendtraces():
    _test_definition(traces.DetrendTracesTemplate)


def test_definition_detrendsnippets():
    _test_definition(traces.DetrendSnippetsTemplate)


# presentation
def test_definition_presentation():
    _test_definition(presentation.PresentationTemplate)



