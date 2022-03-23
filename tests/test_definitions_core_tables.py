from djimaging.tables.core import traces, presentation, experiment, field, roi

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
    _test_definition(roi.RoiTemplate)


# traces
def test_definition_traces():
    _test_definition(traces.TracesTemplate)


def test_definition_preprocesstraces():
    _test_definition(traces.PreprocessTracesTemplate)


def test_definition_snippets():
    _test_definition(traces.SnippetsTemplate)


# presentation
def test_definition_presentation():
    _test_definition(presentation.PresentationTemplate)



