import datajoint as dj

from djimaging.tables.core import stimulus, traces, preprocesstraces, userinfo, presentation, experiment, field, roi

schema = dj.Schema()


@schema
class UserInfo(userinfo.UserInfoTemplate):
    pass


@schema
class Experiment(experiment.ExperimentTemplate):
    userinfo_table = UserInfo

    class ExpInfo(experiment.ExperimentTemplate.ExpInfo):
        pass

    class Animal(experiment.ExperimentTemplate.Animal):
        pass

    class Indicator(experiment.ExperimentTemplate.Indicator):
        pass

    class PharmInfo(experiment.ExperimentTemplate.PharmInfo):
        pass


@schema
class Field(field.FieldTemplate):
    userinfo_table = UserInfo
    experiment_table = Experiment

    class RoiMask(field.FieldTemplate.RoiMask):
        pass

    class Zstack(field.FieldTemplate.Zstack):
        pass


@schema
class Roi(roi.RoiTemplate):
    field_table = Field


@schema
class Stimulus(stimulus.StimulusTemplate):
    pass


@schema
class Presentation(presentation.PresentationTemplate):
    experiment_table = Experiment
    userinfo_table = UserInfo
    field_table = Field
    stimulus_table = Stimulus

    class ScanInfo(presentation.PresentationTemplate.ScanInfo):
        pass


@schema
class Traces(traces.TracesTemplate):
    presentation_table = Presentation
    field_table = Field
    roi_table = Roi


@schema
class PreprocessParams(preprocesstraces.PreprocessParamsTemplate):
    pass


@schema
class PreprocessTraces(preprocesstraces.PreprocessTracesTemplate):
    presentation_table = Presentation
    preprocessparams_table = PreprocessParams
    traces_table = Traces


@schema
class Snippets(traces.SnippetsTemplate):
    stimulus_table = Stimulus
    presentation_table = Presentation
    traces_table = Traces
    preprocesstraces_table = PreprocessTraces


@schema
class Averages(traces.AveragesTemplate):
    snippets_table = Snippets
