import datajoint as dj

from djimaging.tables import core

schema = dj.Schema()


@schema
class UserInfo(core.UserInfoTemplate):
    pass


@schema
class Experiment(core.ExperimentTemplate):
    userinfo_table = UserInfo

    class ExpInfo(core.ExperimentTemplate.ExpInfo):
        pass

    class Animal(core.ExperimentTemplate.Animal):
        pass

    class Indicator(core.ExperimentTemplate.Indicator):
        pass

    class PharmInfo(core.ExperimentTemplate.PharmInfo):
        pass


@schema
class Field(core.FieldTemplate):
    userinfo_table = UserInfo
    experiment_table = Experiment

    class RoiMask(core.FieldTemplate.RoiMask):
        pass

    class Zstack(core.FieldTemplate.Zstack):
        pass


@schema
class Roi(core.RoiTemplate):
    field_table = Field


@schema
class Stimulus(core.StimulusTemplate):
    pass


@schema
class Presentation(core.PresentationTemplate):
    experiment_table = Experiment
    userinfo_table = UserInfo
    field_table = Field
    stimulus_table = Stimulus

    class ScanInfo(core.PresentationTemplate.ScanInfo):
        pass


@schema
class Traces(core.TracesTemplate):
    presentation_table = Presentation
    field_table = Field
    roi_table = Roi


@schema
class PreprocessParams(core.PreprocessParamsTemplate):
    pass


@schema
class PreprocessTraces(core.PreprocessTracesTemplate):
    presentation_table = Presentation
    preprocessparams_table = PreprocessParams
    traces_table = Traces


@schema
class Snippets(core.SnippetsTemplate):
    stimulus_table = Stimulus
    presentation_table = Presentation
    traces_table = Traces
    preprocesstraces_table = PreprocessTraces


@schema
class Averages(core.AveragesTemplate):
    snippets_table = Snippets
