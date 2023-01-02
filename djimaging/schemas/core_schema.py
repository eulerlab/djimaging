import datajoint as dj

from djimaging.tables.core import *
from djimaging.tables.core import SnippetsTemplate, AveragesTemplate

schema = dj.Schema()


@schema
class UserInfo(UserInfoTemplate):
    pass


@schema
class Experiment(ExperimentTemplate):
    userinfo_table = UserInfo

    class ExpInfo(ExperimentTemplate.ExpInfo):
        pass

    class Animal(ExperimentTemplate.Animal):
        pass

    class Indicator(ExperimentTemplate.Indicator):
        pass

    class PharmInfo(ExperimentTemplate.PharmInfo):
        pass


@schema
class Field(FieldTemplate):
    userinfo_table = UserInfo
    experiment_table = Experiment

    class RoiMask(FieldTemplate.RoiMask):
        pass

    class Zstack(FieldTemplate.Zstack):
        pass


@schema
class Roi(RoiTemplate):
    field_table = Field


@schema
class Stimulus(StimulusTemplate):
    pass


@schema
class Presentation(PresentationTemplate):
    experiment_table = Experiment
    userinfo_table = UserInfo
    field_table = Field
    stimulus_table = Stimulus

    class ScanInfo(PresentationTemplate.ScanInfo):
        pass


@schema
class Traces(TracesTemplate):
    presentation_table = Presentation
    field_table = Field
    roi_table = Roi


@schema
class PreprocessParams(PreprocessParamsTemplate):
    pass


@schema
class PreprocessTraces(PreprocessTracesTemplate):
    presentation_table = Presentation
    preprocessparams_table = PreprocessParams
    traces_table = Traces


@schema
class Snippets(SnippetsTemplate):
    stimulus_table = Stimulus
    presentation_table = Presentation
    traces_table = Traces
    preprocesstraces_table = PreprocessTraces


@schema
class Averages(AveragesTemplate):
    snippets_table = Snippets
