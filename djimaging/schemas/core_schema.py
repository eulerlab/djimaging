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
    _load_field_roi_masks = True  # Set to False if you don't want to use Field level Roi masks!
    userinfo_table = UserInfo
    experiment_table = Experiment

    class RoiMask(core.FieldTemplate.RoiMask):
        pass

    class StackAverages(core.FieldTemplate.StackAverages):
        pass


@schema
class Stimulus(core.StimulusTemplate):
    pass


@schema
class RawDataParams(core.RawDataParamsTemplate):
    pass


@schema
class Presentation(core.PresentationTemplate):
    userinfo_table = UserInfo
    experiment_table = Experiment
    field_table = Field
    stimulus_table = Stimulus
    params_table = RawDataParams

    class ScanInfo(core.PresentationTemplate.ScanInfo):
        pass

    class StackAverages(core.PresentationTemplate.StackAverages):
        pass

    class RoiMask(core.PresentationTemplate.RoiMask):
        pass


@schema
class Roi(core.RoiTemplate):
    userinfo_table = UserInfo
    field_or_pres_table = Field  # Can also be set to Presentation


@schema
class Traces(core.TracesTemplate):
    userinfo_table = UserInfo
    params_table = RawDataParams
    presentation_table = Presentation
    roi_mask_table = Field.RoiMask  # Can also be set to Presentation.RoiMask
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
    _norm_kind = 'amp_one'
    snippets_table = Snippets
