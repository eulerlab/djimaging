import datajoint as dj

from djimaging.tables import core, core_autorois

schema = dj.Schema()


@schema
class UserInfo(core.UserInfoTemplate):
    pass


@schema
class RawDataParams(core_autorois.RawDataParamsTemplate):
    userinfo_table = UserInfo


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
class Field(core_autorois.FieldTemplate):
    userinfo_table = UserInfo
    raw_params_table = RawDataParams
    experiment_table = Experiment

    class StackAverages(core_autorois.FieldTemplate.StackAverages):
        pass


@schema
class Stimulus(core.StimulusTemplate):
    pass


@schema
class Presentation(core_autorois.PresentationTemplate):
    userinfo_table = UserInfo
    experiment_table = Experiment
    field_table = Field
    stimulus_table = Stimulus
    raw_params_table = RawDataParams

    class ScanInfo(core_autorois.PresentationTemplate.ScanInfo):
        pass

    class StackAverages(core_autorois.PresentationTemplate.StackAverages):
        pass


@schema
class RoiMask(core_autorois.RoiMaskTemplate):
    field_table = Field
    presentation_table = Presentation
    experiment_table = Experiment
    userinfo_table = UserInfo
    raw_params_table = RawDataParams

    class RoiMaskPresentation(core_autorois.RoiMaskTemplate.RoiMaskPresentation):
        presentation_table = Presentation


@schema
class Roi(core_autorois.RoiTemplate):
    roi_mask_table = RoiMask
    userinfo_table = UserInfo
    field_table = Field


@schema
class Traces(core_autorois.TracesTemplate):
    userinfo_table = UserInfo
    raw_params_table = RawDataParams
    presentation_table = Presentation
    roi_table = Roi
    roi_mask_table = RoiMask


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
