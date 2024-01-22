import datajoint as dj

from djimaging.tables import core, misc

schema = dj.Schema()


@schema
class UserInfo(core.UserInfoTemplate):
    pass


@schema
class RawDataParams(core.RawDataParamsTemplate):
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
class Field(core.FieldTemplate):
    incl_region = True  # Include region as primary key?
    incl_cond1 = True  # Include condition 1 as primary key?
    incl_cond2 = False  # Include condition 2 as primary key?
    incl_cond3 = False  # Include condition 3 as primary key?

    userinfo_table = UserInfo
    raw_params_table = RawDataParams
    experiment_table = Experiment

    class StackAverages(core.FieldTemplate.StackAverages):
        pass


@schema
class Stimulus(core.StimulusTemplate):
    pass


@schema
class Presentation(core.PresentationTemplate):
    incl_region = True  # Include region as primary key?
    incl_cond1 = True  # Include condition 1 as primary key?
    incl_cond2 = True  # Include condition 2 as primary key?
    incl_cond3 = False  # Include condition 3 as primary key?

    userinfo_table = UserInfo
    experiment_table = Experiment
    field_table = Field
    stimulus_table = Stimulus
    raw_params_table = RawDataParams

    class ScanInfo(core.PresentationTemplate.ScanInfo):
        pass

    class StackAverages(core.PresentationTemplate.StackAverages):
        pass


# Misc
@schema
class HighRes(misc.HighResTemplate):
    field_table = Field
    experiment_table = Experiment
    userinfo_table = UserInfo

    class StackAverages(misc.HighResTemplate.StackAverages):
        pass


@schema
class RoiMask(core.RoiMaskTemplate):
    _max_shift = 5  # Maximum shift of ROI mask in pixels

    field_table = Field
    presentation_table = Presentation
    experiment_table = Experiment
    userinfo_table = UserInfo
    raw_params_table = RawDataParams
    highres_table = HighRes

    class RoiMaskPresentation(core.RoiMaskTemplate.RoiMaskPresentation):
        presentation_table = Presentation


@schema
class Roi(core.RoiTemplate):
    roi_mask_table = RoiMask
    userinfo_table = UserInfo
    field_table = Field


@schema
class Traces(core.TracesTemplate):
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
    _baseline_max_dt = 2.  # seconds before stimulus used for baseline calculation

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
    _norm_kind = 'amp_one'  # How to normalize averages

    snippets_table = Snippets
