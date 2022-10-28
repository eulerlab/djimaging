from djimaging.schemas.advanced_schema import *
from djimaging.tables.morphology import *
from djimaging.tables.optional.location_from_table import *
from djimaging.tables.receptivefield import *
from djimaging.tables.special import *


@schema
class RFGLMParams(RFGLMParamsTemplate):
    pass


@schema
class RFGLM(RFGLMTemplate):
    params_table = RFGLMParams
    preprocesstraces_table = PreprocessTraces
    stimulus_table = Stimulus
    presentation_table = Presentation
    traces_table = Traces


@schema
class SplitRFParams(SplitRFParamsTemplate):
    pass


@schema
class SplitRF(SplitRFTemplate):
    rf_table = RFGLM
    split_rf_params_table = SplitRFParams


@schema
class FitGauss2DRF(FitGauss2DRFTemplate):
    split_rf_table = SplitRF
    stimulus_table = Stimulus


@schema
class FitDoG2DRF(FitDoG2DRFTemplate):
    split_rf_table = SplitRF
    stimulus_table = Stimulus


@schema
class SineSpotQI(SineSpotQITemplate):
    stimulus_table = Stimulus
    snippets_table = Snippets


@schema
class SineSpotFeatures(SineSpotFeaturesTemplate):
    stimulus_table = Stimulus
    preprocesstraces_table = PreprocessTraces
    presentation_table = Presentation


@schema
class RetinalFieldLocationTableParams(RetinalFieldLocationTableParamsTemplate):
    pass


@schema
class RetinalFieldLocationFromTable(RetinalFieldLocationFromTableTemplate):
    field_table = Field
    params_table = RetinalFieldLocationTableParams
    expinfo_table = Experiment.ExpInfo


@schema
class SWC(SWCTemplate):
    field_table = Field
    experiment_table = Experiment


@schema
class MorphPaths(MorphPathsTemplate):
    field_table = Field
    swc_table = SWC


@schema
class IPL(IPLTemplate):
    field_table = Field
    experiment_table = Experiment


@schema
class Stratification(StratificationTemplate):
    field_table = Field
    experiment_table = Experiment


@schema
class LineStack(LineStackTemplate):
    experiment_table = Experiment
    morph_table = MorphPaths
    field_table = Field
    ipl_table = IPL


@schema
class RoiStackPosParams(RoiStackPosParamsTemplate):
    pass


@schema
class FieldStackPos(FieldStackPosTemplate):
    userinfo_table = UserInfo
    experiment_table = Experiment
    field_table = Field
    linestack_table = LineStack
    params_table = RoiStackPosParams
    morph_table = MorphPaths

    class RoiStackPos(FieldStackPosTemplate.RoiStackPos):
        roi_table = Roi

    class FitInfo(FieldStackPosTemplate.FitInfo):
        pass


@schema
class FieldCalibratedStackPos(FieldCalibratedStackPosTemplate):
    fieldstackpos_table = FieldStackPos
    linestack_table = LineStack
    field_table = Field
    morph_table = MorphPaths
    params_table = RoiStackPosParams

    class RoiCalibratedStackPos(FieldCalibratedStackPosTemplate.RoiCalibratedStackPos):
        roi_table = Roi


@schema
class FieldPosMetrics(FieldPosMetricsTemplate):
    fieldcalibratedstackpos_table = FieldCalibratedStackPos
    morph_table = MorphPaths

    class RoiPosMetrics(FieldPosMetricsTemplate.RoiPosMetrics):
        roi_table = Roi
