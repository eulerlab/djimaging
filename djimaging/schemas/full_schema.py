from djimaging.schemas.advanced_schema import *
from djimaging.tables.optional.location_from_table import RetinalFieldLocationTableParamsTemplate, \
    RetinalFieldLocationFromTableTemplate
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
