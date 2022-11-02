from djimaging.schemas.advanced_schema import *
from djimaging.tables.receptivefield import *


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
class RFGLMQualityParams(RFGLMQualityParamsTemplate):
    pass


@schema
class RFGLMQuality(RFGLMQualityTemplate):
    glm_table = RFGLM
    params_table = RFGLMQualityParams


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
