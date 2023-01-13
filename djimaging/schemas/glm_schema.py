from djimaging.schemas.advanced_schema import *
from djimaging.tables import receptivefield, rf_glm


@schema
class DNoiseTraceParams(receptivefield.DNoiseTraceParamsTemplate):
    pass


@schema
class DNoiseTrace(receptivefield.DNoiseTraceTemplate):
    presentation_table = Presentation
    stimulus_table = Stimulus
    traces_table = PreprocessTraces
    params_table = DNoiseTraceParams


@schema
class RFGLMParams(rf_glm.RFGLMParamsTemplate):
    pass


@schema
class RFGLM(rf_glm.RFGLMTemplate):
    noise_traces_table = DNoiseTrace
    params_table = RFGLMParams
    preprocesstraces_table = PreprocessTraces
    stimulus_table = Stimulus
    presentation_table = Presentation
    traces_table = Traces


@schema
class RFGLMSingleModel(rf_glm.RFGLMSingleModelTemplate):
    glm_table = RFGLM


@schema
class RFGLMQualityParams(rf_glm.RFGLMQualityParamsTemplate):
    pass


@schema
class RFGLMQuality(rf_glm.RFGLMQualityTemplate):
    glm_single_model_table = RFGLMSingleModel
    glm_table = RFGLM
    params_table = RFGLMQualityParams


@schema
class SplitRFParams(receptivefield.SplitRFParamsTemplate):
    pass


@schema
class SplitRF(receptivefield.SplitRFTemplate):
    rf_table = RFGLM
    split_rf_params_table = SplitRFParams


@schema
class FitGauss2DRF(receptivefield.FitGauss2DRFTemplate):
    split_rf_table = SplitRF
    stimulus_table = Stimulus


@schema
class FitDoG2DRF(receptivefield.FitDoG2DRFTemplate):
    split_rf_table = SplitRF
    stimulus_table = Stimulus
