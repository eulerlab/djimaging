import djimaging.tables.receptivefield.split_strf
from djimaging.schemas.core_schema import *
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
class RfGlmParams(rf_glm.RfGlmParamsTemplate):
    pass


@schema
class RfGlm(rf_glm.RfGlmTemplate):
    noise_traces_table = DNoiseTrace
    params_table = RfGlmParams
    preprocesstraces_table = PreprocessTraces
    stimulus_table = Stimulus
    presentation_table = Presentation
    traces_table = Traces


@schema
class RfGlmSingleModel(rf_glm.RfGlmSingleModelTemplate):
    noise_traces_table = DNoiseTrace
    glm_table = RfGlm


@schema
class RfGlmQualityParams(rf_glm.RfGlmQualityParamsTemplate):
    pass


@schema
class RfGlmQuality(rf_glm.RfGlmQualityTemplate):
    glm_single_model_table = RfGlmSingleModel
    glm_table = RfGlm
    params_table = RfGlmQualityParams


@schema
class SplitRfGlmParams(djimaging.tables.receptivefield.split_strf.SplitRFParamsTemplate):
    pass


@schema
class SplitRfGlm(djimaging.tables.receptivefield.split_strf.SplitRFTemplate):
    rf_table = RfGlm
    split_rf_params_table = SplitRfGlmParams


@schema
class FitGauss2DRfGlm(receptivefield.FitGauss2DRFTemplate):
    split_rf_table = SplitRfGlm
    stimulus_table = Stimulus


@schema
class FitDoG2DRfGlm(receptivefield.FitDoG2DRFTemplate):
    split_rf_table = SplitRfGlm
    stimulus_table = Stimulus
