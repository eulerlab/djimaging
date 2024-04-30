from djimaging.schemas.core_schema import *
from djimaging.tables import receptivefield


@schema
class DNoiseTraceParams(receptivefield.DNoiseTraceParamsTemplate):
    stimulus_table = Stimulus


@schema
class DNoiseTrace(receptivefield.DNoiseTraceTemplate):
    presentation_table = Presentation
    stimulus_table = Stimulus
    traces_table = PreprocessTraces
    params_table = DNoiseTraceParams


@schema
class RfGlmParams(receptivefield.glm.RfGlmParamsTemplate):
    pass


@schema
class RfGlm(receptivefield.RfGlmTemplate):
    noise_traces_table = DNoiseTrace
    params_table = RfGlmParams
    preprocesstraces_table = PreprocessTraces
    stimulus_table = Stimulus
    presentation_table = Presentation
    traces_table = Traces


@schema
class RfGlmSingleModel(receptivefield.RfGlmSingleModelTemplate):
    noise_traces_table = DNoiseTrace
    glm_table = RfGlm


@schema
class RfGlmQualityParams(receptivefield.RfGlmQualityParamsTemplate):
    pass


@schema
class RfGlmQuality(receptivefield.RfGlmQualityTemplate):
    glm_single_model_table = RfGlmSingleModel
    glm_table = RfGlm
    params_table = RfGlmQualityParams


@schema
class SplitRfGlmParams(receptivefield.split_strf.SplitRFParamsTemplate):
    pass


@schema
class SplitRfGlm(receptivefield.split_strf.SplitRFTemplate):
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
