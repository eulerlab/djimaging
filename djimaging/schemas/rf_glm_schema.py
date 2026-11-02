from djimaging.schemas.core_schema import *
from djimaging.tables import receptivefield


# Receptive fields: fast (optionally whitened) STA
@schema
class FastStaParams(receptivefield.FastStaParamsTemplate):
    stimulus_table = Stimulus
    presentation_table = Presentation


@schema
class FastSta(receptivefield.FastStaTemplate):
    params_table = FastStaParams
    presentation_table = Presentation
    traces_table = PreprocessTraces


@schema
class FastStaQuality(receptivefield.FastStaQualityTemplate):
    sta_table = FastSta


@schema
class SplitRFParams(receptivefield.SplitRFParamsTemplate):
    pass


@schema
class SplitRF(receptivefield.SplitRFTemplate):
    rf_table = FastSta
    split_rf_params_table = SplitRFParams


@schema
class FitGauss2DRF(receptivefield.FitGauss2DRFTemplate):
    split_rf_table = SplitRF
    stimulus_table = Stimulus


@schema
class FitDoG2DRF(receptivefield.FitDoG2DRFTemplate):
    split_rf_table = SplitRF
    stimulus_table = Stimulus


# Receptive fields: GLM
@schema
class RfGlmParams(receptivefield.RfGlmParamsTemplate):
    stimulus_table = Stimulus
    presentation_table = Presentation


@schema
class RfGlm(receptivefield.RfGlmTemplate):
    params_table = RfGlmParams
    presentation_table = Presentation
    traces_table = PreprocessTraces


@schema
class RfGlmQualityParams(receptivefield.RfGlmQualityParamsTemplate):
    pass


@schema
class RfGlmQuality(receptivefield.RfGlmQualityTemplate):
    glm_table = RfGlm
    params_table = RfGlmQualityParams


@schema
class SplitRfGlmParams(receptivefield.SplitRFParamsTemplate):
    pass


@schema
class SplitRfGlm(receptivefield.SplitRFTemplate):
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
