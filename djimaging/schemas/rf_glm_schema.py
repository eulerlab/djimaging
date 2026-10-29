from djimaging.schemas.core_schema import *
from djimaging.tables import receptivefield


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
