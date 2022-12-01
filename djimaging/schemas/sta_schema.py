from djimaging.schemas.advanced_schema import *
from djimaging.tables.receptivefield import *


@schema
class DNoiseTraceParams(DNoiseTraceParamsTemplate):
    pass


@schema
class DNoiseTrace(DNoiseTraceTemplate):
    presentation_table = Presentation
    stimulus_table = Stimulus
    traces_table = PreprocessTraces
    params_table = DNoiseTraceParams


@schema
class STAParams(STAParamsTemplate):
    pass


@schema
class STA(STATemplate):
    noise_traces_table = DNoiseTrace
    params_table = STAParams

    class DataSet(STATemplate.DataSet):
        pass


@schema
class SplitRFParams(SplitRFParamsTemplate):
    pass


@schema
class SplitRF(SplitRFTemplate):
    rf_table = STA
    split_rf_params_table = SplitRFParams


@schema
class FitGauss2DRF(FitGauss2DRFTemplate):
    split_rf_table = SplitRF
    stimulus_table = Stimulus


@schema
class FitDoG2DRF(FitDoG2DRFTemplate):
    split_rf_table = SplitRF
    stimulus_table = Stimulus
