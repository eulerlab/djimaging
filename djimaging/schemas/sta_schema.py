from djimaging.schemas.core_schema import *
from djimaging.tables import receptivefield


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
class STAParams(receptivefield.STAParamsTemplate):
    pass


@schema
class STA(receptivefield.STATemplate):
    noise_traces_table = DNoiseTrace
    params_table = STAParams

    class DataSet(receptivefield.STATemplate.DataSet):
        pass


@schema
class SplitRFParams(receptivefield.SplitRFParamsTemplate):
    pass


@schema
class SplitRF(receptivefield.SplitRFTemplate):
    stimulus_table = Stimulus
    rf_table = STA
    split_rf_params_table = SplitRFParams


@schema
class FitGauss2DRF(receptivefield.FitGauss2DRFTemplate):
    split_rf_table = SplitRF
    stimulus_table = Stimulus


@schema
class FitDoG2DRF(receptivefield.FitDoG2DRFTemplate):
    split_rf_table = SplitRF
    stimulus_table = Stimulus
