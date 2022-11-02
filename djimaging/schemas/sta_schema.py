from djimaging.schemas.advanced_schema import *
from djimaging.tables.receptivefield import *


@schema
class STAParams(STAParamsTemplate):
    pass


@schema
class STA(STATemplate):
    stimulus_table = Stimulus
    presentation_table = Presentation
    traces_table = Traces
    preprocesstraces_table = PreprocessTraces
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
