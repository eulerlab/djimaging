from djimaging.schemas.core_schema import *
from djimaging.tables import response, classifier_v2, receptivefield, location


# Classifier
@schema
class ChirpQI(response.ChirpQITemplate):
    stimulus_table = Stimulus
    snippets_table = Snippets


@schema
class ChirpFeatures(response.ChirpFeaturesRgcTemplate):
    stimulus_table = Stimulus
    snippets_table = Snippets
    presentation_table = Presentation


@schema
class OsDsIndexes(response.OsDsIndexesTemplate):
    _reduced_storage = True
    _n_shuffles = 1000

    stimulus_table = Stimulus
    snippets_table = Snippets


@schema
class Baden16TracesV2(classifier_v2.Baden16TracesV2Template):
    _stim_name_chirp = 'gChirp'
    _stim_name_bar = 'movingbar'

    traces_table = Traces
    presentation_table = Presentation
    stimulus_table = Stimulus


@schema
class ClassifierV2(classifier_v2.ClassifierV2Template):
    pass


@schema
class CelltypeAssignmentV2(classifier_v2.CelltypeAssignmentV2Template):
    classifier_table = ClassifierV2
    baden_trace_table = Baden16TracesV2
    field_table = Field
    roi_table = Roi


# Receptive fields
@schema
class FastStaParams(receptivefield.FastStaParamsTemplate):
    stimulus_table = Stimulus
    presentation_table = Presentation


@schema
class FastSta(receptivefield.FastStaTemplate):
    _traces_prefix = ''

    params_table = FastStaParams
    presentation_table = Presentation
    traces_table = Traces


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


# Retinal field location
@schema
class OpticDisk(location.OpticDiskTemplate):
    raw_params_table = RawDataParams
    userinfo_table = UserInfo
    experiment_table = Experiment


@schema
class RelativeFieldLocation(location.RelativeFieldLocationTemplate):
    field_table = Field
    opticdisk_table = OpticDisk


@schema
class RetinalFieldLocation(location.RetinalFieldLocationTemplate):
    relativefieldlocation_table = RelativeFieldLocation
    expinfo_table = Experiment.ExpInfo
