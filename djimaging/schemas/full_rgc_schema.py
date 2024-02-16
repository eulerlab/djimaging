import djimaging.tables.receptivefield.split_strf
from djimaging.schemas.core_schema import *
from djimaging.tables import response, classifier, receptivefield, location


# Classifier
@schema
class ChirpQI(response.ChirpQITemplate):
    stimulus_table = Stimulus
    snippets_table = Snippets


@schema
class ChirpFeatures(response.ChirpFeaturesTemplate):
    stimulus_table = Stimulus
    snippets_table = Snippets
    presentation_table = Presentation


@schema
class OsDsIndexes(response.OsDsIndexesTemplate):
    stimulus_table = Stimulus
    snippets_table = Snippets


@schema
class Baden16Traces(classifier.Baden16TracesTemplate):
    _shift_chirp = 1
    _shift_bar = -4

    _stim_name_chirp = 'gChirp'
    _stim_name_bar = 'movingbar'

    averages_table = Averages
    os_ds_table = OsDsIndexes


@schema
class ClassifierTrainingData(classifier.ClassifierTrainingDataTemplate):
    pass


@schema
class ClassifierMethod(classifier.ClassifierMethodTemplate):
    classifier_training_data_table = ClassifierTrainingData


@schema
class Classifier(classifier.ClassifierTemplate):
    classifier_training_data_table = ClassifierTrainingData
    classifier_method_table = ClassifierMethod


@schema
class CelltypeAssignment(classifier.CelltypeAssignmentTemplate):
    classifier_training_data_table = ClassifierTrainingData
    classifier_table = Classifier
    baden_trace_table = Baden16Traces
    field_table = Field
    roi_table = Roi
    os_ds_table = OsDsIndexes


# Receptive fields
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
class SplitRFParams(djimaging.tables.receptivefield.split_strf.SplitRFParamsTemplate):
    pass


@schema
class SplitRF(djimaging.tables.receptivefield.split_strf.SplitRFTemplate):
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
