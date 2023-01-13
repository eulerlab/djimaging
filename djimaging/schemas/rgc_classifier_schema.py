from djimaging.schemas.advanced_schema import *
from djimaging.tables import optional


@schema
class CellFilterParams(optional.CellFilterParamsTemplate):
    pass


@schema
class ClassifierTrainingData(optional.ClassifierTrainingDataTemplate):
    pass


@schema
class ClassifierMethod(optional.ClassifierMethodTemplate):
    classifier_training_data_table = ClassifierTrainingData


@schema
class Classifier(optional.ClassifierTemplate):
    classifier_training_data_table = ClassifierTrainingData
    classifier_method_table = ClassifierMethod


@schema
class CelltypeAssignment(optional.CelltypeAssignmentTemplate):
    field_table = Field
    classifier_training_data_table = ClassifierTrainingData
    cell_filter_parameter_table = CellFilterParams
    classifier_table = Classifier
    roi_table = Roi
    snippets_table = Snippets
    chirp_qi_table = ChirpQI
    or_dir_index_table = OsDsIndexes
    detrend_params_table = PreprocessParams
