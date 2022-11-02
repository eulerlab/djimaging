from djimaging.schemas.advanced_schema import *


@schema
class CellFilterParams(CellFilterParamsTemplate):
    pass


@schema
class ClassifierTrainingData(ClassifierTrainingDataTemplate):
    pass


@schema
class ClassifierMethod(ClassifierMethodTemplate):
    classifier_training_data_table = ClassifierTrainingData


@schema
class Classifier(ClassifierTemplate):
    classifier_training_data_table = ClassifierTrainingData
    classifier_method_table = ClassifierMethod


@schema
class CelltypeAssignment(CelltypeAssignmentTemplate):
    classifier_training_data_table = ClassifierTrainingData
    cell_filter_parameter_table = CellFilterParams
    classifier_table = Classifier
    user_info_table = UserInfo
    field_table = Field
    roi_table = Roi
    presentation_table = Presentation
    snippets_table = Snippets
    chirp_qi_table = ChirpQI
    or_dir_index_table = OsDsIndexes
    detrend_params_table = PreprocessParams
