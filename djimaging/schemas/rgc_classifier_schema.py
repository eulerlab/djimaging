from djimaging.schemas.advanced_schema import *


@schema
class CellFilterParams(rgc_classifier.CellFilterParamsTemplate):
    pass


@schema
class ClassifierTrainingData(rgc_classifier.ClassifierTrainingDataTemplate):
    store = "classifier_input"
    pass


@schema
class ClassifierMethod(rgc_classifier.ClassifierMethodTemplate):
    classifier_training_data_table = ClassifierTrainingData


@schema
class Classifier(rgc_classifier.ClassifierTemplate):
    store = "classifier_output"
    classifier_training_data_table = ClassifierTrainingData
    classifier_method_table = ClassifierMethod


@schema
class CelltypeAssignment(rgc_classifier.CelltypeAssignmentTemplate):
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
