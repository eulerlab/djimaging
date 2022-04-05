from djimaging.schemas.core_schema import UserInfo, Field, Roi, Presentation, Snippets, PreprocessParams
from djimaging.schemas.core_plus_optional_schema import schema, ChirpQI, OsDsIndexes
from djimaging.tables.optional import rgc_classifier


@schema
class ClassifierSeed(rgc_classifier.ClassifierSeedTemplate):
    pass


@schema
class CellFilterParameters(rgc_classifier.CellFilterParametersTemplate):
    pass


@schema
class ClassifierTrainingData(rgc_classifier.ClassifierTrainingDataTemplate):
    store = "classifier"
    pass


@schema
class ClassifierMethod(rgc_classifier.ClassifierMethodTemplate):
    classifier_training_data_table = ClassifierTrainingData


@schema
class Classifier(rgc_classifier.ClassifierTemplate):
    store = "classifier"
    classifier_training_data_table = ClassifierTrainingData
    classifier_method_table = ClassifierMethod
    classifier_seed_table = ClassifierSeed


@schema
class CelltypeAssignment(rgc_classifier.CelltypeAssignmentTemplate):
    classifier_training_data_table = ClassifierTrainingData
    cell_filter_parameter_table = CellFilterParameters
    classifier_table = Classifier
    userInfo_table = UserInfo
    field_table = Field
    roi_table = Roi
    presentation_table = Presentation
    detrend_snippets_table = Snippets
    chirp_qi_table = ChirpQI
    or_dir_index_table = OsDsIndexes
    detrend_params_table = PreprocessParams

    @property
    def key_source(self):
        return self.classifier_training_data_table() * self.classifier_table() * \
               self.field_table() * self.detrend_params_table() * self.cell_filter_parameter_table()
