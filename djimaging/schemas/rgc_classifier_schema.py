from djimaging.schemas.advanced_schema import *
from djimaging.tables import classifier


@schema
class Baden16Traces(classifier.Baden16TracesTemplate):
    roi_table = Roi
    preprocessparams_table = PreprocessParams
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
