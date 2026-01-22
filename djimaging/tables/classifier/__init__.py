from .rgc_classifier import ClassifierTemplate, ClassifierMethodTemplate, ClassifierTrainingDataTemplate
from .celltype_assignment import CelltypeAssignmentTemplate
from .baden16_traces import Baden16TracesTemplate

"""
DEPRECATED: Use djimaging.tables.classifier_v2 instead.

Classifier for RGCs used Baden et al. 2016 dataset.

Example usage:

from djimaging.tables import classifier

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
"""
