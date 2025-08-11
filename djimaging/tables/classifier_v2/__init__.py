from .rgc_classifier_v2 import ClassifierV2Template
from .baden16_traces_v2 import Baden16TracesV2Template
from .celltype_assignment_v2 import CelltypeAssignmentV2Template, extract_features

"""
Improved classifier for RGCs used Baden et al. 2016 dataset.

Example usage:

from djimaging.tables import classifier_v2

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
"""
