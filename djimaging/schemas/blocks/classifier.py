from djimaging.schemas.blocks.core import *
from djimaging.tables import classifier_v2


# RGC classifier
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
