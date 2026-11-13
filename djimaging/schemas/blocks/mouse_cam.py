from djimaging.schemas.blocks.stimulator import *
from djimaging.tables import stimulator


# Orientation of mouse cam (natural movie) presentations in the recording field
@schema
class MouseCamGeometry(stimulator.MouseCamGeometryTemplate):
    presentation_table = Presentation
    stimulus_table = Stimulus
    experiment_stimulator_table = ExperimentStimulator
    expinfo_table = Experiment.ExpInfo
    # The MouseCam version is read from the QDSpy logs (history block). Without them every row has low confidence.
    presentation_log_table = None  # PresentationLog
    stim_log_table = None  # QdsPyLog.StimLog
