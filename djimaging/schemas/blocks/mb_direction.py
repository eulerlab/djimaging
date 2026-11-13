from djimaging.schemas.blocks.basic import *
from djimaging.schemas.blocks.stimulator import *
from djimaging.tables import stimulator


# Moving bar directions on the retina
@schema
class MovingBarVersion(stimulator.MovingBarVersionTemplate):
    pass


@schema
class MbDirectionMap(stimulator.MbDirectionMapTemplate):
    presentation_table = Presentation
    stimulus_table = Stimulus
    experiment_stimulator_table = ExperimentStimulator
    mb_version_table = MovingBarVersion
    expinfo_table = Experiment.ExpInfo
    # The moving bar version is identified by the QDSpy md5 in the logs. Without logs every row has low confidence.
    # Set these if the schema has QDSpy log tables:
    presentation_log_table = None  # PresentationLog
    stim_log_table = None  # QdsPyLog.StimLog


@schema
class MbRetinalPrefDir(stimulator.MbRetinalPrefDirTemplate):
    osds_table = OsDsIndexes
    mb_direction_map_table = MbDirectionMap
    retinal_field_location_table = None  # RetinalFieldLocation, for the angle relative to the optic disc
