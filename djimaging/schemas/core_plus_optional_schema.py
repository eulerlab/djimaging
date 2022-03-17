from djimaging.schemas.core_schema import schema, Stimulus, Field, Experiment, Presentation, DetrendSnippets
from djimaging.tables.optional import location, chirp, orientation


@schema
class OpticDisk(location.OpticDiskTemplate):
    expinfo_table = Experiment.ExpInfo
    field_table = Field


@schema
class RelativeFieldLocation(location.RelativeFieldLocationTemplate):
    field_table = Field
    opticdisk_table = OpticDisk


@schema
class RetinalFieldLocation(location.RetinalFieldLocationTemplate):
    relativefieldlocalation_table = RelativeFieldLocation
    expinfo_table = Experiment.ExpInfo


@schema
class ChirpQI(chirp.ChirpQITemplate):
    stimulus_table = Stimulus
    detrendsnippets_table = DetrendSnippets


@schema
class ChirpFeatures(chirp.ChirpFeaturesTemplate):
    detrendsnippets_table = DetrendSnippets
    presentation_table = Presentation


@schema
class OsDsIndexes(orientation.OsDsIndexesTemplate):
    stimulus_table = Stimulus
    detrendsnippets_table = DetrendSnippets
