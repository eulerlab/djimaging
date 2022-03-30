from djimaging.schemas.core_schema import schema, Stimulus, Field, Experiment, Presentation, Snippets, UserInfo
from djimaging.tables.optional import location, chirp, orientation


@schema
class OpticDisk(location.OpticDiskTemplate):
    experiment_table = Experiment
    userinfo_table = UserInfo


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
    snippets_table = Snippets


@schema
class ChirpFeatures(chirp.ChirpFeaturesTemplate):
    stimulus_table = Stimulus
    snippets_table = Snippets
    presentation_table = Presentation


@schema
class OsDsIndexes(orientation.OsDsIndexesTemplate):
    stimulus_table = Stimulus
    snippets_table = Snippets
