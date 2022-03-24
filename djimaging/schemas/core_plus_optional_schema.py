from djimaging.schemas.core_schema import schema, Stimulus, Field, Experiment, Presentation, Snippets, UserInfo, Roi
from djimaging.tables.optional import location, chirp, orientation, brightness


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
    snippets_table = Snippets
    presentation_table = Presentation


@schema
class OsDsIndexes(orientation.OsDsIndexesTemplate):
    stimulus_table = Stimulus
    snippets_table = Snippets


@schema
class RoiBrightness(brightness.RoiBrightnessTemaplte):
    field_table = Field
    presentation_table = Presentation
    roi_table = Roi
