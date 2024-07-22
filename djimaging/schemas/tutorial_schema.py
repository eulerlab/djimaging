import djimaging.tables.response.chirp_features_bc
from djimaging.schemas.core_schema import *
from djimaging.tables import misc, response, location


@schema
class OpticDisk(location.OpticDiskTemplate):
    userinfo_table = UserInfo
    experiment_table = Experiment
    raw_params_table = RawDataParams


@schema
class RelativeFieldLocation(location.RelativeFieldLocationTemplate):
    field_table = Field
    opticdisk_table = OpticDisk


@schema
class RetinalFieldLocation(location.RetinalFieldLocationTemplate):
    relativefieldlocation_table = RelativeFieldLocation
    expinfo_table = Experiment.ExpInfo


@schema
class ChirpQI(response.ChirpQITemplate):
    stimulus_table = Stimulus
    snippets_table = Snippets


@schema
class ChirpFeatures(response.ChirpFeaturesBcTemplate):
    stimulus_table = Stimulus
    snippets_table = Snippets
    presentation_table = Presentation


@schema
class OsDsIndexes(response.OsDsIndexesTemplate):
    stimulus_table = Stimulus
    snippets_table = Snippets


@schema
class HighRes(misc.HighResTemplate):
    field_table = Field
    experiment_table = Experiment
    userinfo_table = UserInfo

    class StackAverages(misc.HighResTemplate.StackAverages):
        pass
