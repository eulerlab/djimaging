# Choose the core schema you want to use:
# if you want to change the core schema, you have to drop the schema first, then change it here and restart.
# Use this for standard schema:
from djimaging.schemas.core_schema import *
# Use this for autorois schema:
# from djimaging.schemas.core_autorois_schema import *

# Load some additional tables, compatible with both core schemas.
from djimaging.tables import misc, response, location


@schema
class OpticDisk(location.OpticDiskTemplate):
    userinfo_table = UserInfo
    experiment_table = Experiment


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
class ChirpFeatures(response.ChirpFeaturesRgcTemplate):
    stimulus_table = Stimulus
    snippets_table = Snippets
    presentation_table = Presentation


@schema
class OsDsIndexes(response.OsDsIndexesTemplateV2):
    stimulus_table = Stimulus
    snippets_table = Snippets


@schema
class HighRes(misc.HighResTemplate):
    field_table = Field
    experiment_table = Experiment
    userinfo_table = UserInfo

    class StackAverages(misc.HighResTemplate.StackAverages):
        pass
