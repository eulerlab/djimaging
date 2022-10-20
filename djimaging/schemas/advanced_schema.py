from djimaging.schemas.core_schema import *
from djimaging.tables.optional import *


@schema
class OpticDisk(OpticDiskTemplate):
    userinfo_table = UserInfo
    experiment_table = Experiment


@schema
class RelativeFieldLocation(RelativeFieldLocationTemplate):
    field_table = Field
    opticdisk_table = OpticDisk


@schema
class RetinalFieldLocation(RetinalFieldLocationTemplate):
    relativefieldlocalation_table = RelativeFieldLocation
    expinfo_table = Experiment.ExpInfo


@schema
class ChirpQI(ChirpQITemplate):
    stimulus_table = Stimulus
    snippets_table = Snippets


@schema
class ChirpFeatures(ChirpFeaturesTemplate):
    stimulus_table = Stimulus
    snippets_table = Snippets
    presentation_table = Presentation


@schema
class OsDsIndexes(orientation.OsDsIndexesTemplate):
    stimulus_table = Stimulus
    snippets_table = Snippets


@schema
class HighRes(highresolution.HighResTemplate):
    field_table = Field
    experiment_table = Experiment
    userinfo_table = UserInfo


@schema
class ReceptiveFieldParams(receptivefield.ReceptiveFieldParamsTemplate):
    pass


@schema
class ReceptiveField(receptivefield.ReceptiveFieldTemplate):
    stimulus_table = Stimulus
    presentation_table = Presentation
    traces_table = Traces
    preprocesstraces_table = PreprocessTraces
    receptivefieldparams_table = ReceptiveFieldParams

    class DataSet(receptivefield.ReceptiveFieldTemplate.DataSet):
        pass
