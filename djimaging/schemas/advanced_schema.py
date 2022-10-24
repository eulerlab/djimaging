from djimaging.schemas.core_schema import *
from djimaging.tables.optional import *
from djimaging.tables.optional import RetinalFieldLocationTemplate


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
class OsDsIndexes(OsDsIndexesTemplate):
    stimulus_table = Stimulus
    snippets_table = Snippets


@schema
class HighRes(HighResTemplate):
    field_table = Field
    experiment_table = Experiment
    userinfo_table = UserInfo


@schema
class ReceptiveFieldParams(STAParamsTemplate):
    pass


@schema
class ReceptiveField(STATemplate):
    stimulus_table = Stimulus
    presentation_table = Presentation
    traces_table = Traces
    preprocesstraces_table = PreprocessTraces
    params_table = ReceptiveFieldParams

    class DataSet(STATemplate.DataSet):
        pass
