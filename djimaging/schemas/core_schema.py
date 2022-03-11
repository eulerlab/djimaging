import datajoint as dj

from djimaging.core_tables import userinfo
from djimaging.core_tables import experiment
from djimaging.core_tables import field
from djimaging.core_tables import location
from djimaging.core_tables import stimulus
from djimaging.core_tables import traces


schema = dj.schema(dj.config["schema_name"], locals())


@schema
class UserInfo(userinfo.UserInfo):
    pass


@schema
class Experiment(experiment.Experiment):
    userinfo_table = UserInfo


@schema
class Field(field.Field):
    userinfo_table = UserInfo
    experiment_table = Experiment


@schema
class Roi(field.Roi):
    field_table = Field


@schema
class Stimulus(stimulus.Stimulus):
    pass


@schema
class Presentation(stimulus.Presentation):
    experiment_table = Experiment
    userinfo_table = UserInfo
    field_table = Field
    stimulus_table = Stimulus


@schema
class RelativeFieldLocation(location.RelativeFieldLocation):
    field_table = Field
    expinfo_table = Experiment.ExpInfo


@schema
class RetinalFieldLocation(location.RetinalFieldLocation):
    relativefieldlocalation_table = RelativeFieldLocation
    expinfo_table = Experiment.ExpInfo


@schema
class Traces(traces.Traces):
    presentation_table = Presentation
    field_table = Field
    roi_table = Roi


@schema
class DetrendParams(traces.DetrendParams):
    pass


@schema
class DetrendTraces(traces.DetrendTraces):
    presentation_table = Presentation
    detrendparams_table = DetrendParams
    traces_table = Traces


@schema
class DetrendSnippets(traces.DetrendSnippets):
    stimulus_table = Stimulus
    presentation_table = Presentation
    traces_table = Traces
    detrendtraces_table = DetrendTraces


@schema
class Averages(traces.Averages):
    detrendsnippets_table = DetrendSnippets






