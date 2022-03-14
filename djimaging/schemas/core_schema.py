import datajoint as dj

from djimaging.core_tables import userinfo
from djimaging.core_tables import experiment
from djimaging.core_tables import field
from djimaging.core_tables import location
from djimaging.core_tables import stimulus
from djimaging.core_tables import traces


schema = dj.Schema()


@schema
class UserInfo(userinfo.UserInfoTemplate):
    pass


@schema
class Experiment(experiment.ExperimentTemplate):
    userinfo_table = UserInfo


@schema
class Field(field.FieldTemplate):
    userinfo_table = UserInfo
    experiment_table = Experiment


@schema
class Roi(field.RoiTemplate):
    field_table = Field


@schema
class Stimulus(stimulus.StimulusTemplate):
    pass


@schema
class Presentation(stimulus.PresentationTemplate):
    experiment_table = Experiment
    userinfo_table = UserInfo
    field_table = Field
    stimulus_table = Stimulus


@schema
class RelativeFieldLocation(location.RelativeFieldLocationTemplate):
    field_table = Field
    expinfo_table = Experiment.ExpInfo


@schema
class RetinalFieldLocation(location.RetinalFieldLocationTemplate):
    relativefieldlocalation_table = RelativeFieldLocation
    expinfo_table = Experiment.ExpInfo


@schema
class Traces(traces.TracesTemplate):
    presentation_table = Presentation
    field_table = Field
    roi_table = Roi


@schema
class DetrendParams(traces.DetrendParamsTemplate):
    pass


@schema
class DetrendTraces(traces.DetrendTracesTemplate):
    presentation_table = Presentation
    detrendparams_table = DetrendParams
    traces_table = Traces


@schema
class DetrendSnippets(traces.DetrendSnippetsTemplate):
    stimulus_table = Stimulus
    presentation_table = Presentation
    traces_table = Traces
    detrendtraces_table = DetrendTraces


@schema
class Averages(traces.AveragesTemplate):
    detrendsnippets_table = DetrendSnippets
