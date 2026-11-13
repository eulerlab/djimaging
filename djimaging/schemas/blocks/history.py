from djimaging.schemas.blocks.core import *
from djimaging.tables import core, history


# Experiment history: every recording of every experiment, also of unlisted stimuli, and the stimulus logs
@schema
class QdsPyLogFile(core.QdsPyLogFileTemplate):
    exp_table = Experiment


@schema
class QdsPyLog(core.QdsPyLogTemplate):
    log_file_table = QdsPyLogFile

    class StimLog(core.QdsPyLogTemplate.StimLog):
        pass


@schema
class ExpHistory(history.ExpHistoryTemplate):
    extra_data_dirs = ()  # More folders (relative to the experiment folder) to scan, besides Pre and Raw

    experiment_table = Experiment
    userinfo_table = UserInfo

    class Recording(history.ExpHistoryTemplate.Recording):
        pass


@schema
class ExpHistoryField(history.ExpHistoryFieldTemplate):
    field_table = Field
    exp_history_table = ExpHistory

    class Recording(history.ExpHistoryFieldTemplate.Recording):
        recording_table = ExpHistory.Recording


@schema
class ExpHistoryStimLog(history.ExpHistoryStimLogTemplate):
    exp_history_table = ExpHistory
    log_table = QdsPyLog

    class Stim(history.ExpHistoryStimLogTemplate.Stim):
        stim_log_table = QdsPyLog.StimLog


@schema
class PresentationLog(core.PresentationLogTemplate):
    log_table = QdsPyLog.StimLog
    stimulus_table = Stimulus
    exp_table = Experiment
    presentation_table = Presentation
