# %% [markdown]
# # Experiment history
#
# Registers **every** ScanM recording of every experiment, independent of stimuli and fields: also z-stacks, high
# resolution and blood vessel recordings, test recordings, recordings that were stopped early and recordings of
# stimuli that are not in the `Stimulus` table ("unlisted", e.g. not yet published stimuli).
#
# - `ExpHistory`: per experiment the scanned folders (`pre_data_dir` and `raw_data_dir` of `UserInfo`, plus
#   `extra_data_dirs`), and per recording (`ExpHistory.Recording`) the files (`.smp` + `.smh` and/or `.h5`, and how
#   they were linked), the parts of the file name and the scan parameters needed for the laser exposure history:
#   start time, duration, zoom, pixel size, pixel dwell time, stage position, setup, number of triggers.
#   Laser power and stimulus intensity are not stored digitally and are therefore not part of the history.
# - `ExpHistoryField`: the recordings of each field.
# - `ExpHistoryStimLog`: the QDSpy stimuli of an experiment and the recording during which each was shown (the clocks
#   of the QDSpy and ScanM computers are not synchronized, the offset is estimated).
#
# **Listed / unlisted**: a recording is *listed* if the stimulus part of its file name is an alias of a stimulus in
# `Stimulus`. Only listed recordings become `Presentation`s, so only they get traces, snippets and averages.

# %%
ExpHistory().populate(display_progress=True, processes=5)
ExpHistory.Recording()

# %%
# Recordings of stimuli that are not in the Stimulus table: registered, but no traces
df_listed = ExpHistory().listed(Stimulus)
if len(df_listed) > 0:
    print(df_listed.groupby(['kind', 'listed']).size())
    print(df_listed[(df_listed.kind == 'response') & ~df_listed.listed].stim_token.value_counts())

# %%
# ScanM files that are not in the scanned folders. Add their folders to ExpHistory.extra_data_dirs to register them.
for key, unscanned in zip(*ExpHistory().to_arrays('unscanned_files', include_key=True)):
    if len(unscanned) > 0:
        print(key, unscanned)

# %%
plot_if_data(ExpHistory())

# %%
ExpHistoryField().populate(display_progress=True, processes=5)
ExpHistoryField.Recording()

# %% [markdown]
# ## Stimulus logs (QDSpy)
#
# Set the folders with the QDSpy log files of your setups. Log files are matched to experiments by the date at the
# start of their file name (`YYYYMMDD_...`).

# %%
qdspy_log_dirs = dict()  # {setupid: folder with QDSpy log files}, e.g. {1: reference_root / 'Resources/__Setup_1/log files'}

for setupid, log_dir in qdspy_log_dirs.items():
    log_files = sorted(f for f in os.listdir(log_dir) if f[:8].isdigit())
    for exp_key in (Experiment.ExpInfo() & dict(setupid=str(setupid))).proj().to_dicts():
        date_str = exp_key['date'].strftime('%Y%m%d')
        for log_file in log_files:
            log_path = str(Path(log_dir) / log_file)
            if log_file.startswith(date_str) and log_path not in (QdsPyLogFile() & exp_key).to_arrays('log_path'):
                QdsPyLogFile().add1(key=exp_key, log_path=log_path)

QdsPyLog().populate(display_progress=True)

# %%
ExpHistoryStimLog().populate(display_progress=True)
ExpHistoryStimLog()

# %% [markdown]
# `PresentationLog`: the QDSpy log entry of each `Presentation` (matched by the QDSpy md5 or file name in
# `stim_dict['stim_hash_list']` / `stim_dict['stim_name_list']` of the stimulus). Used e.g. to identify moving bar and
# mouse cam versions.

# %%
PresentationLog().populate(make_kwargs=dict(verbose=0, skip_if_no_log=True), suppress_errors=True,
                           display_progress=True)
print(f"{len(PresentationLog())} of {len(Presentation())} presentations have a log entry")

# %% [markdown]
# ## Light exposure history report
#
# Per experiment, every laser scan and every stimulus in time order: where and how long the retina was exposed, which
# stimulus (unlisted stimuli only by name, never with results), setup and stimulator, and the processing status.
# Unknown values are written as `unknown`, gaps are listed at the top of the report. Writes CSV, JSON and HTML to one
# folder per experiment. Run again at the end of the pipeline to include traces and motion QC.

# %%
from djimaging.utils import exposure_history as eh
from djimaging.utils.exposure_history_dj import fetch_exposure_history

exposure_dir = os.path.expanduser('~/exposure_history')  # One sub-folder per experiment
exposure_tables = dict(
    stimulus_table=Stimulus, exp_history_field_table=ExpHistoryField, stim_log_table=ExpHistoryStimLog,
    presentation_table=Presentation, traces_table=Traces,
    # Optional tables (None if the block is not part of the schema)
    motion_detection_table=globals().get('MotionDetection'), motion_trust_table=globals().get('MotionTrust'),
    experiment_stimulator_table=globals().get('ExperimentStimulator'),
    experiment_imaging_setup_table=globals().get('ExperimentImagingSetup'),
)

for exp_key in ExpHistory().proj().to_dicts():
    events, meta = fetch_exposure_history(exp_key, ExpHistory, **exposure_tables)
    files = eh.export_experiment(os.path.join(exposure_dir, eh.experiment_folder_name(exp_key)), events, meta)
print(f"Last report: {files['html']}")
