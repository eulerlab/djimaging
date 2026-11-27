from djimaging.schemas.blocks.core import *
from djimaging.tables import stimulator


# Setup tables, assigned per experiment from setupid and date. make_project inserts them before Field and binds them
# in Field, Presentation and HighRes, which then take stimulator delay, scan width and light artifact rows from here.
@schema
class Stimulator(stimulator.StimulatorTemplate):
    pass


@schema
class ExperimentStimulator(stimulator.ExperimentStimulatorTemplate):
    experiment_table = Experiment
    stimulator_table = Stimulator


@schema
class ImagingSetup(stimulator.ImagingSetupTemplate):
    pass


@schema
class ExperimentImagingSetup(stimulator.ExperimentImagingSetupTemplate):
    experiment_table = Experiment
    imaging_setup_table = ImagingSetup
