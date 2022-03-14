from djimaging.schemas.core_schema import schema, Stimulus, DetrendSnippets
from djimaging.basic_tables import chirp, orientation


@schema
class ChirpQI(chirp.ChirpQITemplate):
    stimulus_table = Stimulus
    detrendsnippets_table = DetrendSnippets


@schema
class OsDsIndexes(orientation.OsDsIndexesTemplate):
    stimulus_table = Stimulus
    detrendsnippets_table = DetrendSnippets
