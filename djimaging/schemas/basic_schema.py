from djimaging.schemas.core_schema import schema, DetrendSnippets, Stimulus, Presentation
from djimaging.basic_tables import chirp, orientation


@schema
class ChirpQI(chirp.ChirpQITemplate):
    stimulus_table = Stimulus
    detrendsnippets_table = DetrendSnippets


@schema
class ChirpFeatures(chirp.ChirpFeaturesTemplate):
    detrendsnippets_table = DetrendSnippets
    presentation_table = Presentation


@schema
class OsDsIndexes(orientation.OsDsIndexesTemplate):
    stimulus_table = Stimulus
    detrendsnippets_table = DetrendSnippets


