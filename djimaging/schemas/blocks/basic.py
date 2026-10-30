from djimaging.schemas.blocks.core import *
from djimaging.tables import response


# Basic response analysis
@schema
class ChirpQI(response.ChirpQITemplate):
    stimulus_table = Stimulus
    snippets_table = Snippets


@schema
class ChirpFeaturesBc(response.ChirpFeaturesBcTemplate):
    stimulus_table = Stimulus
    snippets_table = Snippets
    presentation_table = Presentation


@schema
class ChirpFeaturesRgc(response.ChirpFeaturesRgcTemplate):
    stimulus_table = Stimulus
    snippets_table = Snippets
    presentation_table = Presentation


@schema
class OsDsIndexes(response.OsDsIndexesTemplate):
    _reduced_storage = True
    _n_shuffles = 1000

    stimulus_table = Stimulus
    snippets_table = Snippets
