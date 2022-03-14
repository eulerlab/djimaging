from djimaging.basic_tables import chirp
from djimaging.basic_tables import orientation

from .utils import _test_definition


# chirp
def test_definition_chirpqi():
    _test_definition(chirp.ChirpQITemplate)


# orientation
def test_definition_osdsindexestemplate():
    _test_definition(orientation.OsDsIndexesTemplate)





