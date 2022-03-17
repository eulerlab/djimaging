from djimaging.tables.optional import location, chirp, orientation

from .utils import _test_definition


# location
def test_definition_relativefieldlocation():
    _test_definition(location.RelativeFieldLocationTemplate)


def test_definition_retinalfieldlocation():
    _test_definition(location.RetinalFieldLocationTemplate)


# chirp
def test_definition_chirpqi():
    _test_definition(chirp.ChirpQITemplate)


# orientation
def test_definition_osdsindexestemplate():
    _test_definition(orientation.OsDsIndexesTemplate)
