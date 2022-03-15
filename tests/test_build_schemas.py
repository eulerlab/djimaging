import datajoint as dj


def test_build_core_schema():
    from djimaging.schemas.core_schema import schema
    assert isinstance(schema, dj.schemas.Schema)


def test_build_basic_schema():
    from djimaging.schemas.basic_schema import schema
    assert isinstance(schema, dj.schemas.Schema)
