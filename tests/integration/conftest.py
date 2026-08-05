import os
import uuid

import datajoint as dj
import pytest


@pytest.fixture(scope="session")
def tutorial_schema():
    dj.config["database.host"] = os.environ.get("DJ_HOST", "127.0.0.1")
    dj.config["database.port"] = int(os.environ.get("DJ_PORT", "3306"))
    dj.config["database.user"] = os.environ.get("DJ_USER", "root")
    dj.config["database.password"] = os.environ.get("DJ_PASS", "datajoint")

    connection = dj.conn()
    schema_name = f"djimaging_ci_{uuid.uuid4().hex[:8]}"

    from djimaging.schemas import tutorial_schema as schema_module
    from djimaging.utils.dj_utils import activate_schema

    try:
        activate_schema(schema_module.schema, schema_name=schema_name)
        yield schema_module
    finally:
        connection.query(f"DROP DATABASE IF EXISTS `{schema_name}`")
