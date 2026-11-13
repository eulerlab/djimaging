"""Names, creation and cleanup of the MySQL test schemas (see tests/integration/conftest.py)."""
import os
import uuid
from contextlib import contextmanager

import pytest


def test_schema_name(suffix: str) -> str:
    """Name of a test schema: `<DJ_TEST_SCHEMA>_<suffix>`, or random if DJ_TEST_SCHEMA is not set."""
    prefix = os.environ.get("DJ_TEST_SCHEMA")
    if prefix:
        return f"{prefix}_{suffix}"
    return f"djimaging_ci_{suffix}_{uuid.uuid4().hex[:8]}"


test_schema_name.__test__ = False  # Not a test


@contextmanager
def new_test_schema(connection, schema, suffix: str):
    """Activate `schema` under a new test schema name. Aborts the test run if the schema exists already; drops it at
    the end only if it was created here."""
    from djimaging.utils.dj_utils import activate_schema

    schema_name = test_schema_name(suffix)
    exists = connection.query(
        "SELECT COUNT(*) FROM information_schema.schemata WHERE schema_name = %s", args=(schema_name,)).fetchone()[0]
    if exists:
        pytest.exit(f"Test schema `{schema_name}` exists already. Nothing was changed. Drop it yourself or choose "
                    f"another DJ_TEST_SCHEMA.", returncode=1)
    created = False
    try:
        activate_schema(schema, schema_name=schema_name)
        created = True
        yield schema_name
    finally:
        if created:
            connection.query(f"DROP DATABASE IF EXISTS `{schema_name}`")
