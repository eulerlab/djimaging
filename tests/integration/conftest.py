"""MySQL integration tests (run with DJ_TEST_MYSQL=1).

Connection: DJ_TEST_CONFIG=<path to a DataJoint config file> (loaded with `dj.config.load`), else
DJ_HOST / DJ_PORT / DJ_USER / DJ_PASS (defaults 127.0.0.1:3306 root/datajoint).
The stores of the config are always replaced by temporary folders, so tests never write to real stores.

Schema names: DJ_TEST_SCHEMA=<prefix> gives `<prefix>_tutorial`, `<prefix>_motion`, `<prefix>_motion_refine` and
`<prefix>_stimulator`; default prefix `djimaging_ci` plus a random suffix. If a test schema already exists, the run
aborts without touching it. Only schemas created by the run are dropped afterwards.
"""
import os
from pathlib import Path

import datajoint as dj
import pytest

from tests.fixtures.dj_test_db import new_test_schema


@pytest.fixture(scope="session")
def dj_test_connection():
    config_file = os.environ.get("DJ_TEST_CONFIG")
    if config_file:
        if not os.path.isfile(config_file):
            pytest.exit(f"DJ_TEST_CONFIG={config_file} does not exist", returncode=1)
        dj.config.load(config_file)
    else:
        dj.config["database.host"] = os.environ.get("DJ_HOST", "127.0.0.1")
        dj.config["database.port"] = int(os.environ.get("DJ_PORT", "3306"))
        dj.config["database.user"] = os.environ.get("DJ_USER", "root")
        dj.config["database.password"] = os.environ.get("DJ_PASS", "datajoint")
    return dj.conn()


@pytest.fixture(scope="session")
def dj_test_stores(dj_test_connection, tmp_path_factory: pytest.TempPathFactory,
                   tutorial_data_dir: Path) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("datajoint-v2-stores")
    stores = {
        "reference": tutorial_data_dir,
        "processed": root / "processed",
    }
    for path in stores.values():
        path.mkdir(exist_ok=True)

    dj.config["stores"] = {
        "default": "processed",
        "filepath_default": "reference",
        **{name: {"protocol": "file", "location": str(path)} for name, path in stores.items()},
    }
    download_path = root / "downloads"
    download_path.mkdir()
    dj.config["download_path"] = str(download_path)
    return stores


@pytest.fixture(scope="session")
def tutorial_schema(dj_test_connection, dj_test_stores):
    from djimaging.schemas import tutorial_schema as schema_module

    @schema_module.schema
    class CodecRoundTrip(dj.Manual):
        definition = """
        object_id : int32
        ---
        source_file : <filepath@reference>
        attachment : <attach@processed>
        small_object : <blob>
        large_object : <blob@processed>
        array : <npy@processed>
        """

    schema_module.CodecRoundTrip = CodecRoundTrip

    with new_test_schema(dj_test_connection, schema_module.schema, "tutorial"):
        yield schema_module
