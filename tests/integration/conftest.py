import os
import uuid
from pathlib import Path

import datajoint as dj
import pytest


@pytest.fixture(scope="session")
def dj_test_stores(tmp_path_factory: pytest.TempPathFactory, tutorial_data_dir: Path) -> dict[str, Path]:
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
def tutorial_schema(dj_test_stores):
    dj.config["database.host"] = os.environ.get("DJ_HOST", "127.0.0.1")
    dj.config["database.port"] = int(os.environ.get("DJ_PORT", "3306"))
    dj.config["database.user"] = os.environ.get("DJ_USER", "root")
    dj.config["database.password"] = os.environ.get("DJ_PASS", "datajoint")

    connection = dj.conn()
    schema_name = f"djimaging_ci_{uuid.uuid4().hex[:8]}"

    from djimaging.schemas import tutorial_schema as schema_module
    from djimaging.utils.dj_utils import activate_schema

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

    try:
        activate_schema(schema_module.schema, schema_name=schema_name)
        yield schema_module
    finally:
        connection.query(f"DROP DATABASE IF EXISTS `{schema_name}`")
