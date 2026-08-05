import os
from pathlib import Path

import datajoint as dj
import numpy as np
import pytest

from djimaging.utils.dj_storage import load_array, local_path, open_object
from scripts.migrate_datajoint_v2 import _verify_schema_v20


pytestmark = pytest.mark.skipif(
    os.environ.get("DJ_TEST_MYSQL") != "1",
    reason="requires the MySQL integration-test service",
)


def test_datajoint_v2_storage_round_trip_and_deletion(tutorial_schema, dj_test_stores, tmp_path: Path):
    source = dj_test_stores["acquisition"] / "raw" / "source.bin"
    source.parent.mkdir()
    source.write_bytes(b"source payload")

    attachment = tmp_path / "model.txt"
    attachment.write_text("model payload")

    array = np.arange(24, dtype=np.float32).reshape(4, 6)
    large_object = {"names": ["a", "b"], "values": np.arange(10)}
    table = tutorial_schema.CodecRoundTrip()
    table.insert1(
        {
            "object_id": 1,
            "source_file": "raw/source.bin",
            "attachment": attachment,
            "small_object": {"enabled": True},
            "large_object": large_object,
            "array": array,
        }
    )

    row = (table & {"object_id": 1}).fetch1()
    assert isinstance(row["source_file"], dj.ObjectRef)
    assert isinstance(row["array"], dj.NpyRef)
    assert row["small_object"] == {"enabled": True}
    assert row["large_object"]["names"] == large_object["names"]
    np.testing.assert_array_equal(row["large_object"]["values"], large_object["values"])
    np.testing.assert_array_equal(load_array(row["array"]), array)

    with open_object(row["source_file"]) as file:
        assert file.read() == b"source payload"
    with local_path(row["source_file"]) as path:
        assert path == source
    assert Path(row["attachment"]).read_text() == "model payload"

    npy_path = row["array"].path
    (table & {"object_id": 1}).delete(prompt=False)
    assert len(table) == 0
    assert source.exists(), "filepath data is user-managed and must not be deleted with a row"

    processed_report = dj.gc.GarbageCollector(tutorial_schema.schema, store="processed").collect(dry_run=True)
    models_report = dj.gc.GarbageCollector(tutorial_schema.schema, store="models").collect(dry_run=True)
    assert npy_path in processed_report["orphaned_schema_paths"]
    assert processed_report["hash_paths_orphaned"] >= 1
    assert models_report["hash_paths_orphaned"] >= 1


def test_schema_v2_verifier(tutorial_schema):
    result = _verify_schema_v20(tutorial_schema.schema)

    assert result["compatible"]
    assert result["blob_markers"]
    assert result["issues"] == []
