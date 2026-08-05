from pathlib import Path

import datajoint as dj
import numpy as np

from djimaging.utils.dj_storage import load_array, local_path, open_object, relative_store_path


def test_load_array_preserves_numpy_array():
    array = np.arange(5)
    assert load_array(array) is array


def test_load_array_materializes_npy_ref():
    array = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
    ref = dj.NpyRef(
        {"shape": list(array.shape), "dtype": str(array.dtype), "path": "unused.npy"},
        backend=object(),
    )
    ref._cached = array

    loaded = load_array(ref)

    assert loaded is array
    np.testing.assert_array_equal(loaded.squeeze().copy().astype(np.float64), array.squeeze())


def test_local_path_and_open_object_support_regular_files(tmp_path: Path):
    path = tmp_path / "object.txt"
    path.write_text("payload")

    with local_path(path) as resolved:
        assert resolved == path

    with open_object(path, mode="r") as file:
        assert file.read() == "payload"


def test_relative_store_path_rejects_paths_outside_store(tmp_path: Path):
    import pytest

    store = tmp_path / "store"
    store.mkdir()
    with dj.config.override(
        stores={"default": "test", "test": {"protocol": "file", "location": str(store)}}
    ):
        assert relative_store_path(store / "folder" / "file.bin", "test") == "folder/file.bin"
        with pytest.raises(ValueError, match="outside DataJoint store"):
            relative_store_path(tmp_path / "outside.bin", "test")
