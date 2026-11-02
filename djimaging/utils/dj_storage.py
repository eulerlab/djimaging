"""Helpers for DataJoint 2 object and array references."""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Iterator, TextIO

import datajoint as dj
import numpy as np


@contextmanager
def open_object(
        value: str | Path | dj.ObjectRef,
        mode: str = "rb",
        store: str | None = None,
) -> Iterator[BinaryIO | TextIO]:
    """Open a local path or a DataJoint ``ObjectRef`` as a file-like object."""
    if isinstance(value, dj.ObjectRef):
        with value.open(mode=mode) as file:
            yield file
    else:
        path = file_store_path(value, store) if store is not None else Path(value)
        with path.open(mode=mode) as file:
            yield file


@contextmanager
def local_path(
        value: str | Path | dj.ObjectRef,
        store: str | None = None,
) -> Iterator[Path]:
    """Yield a local path, staging a remote ``ObjectRef`` when necessary."""
    if not isinstance(value, dj.ObjectRef):
        yield file_store_path(value, store) if store is not None else Path(value)
        return

    try:
        path = Path(value.full_path)
    except (AttributeError, TypeError, ValueError):
        path = None

    if path is not None and path.exists():
        yield path
        return

    with tempfile.TemporaryDirectory(prefix="djimaging-object-") as directory:
        destination = Path(directory) / Path(value.path).name
        yield value.download(destination)


def local_file_path(value: str | Path | dj.ObjectRef, store: str | None = None) -> Path:
    """Return the mounted path for a local-file reference without staging it."""
    if not isinstance(value, dj.ObjectRef):
        return file_store_path(value, store) if store is not None else Path(value)
    try:
        full_path = value.full_path
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("This operation requires a locally mounted filepath store") from error
    if "://" in full_path:
        raise ValueError("This operation requires a locally mounted filepath store")
    return Path(full_path)


def load_array(value: np.ndarray | dj.NpyRef) -> np.ndarray:
    """Materialize a DataJoint ``NpyRef`` while leaving arrays unchanged."""
    return value.load() if isinstance(value, dj.NpyRef) else np.asarray(value)


def relative_store_path(value: str | Path, store: str) -> str:
    """Return a safe path relative to a configured file store."""
    path = Path(value)
    store_spec = dj.config.get_store_spec(store)
    if path.is_absolute():
        if store_spec.get("protocol") != "file":
            raise ValueError(f"Absolute paths cannot be inserted into remote store {store!r}")
        try:
            path = path.resolve().relative_to(Path(store_spec["location"]).resolve())
        except ValueError as error:
            raise ValueError(f"Path {value!r} is outside DataJoint store {store!r}") from error
    if ".." in path.parts:
        raise ValueError(f"Path {value!r} escapes DataJoint store {store!r}")
    return path.as_posix()


def file_store_path(value: str | Path, store: str) -> Path:
    """Resolve a relative value against a local DataJoint file store."""
    path = Path(value)
    if path.is_absolute():
        return path

    store_spec = dj.config.get_store_spec(store)
    if store_spec.get("protocol") != "file":
        raise ValueError(f"Store {store!r} is not a local file store")
    return Path(store_spec["location"]) / path
