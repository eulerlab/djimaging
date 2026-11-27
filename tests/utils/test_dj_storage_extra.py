"""dj_storage helpers with fake ObjectRef backends (no database)."""
import io
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import datajoint as dj
import numpy as np
import pytest

from djimaging.utils.dj_storage import (
    file_store_path, load_array, local_file_path, local_path, open_object, relative_store_path,
)


class _RemoteBackend:
    """Minimal remote (s3) backend: serves bytes from a dict."""
    protocol = 's3'
    spec = {'bucket': 'bucket'}

    def __init__(self, files):
        self.files = files
        self.downloads = []

    def get_file(self, remote_path, destination):
        self.downloads.append(remote_path)
        Path(destination).write_bytes(self.files[remote_path])

    def open(self, path, mode):
        data = self.files[path]
        return io.BytesIO(data) if 'b' in mode else io.StringIO(data.decode())


def _ref(path, backend):
    return dj.ObjectRef(path=path, size=None, hash=None, ext=None, is_dir=False, timestamp=datetime.now(),
                        _backend=backend)


def test_local_path_downloads_remote_object_to_temporary_file():
    backend = _RemoteBackend({'folder/data.bin': b'payload'})
    ref = _ref('folder/data.bin', backend)
    with local_path(ref) as path:
        assert path.name == 'data.bin'
        assert path.read_bytes() == b'payload'
        staged = path
    assert backend.downloads == ['folder/data.bin']
    assert not staged.exists()  # temporary copy removed


def test_local_path_uses_mounted_file_without_download(tmp_path):
    (tmp_path / 'a').mkdir()
    (tmp_path / 'a' / 'x.bin').write_bytes(b'1')
    backend = SimpleNamespace(protocol='file', spec={'location': str(tmp_path)})
    with local_path(_ref('a/x.bin', backend)) as path:
        assert path == tmp_path / 'a' / 'x.bin'


def test_local_path_mounted_but_missing_file_is_downloaded(tmp_path):
    backend = _RemoteBackend({'a/y.bin': b'2'})
    backend.protocol = 'file'
    backend.spec = {'location': str(tmp_path)}
    with local_path(_ref('a/y.bin', backend)) as path:
        assert path.read_bytes() == b'2'
        assert path != tmp_path / 'a' / 'y.bin'


def test_local_file_path_rejects_remote_object():
    ref = _ref('folder/data.bin', _RemoteBackend({}))
    with pytest.raises(ValueError, match='locally mounted'):
        local_file_path(ref)


def test_local_file_path_rejects_object_without_full_path():
    backend = SimpleNamespace(protocol='file', spec=None)  # spec.get raises AttributeError
    with pytest.raises(ValueError, match='locally mounted'):
        local_file_path(_ref('x.bin', backend))


@pytest.mark.parametrize('mode, expected', [('rb', b'payload'), ('r', 'payload')])
def test_open_object_reads_object_ref(mode, expected):
    ref = _ref('folder/data.txt', _RemoteBackend({'folder/data.txt': b'payload'}))
    with open_object(ref, mode=mode) as f:
        assert f.read() == expected


def test_open_object_relative_to_store(tmp_path):
    store = tmp_path / 'store'
    (store / 'sub').mkdir(parents=True)
    (store / 'sub' / 'f.txt').write_text('abc')
    with dj.config.override(stores={'default': 'test', 'test': {'protocol': 'file', 'location': str(store)}}):
        with open_object('sub/f.txt', mode='r', store='test') as f:
            assert f.read() == 'abc'


def test_load_array_converts_lists():
    out = load_array([[1, 2], [3, 4]])
    assert isinstance(out, np.ndarray) and out.shape == (2, 2)


def test_store_paths(tmp_path):
    store = tmp_path / 'store'
    store.mkdir()
    stores = {'default': 'test', 'test': {'protocol': 'file', 'location': str(store)},
              'remote': {'protocol': 's3', 'endpoint': 'localhost:9000', 'bucket': 'b', 'location': 'loc',
                         'access_key': 'a', 'secret_key': 's'}}
    with dj.config.override(stores=stores):
        # relative paths are kept (as posix), absolute paths inside the store are made relative
        assert relative_store_path(Path('sub') / 'f.bin', 'test') == 'sub/f.bin'
        assert relative_store_path(store / 'sub' / '..' / 'f.bin', 'test') == 'f.bin'
        with pytest.raises(ValueError, match='escapes'):
            relative_store_path(Path('..') / 'f.bin', 'test')
        with pytest.raises(ValueError, match='escapes'):
            relative_store_path('sub/../../f.bin', 'test')
        # absolute paths are returned as they are
        assert file_store_path(tmp_path / 'abs.bin', 'test') == tmp_path / 'abs.bin'
        assert local_file_path(tmp_path / 'abs.bin') == tmp_path / 'abs.bin'
        # remote store
        assert relative_store_path('sub/f.bin', 'remote') == 'sub/f.bin'
        with pytest.raises(ValueError, match='remote store'):
            relative_store_path(tmp_path / 'f.bin', 'remote')
        with pytest.raises(ValueError, match='not a local file store'):
            file_store_path('sub/f.bin', 'remote')
