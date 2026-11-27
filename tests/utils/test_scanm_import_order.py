"""Every scanm module must be importable first, in a fresh interpreter (the modules import each other in a cycle)."""
import pkgutil
import subprocess
import sys

import pytest

import djimaging.utils.scanm

MODULES = sorted(m.name for m in pkgutil.iter_modules(djimaging.utils.scanm.__path__))


@pytest.mark.parametrize('module', MODULES)
def test_scanm_import_order(module):
    result = subprocess.run([sys.executable, '-W', 'ignore', '-c', f'import djimaging.utils.scanm.{module}'],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
