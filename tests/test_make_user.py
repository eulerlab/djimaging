import ast
import importlib.util
import itertools
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location('make_user', ROOT / 'make_user.py')
make_user = importlib.util.module_from_spec(spec)
spec.loader.exec_module(make_user)

OPTIONAL_BLOCKS = [b for b in make_user.BLOCKS if b != 'core']


def exec_schema(text):
    namespace = {}
    exec(compile(text, '<generated_schema>', 'exec'), namespace)
    return namespace


def test_resolve_blocks_adds_requirements_and_orders():
    assert make_user.resolve_blocks([]) == ['core']
    assert make_user.resolve_blocks(['rf_glm', 'basic', 'basic']) == ['core', 'basic', 'rf_glm']


def test_resolve_blocks_rejects_unknown():
    with pytest.raises(AssertionError):
        make_user.resolve_blocks(['does_not_exist'])


def test_all_blocks_have_files():
    for name in make_user.BLOCKS:
        assert (ROOT / 'djimaging' / 'schemas' / 'blocks' / f'{name}.py').is_file()
        assert (ROOT / 'djimaging' / 'notebooks' / 'blocks' / f'{name}.py').is_file()
    assert (ROOT / 'djimaging' / 'notebooks' / 'blocks' / 'cleanup.py').is_file()


@pytest.mark.parametrize('blocks', [list(c) for r in (0, 1, 2) for c in itertools.combinations(OPTIONAL_BLOCKS, r)])
def test_generated_schema_defines_tables_of_blocks(blocks):
    blocks = make_user.resolve_blocks(blocks)
    namespace = exec_schema(make_user.build_schema_text(ROOT, blocks))
    assert 'schema' in namespace and 'UserInfo' in namespace
    assert ('FastSta' in namespace) == ('rf_sta' in blocks)
    assert ('RfGlm' in namespace) == ('rf_glm' in blocks)
    assert ('OsDsIndexes' in namespace) == ('basic' in blocks)
    assert ('OpticDisk' in namespace) == ('location' in blocks)
    assert ('CelltypeAssignmentV2' in namespace) == ('classifier' in blocks)


def test_generated_schema_has_no_duplicate_tables():
    text = make_user.build_schema_text(ROOT, make_user.resolve_blocks(OPTIONAL_BLOCKS))
    names = [n.name for n in ast.parse(text).body if isinstance(n, ast.ClassDef)]
    assert len(names) == len(set(names))


def test_notebook_contains_only_selected_tables():
    blocks = make_user.resolve_blocks(['rf_sta'])
    text = make_user.build_notebook_text(ROOT, blocks, schema_module='my.schema')
    ast.parse(text)
    assert 'from my.schema import *' in text
    assert 'FastSta()' in text
    assert 'RfGlm()' not in text and 'OpticDisk()' not in text
    assert make_user.SCHEMA_MODULE_PLACEHOLDER not in text and make_user.BLOCKS_PLACEHOLDER not in text
    assert text.rstrip().endswith('Set DROP_SCHEMA = True to drop it.")')


def test_notebook_only_uses_tables_of_schema():
    """Run-all safety: every table that is called in the notebook must exist in the schema of the same blocks."""
    for blocks in (['core'], make_user.resolve_blocks(OPTIONAL_BLOCKS)):
        namespace = exec_schema(make_user.build_schema_text(ROOT, blocks))
        text = make_user.build_notebook_text(ROOT, blocks, schema_module='my.schema')
        table_names = {n for n, v in namespace.items() if isinstance(v, type) and hasattr(v, 'definition')}
        called = {n.func.id for n in ast.walk(ast.parse(text))
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id[:1].isupper()}
        called -= {'Path'}
        assert called <= table_names, f"Not in schema: {called - table_names}"


def test_generated_ipynb(tmp_path):
    jupytext = pytest.importorskip('jupytext')
    user_dir = tmp_path / 'user'
    (user_dir / 'notebooks').mkdir(parents=True)
    make_user.copy_tutorial_notebook(ROOT, str(user_dir), 'tester', make_user.resolve_blocks(['rf_glm']))

    nb = jupytext.read(user_dir / 'notebooks' / 'tutorial_tester_schema.ipynb')
    code = "\n".join(c.source for c in nb.cells if c.cell_type == 'code')
    assert 'djimaging.user.tester.schemas.tester_schema' in code
    assert 'RfGlm()' in code
