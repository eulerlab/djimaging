import ast
import importlib.util
import itertools
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location('make_project', ROOT / 'make_project.py')
make_project = importlib.util.module_from_spec(spec)
spec.loader.exec_module(make_project)

OPTIONAL_BLOCKS = [b for b in make_project.BLOCKS if b != 'core']


def exec_schema(text):
    namespace = {}
    exec(compile(text, '<generated_schema>', 'exec'), namespace)
    return namespace


def test_resolve_blocks_adds_requirements_and_orders():
    assert make_project.resolve_blocks([]) == ['core']
    assert make_project.resolve_blocks(['rf_glm', 'basic', 'basic']) == ['core', 'basic', 'rf_glm']


def test_resolve_blocks_rejects_unknown():
    with pytest.raises(AssertionError):
        make_project.resolve_blocks(['does_not_exist'])


def test_all_blocks_have_files():
    for name in make_project.BLOCKS:
        assert (ROOT / 'djimaging' / 'schemas' / 'blocks' / f'{name}.py').is_file()
        assert (ROOT / 'djimaging' / 'notebooks' / 'blocks' / f'{name}.py').is_file()
    assert (ROOT / 'djimaging' / 'notebooks' / 'blocks' / 'cleanup.py').is_file()


@pytest.mark.parametrize('blocks', [list(c) for r in (0, 1, 2) for c in itertools.combinations(OPTIONAL_BLOCKS, r)])
def test_generated_schema_defines_tables_of_blocks(blocks):
    blocks = make_project.resolve_blocks(blocks)
    namespace = exec_schema(make_project.build_schema_text(ROOT, blocks))
    assert 'schema' in namespace and 'UserInfo' in namespace
    assert ('FastSta' in namespace) == ('rf_sta' in blocks)
    assert ('RfGlm' in namespace) == ('rf_glm' in blocks)
    assert ('OsDsIndexes' in namespace) == ('basic' in blocks)
    assert ('OpticDisk' in namespace) == ('location' in blocks)
    assert ('CelltypeAssignmentV2' in namespace) == ('classifier' in blocks)


def test_generated_schema_has_no_duplicate_tables():
    text = make_project.build_schema_text(ROOT, make_project.resolve_blocks(OPTIONAL_BLOCKS))
    names = [n.name for n in ast.parse(text).body if isinstance(n, ast.ClassDef)]
    assert len(names) == len(set(names))


def test_notebook_contains_only_selected_tables():
    blocks = make_project.resolve_blocks(['rf_sta'])
    text = make_project.build_notebook_text(ROOT, blocks, schema_module='my.schema', project='tester')
    ast.parse(text)
    assert 'from my.schema import *' in text
    assert 'FastSta()' in text
    assert 'RfGlm()' not in text and 'OpticDisk()' not in text
    assert make_project.SCHEMA_MODULE_PLACEHOLDER not in text and make_project.BLOCKS_PLACEHOLDER not in text
    assert text.rstrip().endswith('Set DROP_SCHEMA = True to drop it.")')


def test_notebook_only_uses_tables_of_schema():
    """Run-all safety: every table that is called in the notebook must exist in the schema of the same blocks."""
    for blocks in (['core'], make_project.resolve_blocks(OPTIONAL_BLOCKS)):
        namespace = exec_schema(make_project.build_schema_text(ROOT, blocks))
        text = make_project.build_notebook_text(ROOT, blocks, schema_module='my.schema', project='tester')
        table_names = {n for n, v in namespace.items() if isinstance(v, type) and hasattr(v, 'definition')}
        called = {n.func.id for n in ast.walk(ast.parse(text))
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id[:1].isupper()}
        called -= {'Path'}
        assert called <= table_names, f"Not in schema: {called - table_names}"


def test_generated_ipynb(tmp_path):
    jupytext = pytest.importorskip('jupytext')
    project_dir = tmp_path / 'user'
    (project_dir / 'notebooks').mkdir(parents=True)
    make_project.copy_populate_notebook(ROOT, str(project_dir), 'tester', make_project.resolve_blocks(['rf_glm']))

    nb = jupytext.read(project_dir / 'notebooks' / 'populate_tester_schema.ipynb')
    code = "\n".join(c.source for c in nb.cells if c.cell_type == 'code')
    assert 'djimaging.user.tester.schemas.tester_schema' in code
    assert 'RfGlm()' in code


@pytest.mark.parametrize('project', ['joe', 'Joe_1', '_x'])
def test_validate_project_name_accepts(project):
    make_project.validate_project_name(project)


@pytest.mark.parametrize('project', ['', 'a-b', 'a.b', 'a b', '1abc', 'jöe', 'class', 'a/b'])
def test_validate_project_name_rejects(project):
    with pytest.raises(AssertionError):
        make_project.validate_project_name(project)


def test_notebook_schema_name_uses_project():
    text = make_project.build_notebook_text(ROOT, ['core'], schema_module='my.schema', project='tester')
    assert 'schema_name = f"ageuler_{username}_tester"' in text


def test_default_schema_uses_channel_table_with_one_functional_channel():
    blocks = make_project.resolve_blocks(['rf_sta'])
    default = make_project.build_schema_text(ROOT, blocks)
    assert default == make_project.build_schema_text(ROOT, blocks, channels=(1, 1))
    namespace = exec_schema(default)
    assert namespace['UserInfo']._use_channels and namespace['UserInfo'].channel_table is namespace['Channel']
    assert 'data_stack_name' not in namespace['UserInfo']().definition
    assert not namespace['Traces']._use_channels  # One functional channel: no ch_id in the keys
    assert 'ch_id' not in namespace['Traces']().definition

    text = make_project.build_notebook_text(ROOT, blocks, schema_module='my.schema', project='tester')
    ast.parse(text)
    assert ("Channel().add_channels([('wDataCh0', 0), ('wDataCh1', 1)], functional=['wDataCh0'], "
            "roi_channel='wDataCh0', skip_duplicates=True)") in text
    assert text.index('Channel().add_channels') > text.index('RawDataParams().add_default()')


@pytest.mark.parametrize('blocks', [['core'], OPTIONAL_BLOCKS])
def test_legacy_schema_and_notebook_have_no_channel_table(blocks):
    blocks = make_project.resolve_blocks(blocks)
    namespace = exec_schema(make_project.build_schema_text(ROOT, blocks, channels=None))
    assert 'Channel' not in namespace
    assert not namespace['UserInfo']._use_channels and not namespace['Traces']._use_channels
    assert "data_stack_name='wDataCh0'" in namespace['UserInfo']().definition
    assert "alt_stack_name='wDataCh1'" in namespace['UserInfo']().definition

    text = make_project.build_notebook_text(ROOT, blocks, schema_module='my.schema', project='tester', channels=None)
    assert 'Channel' not in text


@pytest.mark.parametrize('blocks', [['core'], OPTIONAL_BLOCKS])
def test_two_functional_channels_add_ch_id_to_traces(blocks):
    blocks = make_project.resolve_blocks(blocks)
    namespace = exec_schema(make_project.build_schema_text(ROOT, blocks, channels=(2, 1)))
    assert namespace['Traces']._use_channels and namespace['Traces'].channel_table is namespace['Channel']
    assert '-> self.channel_table' in namespace['Traces']().definition
    assert namespace['UserInfo']._use_channels and namespace['UserInfo'].channel_table is namespace['Channel']

    text = make_project.build_notebook_text(ROOT, blocks, schema_module='my.schema', project='tester', channels=(2, 1))
    ast.parse(text)
    assert ("Channel().add_channels([('wDataCh0', 0), ('wDataCh1', 1), ('wDataCh2', 2)], "
            "functional=['wDataCh0', 'wDataCh1'], roi_channel='wDataCh0', skip_duplicates=True)") in text


def test_no_background_channel():
    assert make_project.channels_name_id((1, 0)) == [('wDataCh0', 0)]
    text = make_project.build_notebook_text(ROOT, ['core'], schema_module='my.schema', project='tester',
                                            channels=(1, 0))
    assert "add_channels([('wDataCh0', 0)], functional=['wDataCh0']" in text


@pytest.mark.parametrize('channels', [(0, 1), (1, -1), (1.5, 1), (1, '1'), (True, 1), (1, None)])
def test_invalid_channels_rejected(channels):
    with pytest.raises(AssertionError):
        make_project.validate_channels(channels)


def select(*argv):
    return make_project.select_channels(make_project.parse_args(['proj', *argv]))


def test_channel_arguments(monkeypatch):
    monkeypatch.setattr(make_project.sys.stdin, 'isatty', lambda: False)
    assert select() == (1, 1)
    assert select('--n-functional', '2', '--n-background', '0') == (2, 0)
    assert select('--n-background', '2') == (1, 2)
    assert select('--legacy-userinfo-stacks') is None
    with pytest.raises(AssertionError):
        select('--legacy-userinfo-stacks', '--n-functional', '2')


def test_channels_are_asked_interactively(monkeypatch):
    monkeypatch.setattr(make_project.sys.stdin, 'isatty', lambda: True)
    answers = iter(['2', ''])
    monkeypatch.setattr('builtins.input', lambda prompt: next(answers))
    assert select() == (2, 1)
