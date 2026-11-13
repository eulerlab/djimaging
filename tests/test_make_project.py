import ast
import builtins
import importlib.util
import itertools
import json
import shlex
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
    # The classifier reads its moving bar features from OsDsIndexes (basic)
    assert make_project.resolve_blocks(['classifier']) == ['core', 'basic', 'classifier']


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
    assert ('MotionRoiMask' in namespace) == ('motion' in blocks)
    assert ('ExpHistory' in namespace) == ('history' in blocks)
    assert ('ExpHistoryStimLog' in namespace) == ('history' in blocks)
    assert ('RefinedRoiMask' in namespace) == ('motion_refine' in blocks)
    # Motion mode: one presentation mask format only
    assert (namespace['RoiMask'].RoiMaskPresentation is None) == ('motion' in blocks)
    assert namespace['RoiMask']._presentation_masks == ('motion' not in blocks)
    # Motion mode: the presentation of the field mask is chosen by fit and the mask aligned to it
    assert namespace['RoiMask']()._selects_field_mask == ('motion' in blocks)
    assert ('mask_selection' in namespace['RoiMask']().definition) == ('motion' in blocks)
    assert namespace['Traces']._include_motion_correction == ('motion' in blocks)


@pytest.mark.parametrize('channels', [None, (1, 1), (2, 1)])
def test_motion_mode_schema_and_notebook(channels):
    blocks = make_project.resolve_blocks(['motion', 'rf_sta'])
    text = make_project.build_schema_text(ROOT, blocks, channels=channels)
    namespace = exec_schema(text)
    assert namespace['Traces'].motion_roi_mask_table is namespace['MotionRoiMask']
    assert text.index('class MotionRoiMask(') < text.index('class Traces(')
    assert text.index('class Roi(core.RoiTemplate)') < text.index('class MotionRoiMask(')
    assert namespace['MotionTrust'].Roi.motion_roi_mask_table is namespace['MotionRoiMask']
    # RefinedRoiMask is optional (motion_refine block) and off by default
    assert 'RefinedRoiMask' not in namespace
    assert namespace['MotionRoiMask'].refined_roi_mask_table is None
    assert '-> self.refined_roi_mask_table' not in namespace['MotionRoiMask']().definition
    nb = make_project.build_notebook_text(ROOT, blocks, schema_module='my.schema', project='tester', channels=channels)
    ast.parse(nb)
    assert nb.index('MotionRoiMask().populate') < nb.index('Traces().populate')
    assert nb.index('Roi().populate') < nb.index('MotionReference().populate')
    assert 'RefinedRoiMask' not in nb


@pytest.mark.parametrize('channels', [None, (1, 1)])
def test_motion_refine_schema_and_notebook(channels):
    blocks = make_project.resolve_blocks(['motion_refine'])
    assert blocks == ['core', 'motion', 'motion_refine']
    text = make_project.build_schema_text(ROOT, blocks, channels=channels)
    namespace = exec_schema(text)
    assert namespace['MotionRoiMask'].refined_roi_mask_table is namespace['RefinedRoiMask']
    assert '-> self.refined_roi_mask_table' in namespace['MotionRoiMask']().definition
    assert text.index('class MotionDetection(') < text.index('class RefinedRoiMask(') < text.index('class MotionRoiMask(')
    nb = make_project.build_notebook_text(ROOT, blocks, schema_module='my.schema', project='tester', channels=channels)
    ast.parse(nb)
    assert nb.index('MotionDetection().populate') < nb.index('RefinedRoiMask().populate') \
        < nb.index('MotionRoiMask().populate') < nb.index('Traces().populate')


def test_motion_refine_only_with_flag_or_explicit_block(monkeypatch):
    """motion_refine is not asked, not in --all, but can be forced with --refined-roi-mask or --blocks."""
    def select(*argv):
        return make_project.resolve_blocks(make_project.select_blocks(make_project.parse_args(['proj', *argv])))

    assert 'motion_refine' not in select('--all')
    assert 'motion_refine' in select('--all', '--refined-roi-mask')
    assert select('--refined-roi-mask') == ['core', 'motion', 'motion_refine']  # motion is added as requirement
    assert select('--blocks', 'basic', '--refined-roi-mask') == ['core', 'motion', 'motion_refine', 'basic']
    assert 'motion_refine' in select('--blocks', 'motion_refine')

    asked = []
    monkeypatch.setattr(make_project.sys.stdin, 'isatty', lambda: True)
    monkeypatch.setattr('builtins.input', lambda q: asked.append(q) or 'n')
    assert select() == ['core']
    assert not any("'motion_refine'" in q for q in asked) and any("'motion'" in q for q in asked)


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
    project_dir = tmp_path / 'tester'
    make_project.create_project(ROOT, str(project_dir), 'tester', make_project.resolve_blocks(['rf_glm']))

    nb = jupytext.read(project_dir / 'dji_tester' / 'notebooks' / '04_rf_glm.ipynb')
    assert nb.metadata['kernelspec'] == dict(display_name='tester', language='python', name='tester')
    static = json.loads((project_dir / 'dji_tester' / 'notebooks' / 'connect_to_schema.ipynb').read_text('utf8'))
    assert static['metadata']['kernelspec']['name'] == 'tester'
    code = "\n".join(c.source for c in nb.cells if c.cell_type == 'code')
    assert 'from dji_tester.schemas.tester_schema import *' in code
    assert 'from dji_tester import settings' in code
    assert 'RfGlm()' in code


@pytest.mark.parametrize('project', ['joe', 'Joe_1', '_x'])
def test_validate_project_name_accepts(project):
    make_project.validate_project_name(project)


@pytest.mark.parametrize('project', ['', 'a-b', 'a.b', 'a b', '1abc', 'jöe', 'class', 'a/b'])
def test_validate_project_name_rejects(project):
    with pytest.raises(AssertionError):
        make_project.validate_project_name(project)


def test_notebook_schema_name_from_settings():
    """All notebooks take the schema name from the settings of the project, which default to the project name."""
    notebooks = make_project.build_notebooks(ROOT, ['core'], project='tester')
    for text in notebooks.values():
        assert 'from dji_tester import settings' in text
        assert 'schema_name=settings.SCHEMA_NAME' in text
    namespace = {}
    exec(make_project.build_settings('tester'), namespace)
    assert namespace['SCHEMA_NAME'].startswith('ageuler_') and namespace['SCHEMA_NAME'].endswith('_tester')
    assert namespace['CONFIG_FILE'].endswith('_conf.json')


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
    assert text.index('Channel().add_channels') > text.index('RawDataParams().add_params(')


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


def test_raw_params_in_notebook():
    blocks = make_project.resolve_blocks(['core'])
    text = make_project.build_notebook_text(
        ROOT, blocks, schema_module='my.schema', project='tester',
        raw_params=dict(from_raw_data=True, igor_roi_masks='either'))
    ast.parse(text)
    assert "from_raw_data = True\nigor_roi_masks = 'either'\n" in text
    assert 'RawDataParams().add_params(from_raw_data=from_raw_data, igor_roi_masks=igor_roi_masks)' in text
    assert 'RawDataParams().add_default' not in text and '__FROM_RAW_DATA__' not in text and '__IGOR_ROI_MASKS__' not in text


def test_raw_params_arguments(monkeypatch):
    monkeypatch.setattr('sys.stdin.isatty', lambda: False, raising=False)
    args = make_project.parse_args(['proj', '--from-raw-data', 'yes', '--igor-roi-masks', 'no'])
    assert make_project.select_raw_params(args) == dict(from_raw_data=True, igor_roi_masks='no')
    args = make_project.parse_args(['proj'])
    assert make_project.select_raw_params(args) == make_project.DEFAULT_RAW_PARAMS


def test_add_default_raw_data_params_is_deprecated():
    from djimaging.tables.core.raw_data import RawDataParamsTemplate

    calls = []

    class Params(RawDataParamsTemplate):
        def add_params(self, **kwargs):
            calls.append(kwargs)

    with pytest.warns(DeprecationWarning, match='add_default'):
        Params.add_default(Params.__new__(Params))
    assert calls[0]['igor_roi_masks'] == 'yes'


@pytest.mark.parametrize('extra', [[], ['history']])
def test_setup_tables_before_field(extra):
    """The stimulator block is inserted before Field; Field, Presentation and HighRes take the setup values from it."""
    blocks = make_project.resolve_blocks(['stimulator', 'mouse_cam', 'mb_direction'] + extra)
    text = make_project.build_schema_text(ROOT, blocks)
    namespace = exec_schema(text)
    assert text.index('class ExperimentImagingSetup(') < text.index('class Field(core.FieldTemplate)')
    for name in ('Field', 'Presentation', 'HighRes') + (('ExpHistory',) if 'history' in extra else ()):
        assert namespace[name].experiment_stimulator_table is namespace['ExperimentStimulator']
        assert namespace[name].experiment_imaging_setup_table is namespace['ExperimentImagingSetup']
    log = namespace['PresentationLog'] if 'history' in extra else None
    assert namespace['MbDirectionMap'].presentation_log_table is log
    assert namespace['MouseCamGeometry'].presentation_log_table is log
    nb = make_project.build_notebook_text(ROOT, blocks, schema_module='my.schema', project='tester')
    ast.parse(nb)
    assert nb.index('Experiment().rescan_filesystem') < nb.index('ExperimentImagingSetup().populate') \
        < nb.index('Field().rescan_filesystem')
    if 'history' in extra:
        assert nb.index('Presentation().populate') < nb.index('PresentationLog().populate') \
            < nb.index('MbDirectionMap().populate')


def test_setup_tables_not_bound_without_stimulator_block():
    namespace = exec_schema(make_project.build_schema_text(ROOT, make_project.resolve_blocks([])))
    assert namespace['Field'].experiment_stimulator_table is None
    assert namespace['Presentation'].experiment_imaging_setup_table is None


@pytest.mark.parametrize('extra', [[], ['motion']])
def test_stimulator_blocks(extra):
    """rf_offset and mb_direction bind the stimulator; RfRoiOffset uses MotionRoiMask in motion mode."""
    blocks = make_project.resolve_blocks(['rf_offset', 'mb_direction'] + extra)
    assert {'stimulator', 'location', 'rf_sta', 'basic'} <= set(blocks)
    namespace = exec_schema(make_project.build_schema_text(ROOT, blocks))
    assert namespace['ExperimentStimulator'].stimulator_table is namespace['Stimulator']
    assert namespace['RfRoiOffset'].experiment_stimulator_tab is namespace['ExperimentStimulator']
    assert namespace['MbDirectionMap'].experiment_stimulator_table is namespace['ExperimentStimulator']
    assert namespace['MbRetinalPrefDir'].retinal_field_location_table is namespace['RetinalFieldLocation']
    expected_mask = namespace['MotionRoiMask'] if 'motion' in extra else namespace['RoiMask'].RoiMaskPresentation
    assert namespace['RfRoiOffset'].roimask_tab is expected_mask


def test_mb_direction_without_location():
    blocks = make_project.resolve_blocks(['mb_direction'])
    assert 'location' not in blocks
    namespace = exec_schema(make_project.build_schema_text(ROOT, blocks))
    assert namespace['MbRetinalPrefDir'].retinal_field_location_table is None


# ----------------------------------------------------------------------------------------------------------------------
# Standalone project
# ----------------------------------------------------------------------------------------------------------------------
def undefined_names(text, provided=()):
    """Names that a notebook loads but neither defines, imports nor gets from `provided` (the star import of the
    schema) or the builtins. Notebooks must not depend on each other."""
    tree = ast.parse(text)
    defined = set(provided) | set(dir(builtins)) | {'display'}  # display: IPython
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            defined.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.arg):
            defined.add(node.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            defined.update((alias.asname or alias.name).split('.')[0] for alias in node.names)
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    return used - defined


@pytest.mark.parametrize('blocks', [['core'], ['motion', 'basic', 'classifier', 'rf_sta'], OPTIONAL_BLOCKS])
def test_notebooks_are_numbered_and_self_contained(blocks):
    blocks = make_project.resolve_blocks(blocks)
    namespace = exec_schema(make_project.build_schema_text(ROOT, blocks))
    notebooks = make_project.build_notebooks(ROOT, blocks, project='tester')

    own = [b for b in blocks if make_project.BLOCKS[b].get('notebook', 'own') == 'own']
    expected = ['01_setup', '02_roi_masks', '03_traces'] + [f'{i:02d}_{b}' for i, b in enumerate(own, 4)]
    assert list(notebooks) == expected + ['drop_schema']

    for stem, text in notebooks.items():
        assert undefined_names(text, provided=namespace) == set(), stem
        for placeholder in ('__SCHEMA_MODULE__', '__PACKAGE__', '__PROJECT__', '__BLOCKS__', '__NOTEBOOK_TITLE__',
                            '__NOTEBOOK_LIST__', '__CREATE_SCHEMA__', '__FROM_RAW_DATA__', '__IGOR_ROI_MASKS__'):
            assert placeholder not in text, (stem, placeholder)
        # Only the first notebook creates the schema
        assert ('create_schema=True' in text) == (stem == '01_setup'), stem
        # The list of notebooks is in every notebook
        assert all(f'`{s}`' in text for s in expected), stem

    assert 'DROP_SCHEMA' in notebooks['drop_schema'] and all(
        'DROP_SCHEMA' not in text for stem, text in notebooks.items() if stem != 'drop_schema')


def test_notebooks_place_blocks():
    blocks = make_project.resolve_blocks(['motion_refine', 'history', 'stimulator', 'rf_sta'])
    notebooks = make_project.build_notebooks(ROOT, blocks, project='tester')
    setup, rois, traces = notebooks['01_setup'], notebooks['02_roi_masks'], notebooks['03_traces']
    assert setup.index('Channel().add_channels') > setup.index('RawDataParams().add_params(')
    assert setup.index('ExperimentImagingSetup().populate') < setup.index('Field().rescan_filesystem')
    assert setup.index('Presentation().populate') < setup.index('ExpHistory().populate')
    assert 'Roi().populate' in rois and 'Traces().populate' not in rois
    assert traces.index('MotionDetection().populate') < traces.index('RefinedRoiMask().populate') \
        < traces.index('MotionRoiMask().populate') < traces.index('Traces().populate')
    assert 'Motion correction, traces and responses' in traces
    assert 'FastSta().populate' in notebooks['04_rf_sta']
    assert 'dj.Diagram(schema)' in setup and 'dj.Diagram(schema)' not in rois


def test_notebook_title_is_first_heading():
    assert make_project.notebook_title("# %% [markdown]\n# # Receptive fields: STA\n#\n# ## Sub\n") == \
        'Receptive fields: STA'


def test_default_project_dir_is_next_to_the_repo():
    project_dir = Path(make_project.default_project_dir(ROOT, 'tester'))
    assert project_dir == ROOT.parent / 'djimaging-projects' / 'tester'


def test_make_command_recreates_the_selection():
    blocks = make_project.resolve_blocks(['motion', 'classifier'])
    raw_params = dict(from_raw_data=True, igor_roi_masks='either')
    command = make_project.make_command('tester', blocks, (2, 0), raw_params, target_dir='some dir/tester')
    argv = shlex.split(command)
    assert argv[:2] == ['python', 'make_project.py']
    args = make_project.parse_args(argv[2:])
    assert args.project == 'tester' and args.target_dir == 'some dir/tester'
    assert make_project.resolve_blocks(make_project.select_blocks(args)) == blocks
    assert make_project.select_channels(args) == (2, 0)
    assert make_project.select_raw_params(args) == raw_params

    legacy = make_project.parse_args(shlex.split(make_project.make_command('tester', ['core'], None))[2:])
    assert legacy.legacy_userinfo_stacks and make_project.select_blocks(legacy) == ['core']


def test_create_project(tmp_path):
    pytest.importorskip('jupytext')
    tomllib = pytest.importorskip('tomllib')  # Python >= 3.11
    project_dir = tmp_path / 'projects' / 'tester'
    blocks = make_project.resolve_blocks(['motion', 'basic', 'classifier', 'rf_sta', 'rf_glm'])
    command = make_project.make_command('tester', blocks)
    created, skipped = make_project.create_project(ROOT, str(project_dir), 'tester', blocks, command=command)
    assert skipped == []

    pkg = project_dir / 'dji_tester'
    for relpath in ('README.md', 'pyproject.toml', '.gitignore', '.gitattributes', '.flake8', 'tests/test_schema.py',
                    'dji_tester/__init__.py', 'dji_tester/settings.py', 'dji_tester/schemas/__init__.py',
                    'dji_tester/schemas/tester_schema.py', 'dji_tester/tables/__init__.py',
                    'dji_tester/tables/my_table.py'):
        assert (project_dir / relpath).is_file(), relpath
    notebooks = sorted(p.name for p in (pkg / 'notebooks').iterdir())
    assert notebooks == ['01_setup.ipynb', '02_roi_masks.ipynb', '03_traces.ipynb', '04_basic.ipynb',
                         '05_classifier.ipynb', '06_rf_sta.ipynb', '07_rf_glm.ipynb', 'connect_to_schema.ipynb',
                         'drop_schema.ipynb', 'scan_for_corrupted_recordings.ipynb']
    for nb in notebooks:
        json.loads((pkg / 'notebooks' / nb).read_text(encoding='utf8'))

    # Installable package that depends on the local djimaging repo
    pyproject = tomllib.loads((project_dir / 'pyproject.toml').read_text(encoding='utf8'))
    assert pyproject['project']['name'] == 'dji_tester'
    assert pyproject['project']['dependencies'] == ['djimaging[autorois,rf]']
    assert pyproject['tool']['setuptools']['packages']['find']['include'] == ['dji_tester*']
    source = pyproject['tool']['uv']['sources']['djimaging']
    assert source['editable'] and (project_dir / source['path']).resolve() == ROOT.resolve()
    assert any('fetch' in f for f in pyproject['tool']['pytest']['ini_options']['filterwarnings'])

    # The README records the command, blocks and notebooks
    readme = (project_dir / 'README.md').read_text(encoding='utf8')
    assert make_project.format_command(command) in readme
    assert '| `05_classifier` | RGC classifier |' in readme and '| `motion` |' in readme

    # The schema works, its header records the command
    schema_text = (pkg / 'schemas' / 'tester_schema.py').read_text(encoding='utf8')
    assert make_project.format_command(command, prefix='#   ') in schema_text  # Short command: no noqa
    namespace = exec_schema(schema_text)
    assert namespace['Traces']._include_motion_correction and 'CelltypeAssignmentV2' in namespace
    exec_schema((pkg / 'tables' / 'my_table.py').read_text(encoding='utf8'))

    # Static notebooks use the schema name of the settings
    connect = json.loads((pkg / 'notebooks' / 'connect_to_schema.ipynb').read_text(encoding='utf8'))
    source = "".join("".join(c['source']) for c in connect['cells'])
    assert 'schema_name = settings.SCHEMA_NAME' in source and 'ENTER_SCHEMA_NAME_HERE' not in source

    gitignore = (project_dir / '.gitignore').read_text(encoding='utf8')
    assert '.ipynb_checkpoints/' in gitignore and 'dj_*conf*.json' in gitignore


def test_create_project_never_overwrites(tmp_path):
    project_dir = tmp_path / 'tester'
    (project_dir / 'dji_tester').mkdir(parents=True)
    (project_dir / 'README.md').write_text('my readme', encoding='utf8')
    (project_dir / 'dji_tester' / 'settings.py').write_text('SCHEMA_NAME = "mine"\n', encoding='utf8')
    created, skipped = make_project.create_project(ROOT, str(project_dir), 'tester', ['core'])
    assert set(skipped) == {'README.md', 'dji_tester/settings.py'}
    assert (project_dir / 'README.md').read_text(encoding='utf8') == 'my readme'
    assert (project_dir / 'dji_tester' / 'settings.py').read_text(encoding='utf8') == 'SCHEMA_NAME = "mine"\n'

    created_again, skipped_again = make_project.create_project(ROOT, str(project_dir), 'tester', ['core'])
    assert created_again == [] and set(skipped_again) == set(created) | set(skipped)


def test_create_project_rejects_folder_inside_repo():
    with pytest.raises(AssertionError, match='outside the djimaging repo'):
        make_project.create_project(ROOT, str(ROOT / 'djimaging' / 'user' / 'tester'), 'tester', ['core'])


def test_main_creates_project(tmp_path, monkeypatch):
    monkeypatch.setattr(make_project.sys.stdin, 'isatty', lambda: False)
    project_dir = tmp_path / 'tester'
    make_project.main(['tester', '--blocks', 'location', '--target-dir', str(project_dir)])
    assert (project_dir / 'dji_tester' / 'schemas' / 'tester_schema.py').is_file()
    readme = (project_dir / 'README.md').read_text(encoding='utf8')
    assert ' \\\n    --blocks location \\\n    --n-functional 1 \\\n    --n-background 1 \\\n    --from-raw-data no ' \
           '\\\n    --igor-roi-masks yes \\\n    --target-dir ' in readme


def test_format_command_is_the_same_shell_command():
    command = make_project.make_command('tester', ['core', 'basic'], target_dir='a b/--c')
    lines = make_project.format_command(command, prefix='#   ').splitlines()
    assert all(line.startswith('#   ') for line in lines) and len(lines) == 7
    assert shlex.split(" ".join(line[4:].rstrip('\\') for line in lines)) == shlex.split(command)


def test_merge_imports():
    lines = ['import datajoint as dj', 'from djimaging.tables import core, misc', 'from djimaging.tables import core, history',
             'import numpy as np', 'from djimaging.tables import location']
    assert make_project.merge_imports(lines) == [
        'import datajoint as dj', 'from djimaging.tables import core, misc, history, location', 'import numpy as np']
    long = make_project.merge_imports([f'from m import name_{i}' for i in range(20)], max_length=120)
    assert len(long) == 1 and long[0].startswith('from m import (\n    name_0,\n') and long[0].endswith(',\n)')
    ast.parse(long[0])


def test_generated_schema_has_no_duplicate_imports():
    text = make_project.build_schema_text(ROOT, make_project.resolve_blocks(OPTIONAL_BLOCKS))
    modules = [n.module for n in ast.parse(text).body if isinstance(n, ast.ImportFrom)]
    assert len(modules) == len(set(modules))


def test_djimaging_source_path():
    default = make_project.default_project_dir(ROOT, 'tester')
    assert make_project.djimaging_source_path(ROOT, default) == f'../../{ROOT.name}'
    far = ROOT.parent / 'a' / 'b' / 'c' / 'd' / 'tester'
    assert make_project.djimaging_source_path(ROOT, far) == str(ROOT.resolve()).replace('\\', '/')
