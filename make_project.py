import argparse
import ast
import keyword
import os
import re
import shutil
import sys

# Table blocks that can be combined to a personal schema and a matching tutorial notebook.
# Schema parts live in djimaging/schemas/blocks/<name>.py, notebook parts in djimaging/notebooks/blocks/<name>.py.
# The order of this dict is the order of the blocks in the generated files.
BLOCKS = {
    'core': dict(
        description='UserInfo, Experiment, Field, Stimulus, Presentation, RoiMask, Roi, Traces, Snippets, Averages',
        requires=[]),
    'motion': dict(
        description='Motion correction and motion QC (motion mode): MotionParams, MotionReference, MotionDetection, '
                    'MotionRoiMask, MotionTrust. RoiMask then has no presentation masks and Traces are motion corrected',
        requires=['core']),
    'motion_refine': dict(
        description='Optional for motion mode: RefinedRoiMask, refines the ROI borders (+-1 px) of the field mask on the '
                    'registered reference condition before MotionRoiMask (no measurable gain on tested data)',
        requires=['motion'],
        default=False),  # Not asked, not in --all: only with --refined-roi-mask or explicitly in --blocks
    'basic': dict(
        description='ChirpQI, ChirpFeatures, OsDsIndexes',
        requires=['core']),
    'location': dict(
        description='OpticDisk, RelativeFieldLocation, RetinalFieldLocation',
        requires=['core']),
    'highres': dict(
        description='HighResAlignment: shift of the HighRes scan vs. the field ROI mask, match score, rough z offset '
                    '(also populates HighRes)',
        requires=['core']),
    'classifier': dict(
        description='RGC classifier: Baden16TracesV2, ClassifierV2, CelltypeAssignmentV2',
        requires=['core']),
    'rf_sta': dict(
        description='Receptive fields from STA: FastSta, SplitRF, FitGauss2DRF, FitDoG2DRF',
        requires=['core']),
    'rf_glm': dict(
        description='Receptive fields from GLM: RfGlm, SplitRfGlm, FitGauss2DRfGlm, FitDoG2DRfGlm',
        requires=['core']),
}

SCHEMA_MODULE_PLACEHOLDER = '__SCHEMA_MODULE__'
BLOCKS_PLACEHOLDER = '__BLOCKS__'
PROJECT_PLACEHOLDER = '__PROJECT__'
NOTEBOOK_TEMPLATE_NAME = 'populate_my_schema'  # Replaced by populate_<project>_schema

# Channels (n_functional, n_background) of new schemas. None is the legacy design without Channel table.
DEFAULT_CHANNELS = (1, 1)

# Anchors in the core blocks that are extended by the Channel table.
USER_INFO_CLASS_HEADER = "@schema\nclass UserInfo(core.UserInfoTemplate):\n"
TRACES_CLASS_HEADER = "@schema\nclass Traces(core.TracesTemplate):\n"
NOTEBOOK_RAW_PARAMS_CELL = "RawDataParams()\n"
FROM_RAW_DATA_PLACEHOLDER = '__FROM_RAW_DATA__'
IGOR_ROI_MASKS_PLACEHOLDER = '__IGOR_ROI_MASKS__'

# RawDataParams of new notebooks if they are not given and not asked (e.g. without a terminal).
DEFAULT_RAW_PARAMS = dict(from_raw_data=False, igor_roi_masks='yes')
IGOR_ROI_MASKS_OPTIONS = ('yes', 'no', 'either', 'init')

# Anchors in the core blocks that are changed by the motion block (motion mode).
ROI_MASK_PRESENTATION_CLASS = ("    class RoiMaskPresentation(core.RoiMaskTemplate.RoiMaskPresentation):\n"
                               "        presentation_table = Presentation\n")
NOTEBOOK_TRACES_SECTION = "# %% [markdown]\n# # Core: Traces and responses\n"

# Anchors in the motion block that are changed by the motion_refine block.
MOTION_ROI_MASK_CLASS_HEADER = "@schema\nclass MotionRoiMask(motion_correction.MotionRoiMaskTemplate):\n"
NOTEBOOK_MOTION_ROI_MASK_CELL = "# %%\nMotionRoiMask().populate("


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Create a project folder with a schema and a notebook from table blocks.')
    parser.add_argument('project', help='Project name, used for the folder, schema module and schema name.')
    parser.add_argument(
        '--blocks', type=str, default=None,
        help=f'Comma-separated list of table blocks, e.g. "basic,rf_sta". Required blocks are added automatically. '
             f'Available: {", ".join(BLOCKS)}. Default: ask interactively.')
    parser.add_argument('--all', action='store_true',
                        help='Use all table blocks, except those that are not selected by default (motion_refine).')
    parser.add_argument(
        '--refined-roi-mask', action='store_true',
        help='Also add the optional motion_refine block (RefinedRoiMask, requires and adds the motion block). '
             'Not asked interactively and not part of --all, as it made no measurable difference to the traces.')
    parser.add_argument(
        '--n-functional', type=int, default=None,
        help='Number of functional (imaging) channels (wDataCh0, ...), traces are extracted for these channels. '
             'Use 1 for standard recordings, more for two-color imaging. Default: ask interactively, or 1.')
    parser.add_argument(
        '--n-background', type=int, default=None,
        help='Number of background channels (e.g. 1 for SR-101, a marker for blood vessels) that are recorded after '
             'the functional channels, but no traces are extracted from. Default: ask interactively, or 1.')
    parser.add_argument(
        '--from-raw-data', choices=['yes', 'no'], default=None,
        help="RawDataParams.from_raw_data: read the raw ScanM files (.smp, 'yes') or the preprocessed Igor files "
             "(.h5, 'no'). Default: ask interactively, or 'no'.")
    parser.add_argument(
        '--igor-roi-masks', choices=list(IGOR_ROI_MASKS_OPTIONS), default=None,
        help="RawDataParams.igor_roi_masks: 'yes': ROI masks from the Igor .h5 files, 'no': from mask files only, "
             "'either': per field mask file if available, else Igor, 'init': Igor masks only as starting mask in "
             "the ROI canvas. Default: ask interactively, or 'yes'.")
    parser.add_argument(
        '--legacy-userinfo-stacks', action='store_true',
        help='Use the old design without Channel table: the stacks are defined by data_stack_name and alt_stack_name '
             'in UserInfo (functional channel and alternative channel) and one trace per ROI is extracted.')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    project = args.project
    validate_project_name(project)

    blocks = resolve_blocks(select_blocks(args))
    print(f"Selected blocks: {', '.join(blocks)}")

    channels = select_channels(args)
    print(describe_channels(channels))

    raw_params = select_raw_params(args)
    print(describe_raw_params(raw_params))

    djimaging_dir = os.path.split(os.path.abspath(__file__))[0]
    project_dir = create_project_dir(djimaging_dir, project)
    copy_notebooks(djimaging_dir, project_dir, project)
    copy_table(djimaging_dir, project_dir)
    copy_schema(djimaging_dir, project_dir, project, blocks, channels)
    copy_populate_notebook(djimaging_dir, project_dir, project, blocks, channels, raw_params)


def validate_project_name(project):
    """The project name is used in the import path djimaging.user.<project>, so it must be a valid python identifier."""
    assert re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', project) and not keyword.iskeyword(project),         f"Invalid project name '{project}': use only letters, digits and underscores, and do not start with a digit "         f"(it is used in the python import path djimaging.user.<project>)."


def select_blocks(args):
    """Get requested blocks from the arguments or by asking the user."""
    extra = ['motion_refine'] if args.refined_roi_mask else []
    if args.all:
        return [name for name, info in BLOCKS.items() if info.get('default', True)] + extra
    if args.blocks is not None:
        return [b.strip() for b in args.blocks.split(',') if b.strip()] + extra
    if not sys.stdin.isatty():
        return ['core'] + extra

    print('Select the table blocks for your schema (core is always included):')
    selected = []
    for name, info in BLOCKS.items():
        if name == 'core' or not info.get('default', True):
            continue
        answer = input(f"  Include '{name}' ({info['description']})? [y/N] ")
        if answer.strip().lower() in ('y', 'yes'):
            selected.append(name)
    return selected + extra


def validate_channels(channels):
    """The channels are `None` (legacy design without Channel table) or (n_functional, n_background)."""
    if channels is None:
        return None

    n_functional, n_background = channels
    for n, minimum, name in [(n_functional, 1, 'functional'), (n_background, 0, 'background')]:
        assert isinstance(n, int) and not isinstance(n, bool) and n >= minimum, \
            f"Number of {name} channels must be an integer >= {minimum}, got {n!r}."
    return n_functional, n_background


def ask_n_channels(question, default):
    answer = input(f'{question} [{default}]: ').strip()
    return int(answer) if answer else default


def select_channels(args):
    """Get the channels from the arguments or by asking the user: `None` for the legacy design, otherwise
    (n_functional, n_background)."""
    if args.legacy_userinfo_stacks:
        assert args.n_functional is None and args.n_background is None, \
            "--n-functional and --n-background cannot be used with --legacy-userinfo-stacks."
        return None

    n_functional, n_background = args.n_functional, args.n_background
    if sys.stdin.isatty():
        if n_functional is None:
            n_functional = ask_n_channels('Number of functional channels (traces are extracted), '
                                          'e.g. 2 for two-color recordings', 1)
        if n_background is None:
            n_background = ask_n_channels('Number of background channels (no traces), e.g. 1 for SR-101', 1)

    return validate_channels((1 if n_functional is None else n_functional,
                              1 if n_background is None else n_background))


def ask_choice(question, options, default):
    while True:
        answer = input(f"{question} ({'/'.join(options)}) [{default}]: ").strip().lower()
        if not answer:
            return default
        if answer in options:
            return answer
        print(f"Please answer one of: {', '.join(options)}")


def select_raw_params(args):
    """Get the RawDataParams (from_raw_data, igor_roi_masks) from the arguments or by asking the user."""
    from_raw_data, igor_roi_masks = args.from_raw_data, args.igor_roi_masks
    if sys.stdin.isatty():
        if from_raw_data is None:
            from_raw_data = ask_choice(
                'Read the raw ScanM files (.smp) instead of the preprocessed Igor files (.h5)?',
                ('yes', 'no'), 'yes' if DEFAULT_RAW_PARAMS['from_raw_data'] else 'no')
        if igor_roi_masks is None:
            print("ROI masks: 'yes' = from the Igor .h5 files, 'no' = from mask files only (e.g. AutoROIs, ROI canvas), "
                  "'either' = per field mask file if available, else Igor, "
                  "'init' = Igor masks only as starting mask in the ROI canvas.")
            igor_roi_masks = ask_choice('Where do the ROI masks come from?', IGOR_ROI_MASKS_OPTIONS,
                                        DEFAULT_RAW_PARAMS['igor_roi_masks'])

    return validate_raw_params(dict(
        from_raw_data=DEFAULT_RAW_PARAMS['from_raw_data'] if from_raw_data is None else from_raw_data == 'yes',
        igor_roi_masks=DEFAULT_RAW_PARAMS['igor_roi_masks'] if igor_roi_masks is None else igor_roi_masks))


def validate_raw_params(raw_params):
    assert isinstance(raw_params['from_raw_data'], bool), \
        f"from_raw_data must be bool, got {raw_params['from_raw_data']!r}"
    assert raw_params['igor_roi_masks'] in IGOR_ROI_MASKS_OPTIONS, \
        f"igor_roi_masks must be one of {IGOR_ROI_MASKS_OPTIONS}, got {raw_params['igor_roi_masks']!r}"
    return raw_params


def describe_raw_params(raw_params):
    return f"RawDataParams: from_raw_data={raw_params['from_raw_data']}, igor_roi_masks={raw_params['igor_roi_masks']!r}"


def describe_channels(channels):
    if channels is None:
        return "Legacy design: stacks are defined by data_stack_name and alt_stack_name in UserInfo."
    return f"Functional channels: {channels[0]}, background channels: {channels[1]}"


def channels_name_id(channels):
    """Channel names and ids as used by `ChannelTemplate.add_channels`, e.g. [('wDataCh0', 0), ('wDataCh1', 1)].
    Functional channels come first."""
    n_functional, n_background = validate_channels(channels)
    return [(f'wDataCh{i}', i) for i in range(n_functional + n_background)]


def add_channels_to_schema(code, channels):
    """Add the Channel table, which defines the stacks instead of the UserInfo. Traces are extracted for the
    functional channels, and only get ch_id in the primary key if there are several."""
    if validate_channels(channels) is None:
        return code
    assert USER_INFO_CLASS_HEADER in code, "UserInfo table not found in the schema blocks."
    assert TRACES_CLASS_HEADER in code, "Traces table not found in the schema blocks."
    code = code.replace(
        USER_INFO_CLASS_HEADER,
        "@schema\nclass Channel(core.ChannelTemplate):\n    pass\n\n\n"
        + USER_INFO_CLASS_HEADER
        + "    _use_channels = True  # Stacks are defined by Channel, not by data_stack_name and alt_stack_name\n"
        + "    channel_table = Channel\n",
        1)

    if channels[0] == 1:
        return code
    return code.replace(
        TRACES_CLASS_HEADER,
        TRACES_CLASS_HEADER
        + "    _use_channels = True  # Traces for every functional channel in Channel, adds ch_id to the primary key\n"
        + "    channel_table = Channel\n",
        1)


def add_motion_to_schema(code, motion_code):
    """Motion mode: insert the motion tables before Traces (Traces depends on MotionRoiMask), let RoiMask only store
    the field mask (no RoiMaskPresentation, MotionRoiMask replaces it) and extract motion-corrected Traces."""
    assert ROI_MASK_PRESENTATION_CLASS in code, "RoiMaskPresentation not found in the schema blocks."
    assert TRACES_CLASS_HEADER in code, "Traces table not found in the schema blocks."
    code = code.replace(
        ROI_MASK_PRESENTATION_CLASS,
        "    # Motion mode: only the field mask, the presentation masks are computed by MotionRoiMask\n"
        "    _presentation_masks = False\n"
        "    RoiMaskPresentation = None\n",
        1)
    code = code.replace(TRACES_CLASS_HEADER, motion_code.strip("\n") + "\n\n\n" + TRACES_CLASS_HEADER, 1)
    return code.replace(
        TRACES_CLASS_HEADER,
        TRACES_CLASS_HEADER
        + "    _include_motion_correction = True  # Registered stack + presentation ROI weights of MotionRoiMask\n"
        + "    motion_roi_mask_table = MotionRoiMask\n",
        1)


def add_refine_to_motion_schema(motion_code, refine_code):
    """Insert RefinedRoiMask before MotionRoiMask and let MotionRoiMask start from the refined field mask."""
    assert MOTION_ROI_MASK_CLASS_HEADER in motion_code, "MotionRoiMask table not found in the motion block."
    return motion_code.replace(
        MOTION_ROI_MASK_CLASS_HEADER,
        refine_code.strip("\n") + "\n\n\n" + MOTION_ROI_MASK_CLASS_HEADER
        + "    refined_roi_mask_table = RefinedRoiMask  # Presentation masks start from the refined field mask\n",
        1)


def add_refine_to_motion_notebook(motion_text, refine_text):
    """Insert the RefinedRoiMask cells before the MotionRoiMask cells."""
    assert NOTEBOOK_MOTION_ROI_MASK_CELL in motion_text, "MotionRoiMask cell not found in the motion notebook block."
    return motion_text.replace(
        NOTEBOOK_MOTION_ROI_MASK_CELL, refine_text.strip("\n") + "\n\n" + NOTEBOOK_MOTION_ROI_MASK_CELL, 1)


def add_motion_to_notebook(text, motion_text):
    """Insert the motion section before the Traces section (Traces need MotionRoiMask)."""
    assert NOTEBOOK_TRACES_SECTION in text, "Traces section not found in the notebook blocks."
    return text.replace(NOTEBOOK_TRACES_SECTION, motion_text.strip("\n") + "\n\n" + NOTEBOOK_TRACES_SECTION, 1)


def add_channels_to_notebook(text, channels):
    """Add a cell that fills the Channel table."""
    if validate_channels(channels) is None:
        return text
    assert NOTEBOOK_RAW_PARAMS_CELL in text, "RawDataParams cell not found in the notebook blocks."
    n_functional, n_background = channels
    channels_name_id_ = channels_name_id(channels)
    functional = [name for name, _ in channels_name_id_[:n_functional]]

    markdown = (
        "# ## Imaging channels\n#\n"
        "# The channels of the recordings are defined in the `Channel` table, there are no `data_stack_name` and "
        "`alt_stack_name` in `UserInfo`.\n"
        f"# This schema has {n_functional} functional channel(s), traces are only extracted for these, and "
        f"{n_background} background channel(s), e.g. SR-101 as marker for blood vessels, without traces.\n"
        + ("# `ch_id` is part of the primary key of `Traces` and the tables that depend on it.\n"
           if n_functional > 1 else "")
        + "# Edit the names (the name of the data stack in the recording file) and the `functional` list if yours "
          "differ.\n#\n"
          "# All channels can be shown as background in the ROI canvas. `roi_channel` is the default target for ROI "
          "detection there (cc tool, diagnostics, AutoROIs); it can be changed in the canvas.\n"
          "# AutoROIs only use one (or two) channels, multi-color AutoROIs are not supported.\n")
    cell = (
        NOTEBOOK_RAW_PARAMS_CELL
        + "\n# %% [markdown]\n"
        + markdown
        + "\n# %%\n"
        + f"Channel().add_channels({channels_name_id_!r}, functional={functional!r}, "
          f"roi_channel={functional[0]!r}, skip_duplicates=True)\n"
        + "Channel()\n")
    return text.replace(NOTEBOOK_RAW_PARAMS_CELL, cell, 1)


def resolve_blocks(blocks):
    """Add required blocks, remove duplicates and sort blocks in the order of `BLOCKS`."""
    unknown = [b for b in blocks if b not in BLOCKS]
    assert len(unknown) == 0, f"Unknown blocks: {unknown}. Available: {list(BLOCKS)}"

    resolved = {'core'}
    todo = list(blocks)
    while todo:
        block = todo.pop()
        if block not in resolved:
            resolved.add(block)
        todo.extend(r for r in BLOCKS[block]['requires'] if r not in resolved)

    return [b for b in BLOCKS if b in resolved]


def create_project_dir(djimaging_dir, project):
    assert os.path.isdir(os.path.join(djimaging_dir, 'djimaging'))
    all_user_dir = os.path.join(djimaging_dir, 'djimaging', 'user')
    make_dir_if_new(all_user_dir)  # Make if not exists
    open(os.path.join(all_user_dir, '__init__.py'), 'a').close()
    project_dir = os.path.join(all_user_dir, project)
    make_dir_if_new(project_dir)
    open(os.path.join(project_dir, '__init__.py'), 'a').close()
    make_dir_if_new(os.path.join(project_dir, 'schemas'))
    open(os.path.join(project_dir, 'schemas', '__init__.py'), 'a').close()
    make_dir_if_new(os.path.join(project_dir, 'tables'))
    open(os.path.join(project_dir, 'tables', '__init__.py'), 'a').close()
    make_dir_if_new(os.path.join(project_dir, 'notebooks'))

    return project_dir


def copy_notebooks(djimaging_dir, project_dir, project):
    """Copy the static example notebooks. The populate notebook is generated from blocks, see below."""
    files = os.listdir(os.path.join(djimaging_dir, 'djimaging/notebooks'))
    notebooks = [nb for nb in files
                 if os.path.isfile(os.path.join(djimaging_dir, 'djimaging/notebooks', nb)) and nb.endswith('.ipynb')
                 and nb != f'{NOTEBOOK_TEMPLATE_NAME}.ipynb']
    for nb in notebooks:
        src = os.path.join(djimaging_dir, 'djimaging/notebooks', nb)
        dst = os.path.join(project_dir, 'notebooks', nb.replace('my_schema', f'{project}_schema'))

        if not os.path.isfile(dst):
            shutil.copy(src, dst)
            replace_schema_name(dst, project)


def replace_schema_name(filepath, project):
    with open(filepath, 'r') as file:
        filedata = file.read()

    filedata = filedata.replace('djimaging.schemas.tutorial_schema',
                                f'djimaging.user.{project}.schemas.{project}_schema')

    with open(filepath, 'w') as file:
        file.write(filedata)


def read_block(djimaging_dir, kind, name):
    path = os.path.join(djimaging_dir, 'djimaging', kind, 'blocks', f'{name}.py')
    with open(path, 'r', encoding='utf8') as f:
        return f.read()


def split_imports(source):
    """Split python source into its top-level import lines and the remaining code."""
    lines = source.splitlines()
    import_idxs = set()
    for node in ast.parse(source).body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            import_idxs.update(range(node.lineno - 1, node.end_lineno))

    imports = [lines[i] for i in sorted(import_idxs)]
    code = "\n".join(line for i, line in enumerate(lines) if i not in import_idxs).strip("\n")
    return imports, code


def build_schema_text(djimaging_dir, blocks, channels=DEFAULT_CHANNELS):
    """Combine the schema blocks to one schema file. Imports between blocks are replaced by the block's code."""
    imports, codes, motion_code, refine_code = [], [], None, None
    for name in blocks:
        block_imports, code = split_imports(read_block(djimaging_dir, 'schemas', name))
        for line in block_imports:
            if 'djimaging.schemas.blocks' not in line and line not in imports:
                imports.append(line)
        if name == 'motion':
            motion_code = code
        elif name == 'motion_refine':
            refine_code = code
        else:
            codes.append(code)

    code = "\n\n\n".join(codes)
    if refine_code is not None:
        assert motion_code is not None, "The motion_refine block requires the motion block."
        motion_code = add_refine_to_motion_schema(motion_code, refine_code)
    if motion_code is not None:
        code = add_motion_to_schema(code, motion_code)
    return "\n".join(imports) + "\n\n\n" + add_channels_to_schema(code, channels) + "\n"


def build_notebook_text(djimaging_dir, blocks, schema_module, project, channels=DEFAULT_CHANNELS, raw_params=None):
    """Combine the notebook blocks in the order of the blocks, followed by the clean-up section."""
    parts = [read_block(djimaging_dir, 'notebooks', name).strip("\n") for name in blocks
             if name not in ('motion', 'motion_refine')]
    parts.append(read_block(djimaging_dir, 'notebooks', 'cleanup').strip("\n"))
    text = "\n\n".join(parts) + "\n"
    if 'motion' in blocks:
        motion_text = read_block(djimaging_dir, 'notebooks', 'motion')
        if 'motion_refine' in blocks:
            motion_text = add_refine_to_motion_notebook(
                motion_text, read_block(djimaging_dir, 'notebooks', 'motion_refine'))
        text = add_motion_to_notebook(text, motion_text)
    raw_params = validate_raw_params(dict(DEFAULT_RAW_PARAMS if raw_params is None else raw_params))
    text = add_channels_to_notebook(text, channels)
    text = text.replace(FROM_RAW_DATA_PLACEHOLDER, repr(raw_params['from_raw_data'])) \
        .replace(IGOR_ROI_MASKS_PLACEHOLDER, repr(raw_params['igor_roi_masks']))
    return text.replace(SCHEMA_MODULE_PLACEHOLDER, schema_module).replace(BLOCKS_PLACEHOLDER, ", ".join(blocks))         .replace(PROJECT_PLACEHOLDER, project)


def copy_schema(djimaging_dir, project_dir, project, blocks, channels=DEFAULT_CHANNELS):
    dst = os.path.join(project_dir, f'schemas/{project}_schema.py')

    if os.path.isfile(dst):
        print(f"Schema {dst} already exists, skip.")
        return

    new_import = \
        f"# Remove comments to import templates that listed in user/{project}/tables/__init__.py \n" + \
        f"# from djimaging.user.{project} import tables as user_tables\n"

    new_table = \
        "\n\n" + \
        "# Remove comments to add table to schema\n" + \
        "# @schema\n" + \
        "# class ExampleTable(ExampleTableTemplate):\n" + \
        "#     field_table = Field\n"

    header = f"# Generated by make_project.py. Blocks: {', '.join(blocks)}. {describe_channels(channels)}\n"

    with open(dst, 'w', encoding='utf8') as f:
        f.write(header + new_import + build_schema_text(djimaging_dir, blocks, channels) + new_table)


def copy_populate_notebook(djimaging_dir, project_dir, project, blocks, channels=DEFAULT_CHANNELS, raw_params=None):
    dst = os.path.join(project_dir, 'notebooks', f'populate_{project}_schema.ipynb')

    if os.path.isfile(dst):
        print(f"Notebook {dst} already exists, skip.")
        return

    text = build_notebook_text(
        djimaging_dir, blocks, schema_module=f'djimaging.user.{project}.schemas.{project}_schema',
        project=project, channels=channels, raw_params=raw_params)

    try:
        import jupytext
    except ImportError:
        dst_py = dst.replace('.ipynb', '.py')
        with open(dst_py, 'w', encoding='utf8') as f:
            f.write(text)
        print(f"jupytext is not installed (pip install jupytext), wrote {dst_py} instead of a .ipynb file.")
        return

    nb = jupytext.reads(text, fmt='py:percent')
    nb.metadata['kernelspec'] = dict(display_name='Python 3 (ipykernel)', language='python', name='python3')
    jupytext.write(nb, dst, fmt='ipynb')


def copy_table(djimaging_dir, project_dir):
    src = os.path.join(djimaging_dir, 'djimaging/tables/_new_table_example.py')
    dst = os.path.join(project_dir, 'tables/my_table.py')

    if not os.path.isfile(dst):
        shutil.copy(src, dst)

        with open(os.path.join(project_dir, 'tables/__init__.py'), "a") as f:
            f.write("from .my_table import ExampleTableTemplate\n")


def make_dir_if_new(path):
    if not os.path.exists(path):
        os.mkdir(path)


if __name__ == '__main__':
    main()
