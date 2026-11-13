"""Create a standalone djimaging project from table blocks: an installable, git-ready package with the project schema
and numbered notebooks that populate it.

By default the project is created in <parent folder of the djimaging repo>/djimaging-projects/<project>, see README.md.
"""
import argparse
import ast
import datetime
import json
import keyword
import os
import re
import shlex
import subprocess
import sys

# Table blocks that can be combined to a project schema and matching notebooks.
# Schema parts live in djimaging/schemas/blocks/<name>.py, notebook parts in djimaging/notebooks/blocks/<name>.py.
# The order of this dict is the order of the blocks in the generated files.
# `notebook`: where the notebook part goes. 'own' (default): a numbered notebook of its own, titled by the first
#   heading of the part. 'setup', 'rois' or 'traces': appended to the 01_setup, 02_roi_masks or 03_traces notebook
#   (stimulator, motion and motion_refine are inserted at fixed places instead, see `build_notebooks`).
# `extras`: extras of the djimaging package that the generated project depends on.
BLOCKS = {
    'core': dict(
        description='UserInfo, Experiment, Field, Stimulus, Presentation, RoiMask, Roi, Traces, Snippets, Averages',
        requires=[],
        notebook='core',  # Split into 01_setup, 02_roi_masks and 03_traces
        extras=['autorois']),  # AutoROIs and the ROI canvas in 02_roi_masks
    'history': dict(
        description='Experiment history, independent of stimuli: ExpHistory (every recording of every experiment), '
                    'ExpHistoryField, ExpHistoryStimLog, QdsPyLogFile, QdsPyLog',
        requires=['core'],
        notebook='setup'),
    'motion': dict(
        description='Motion correction and motion QC (motion mode): MotionParams, MotionReference, MotionDetection, '
                    'MotionRoiMask, MotionTrust. RoiMask then has no presentation masks and Traces are motion corrected',
        requires=['core'],
        notebook='traces'),
    'motion_refine': dict(
        description='Optional for motion mode: RefinedRoiMask, refines the ROI borders (+-1 px) of the field mask on the '
                    'registered reference condition before MotionRoiMask (no measurable gain on tested data)',
        requires=['motion'],
        notebook='traces',
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
        description='RGC classifier: Baden16TracesV2 (moving bar features from OsDsIndexes), ClassifierV2, '
                    'CelltypeAssignmentV2',
        requires=['basic']),
    'rf_sta': dict(
        description='Receptive fields from STA: FastSta, SplitRF, FitGauss2DRF, FitDoG2DRF',
        requires=['core']),
    'rf_glm': dict(
        description='Receptive fields from GLM: RfGlm, SplitRfGlm, FitGauss2DRfGlm, FitDoG2DRfGlm',
        requires=['core'],
        extras=['rf']),
    'stimulator': dict(
        description='Setup tables: Stimulator (geometry, delay), ImagingSetup (scan width, light artifact), '
                    'ExperimentStimulator, ExperimentImagingSetup. Inserted before Field, which uses them',
        requires=['core'],
        notebook='setup'),
    'mouse_cam': dict(
        description='Mouse cam (natural movie) orientation: MouseCamGeometry',
        requires=['stimulator']),
    'mb_direction': dict(
        description='Moving bar directions on the retina: MovingBarVersion, MbDirectionMap, MbRetinalPrefDir',
        requires=['basic', 'stimulator']),
    'rf_offset': dict(
        description='RF position relative to the ROI: RelativeRoiLocationWrtField, RfOffset, RfRoiOffset',
        requires=['location', 'rf_sta', 'stimulator']),
}

SCHEMA_MODULE_PLACEHOLDER = '__SCHEMA_MODULE__'
BLOCKS_PLACEHOLDER = '__BLOCKS__'
PROJECT_PLACEHOLDER = '__PROJECT__'
PACKAGE_PLACEHOLDER = '__PACKAGE__'
NOTEBOOK_TITLE_PLACEHOLDER = '__NOTEBOOK_TITLE__'
NOTEBOOK_LIST_PLACEHOLDER = '__NOTEBOOK_LIST__'
CREATE_SCHEMA_PLACEHOLDER = '__CREATE_SCHEMA__'

# Generated projects: <parent folder of the djimaging repo>/PROJECTS_DIR_NAME/<project>, package PACKAGE_PREFIX<project>
PROJECTS_DIR_NAME = 'djimaging-projects'
PACKAGE_PREFIX = 'dji_'

# Static example notebooks that are copied to every project (the numbered notebooks are generated from blocks).
STATIC_NOTEBOOKS = ('connect_to_schema.ipynb', 'scan_for_corrupted_recordings.ipynb')
STATIC_NOTEBOOK_SCHEMA_NAME = 'schema_name = "ageuler_ENTER_SCHEMA_NAME_HERE"'

# First heading of the parts of the core notebook block that start the 02_roi_masks and 03_traces notebooks.
NOTEBOOK_ROIS_SECTION = "# %% [markdown]\n# # Core: ROIs\n"

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

# Anchors in the stimulator blocks that are linked to tables of other blocks.
RF_ROI_OFFSET_ROI_MASK = "    roimask_tab = RoiMask.RoiMaskPresentation\n"
MB_RETINAL_FIELD_LOCATION = "    retinal_field_location_table = None  # RetinalFieldLocation"
PRESENTATION_LOG_ANCHOR = "    presentation_log_table = None  # PresentationLog"
STIM_LOG_ANCHOR = "    stim_log_table = None  # QdsPyLog.StimLog"

# Anchors in the core blocks for the setup tables of the stimulator block (inserted before Field, bound in the
# tables that load recordings).
FIELD_CLASS_HEADER = "@schema\nclass Field(core.FieldTemplate):\n"
SETUP_TABLE_USERS = (FIELD_CLASS_HEADER, "@schema\nclass Presentation(core.PresentationTemplate):\n",
                     "@schema\nclass HighRes(misc.HighResTemplate):\n",
                     "@schema\nclass ExpHistory(history.ExpHistoryTemplate):\n")  # history block: scan width
NOTEBOOK_FIELD_CELL = "# %%\nField().rescan_filesystem("


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Create a standalone project (installable package with a schema and numbered notebooks) from '
                    'table blocks. Existing files are never overwritten.')
    parser.add_argument(
        'project', help=f'Project name, used for the folder, the package ({PACKAGE_PREFIX}<project>), the schema module '
                        f'and the default schema name.')
    parser.add_argument(
        '--target-dir', type=str, default=None,
        help=f'Folder of the new project. Default: <parent folder of the djimaging repo>/{PROJECTS_DIR_NAME}/<project>. '
             f'Must be outside the djimaging repo.')
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

    djimaging_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(args.target_dir) if args.target_dir else default_project_dir(djimaging_dir, project)
    command = make_command(project, blocks, channels, raw_params, target_dir=args.target_dir)
    create_project(djimaging_dir, project_dir, project, blocks, channels, raw_params, command=command)


def validate_project_name(project):
    """The project name is used in the package name dji_<project> and the schema module <project>_schema, so it must be
    a valid python identifier."""
    assert re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', project) and not keyword.iskeyword(project), \
        f"Invalid project name '{project}': use only letters, digits and underscores, and do not start with a digit " \
        f"(it is used in the python package {PACKAGE_PREFIX}<project> and the module <project>_schema)."


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
    code = code.replace(RF_ROI_OFFSET_ROI_MASK, "    roimask_tab = MotionRoiMask  # Motion mode\n", 1)
    return code.replace(
        TRACES_CLASS_HEADER,
        TRACES_CLASS_HEADER
        + "    _include_motion_correction = True  # Registered stack + presentation ROI weights of MotionRoiMask\n"
        + "    motion_roi_mask_table = MotionRoiMask\n",
        1)


def link_optional_tables(code, blocks):
    """Bind optional table attributes to tables of other selected blocks."""
    if 'location' in blocks:
        code = code.replace(MB_RETINAL_FIELD_LOCATION, "    retinal_field_location_table = RetinalFieldLocation  #", 1)
    if 'history' in blocks:
        code = code.replace(PRESENTATION_LOG_ANCHOR, "    presentation_log_table = PresentationLog  #")
        code = code.replace(STIM_LOG_ANCHOR, "    stim_log_table = QdsPyLog.StimLog  #")
    return code


def add_setup_tables_to_schema(code, stimulator_code):
    """Insert the setup tables (stimulator block) before Field and let Field, Presentation and HighRes take the
    stimulator delay, scan width and light artifact rows from them."""
    assert FIELD_CLASS_HEADER in code, "Field table not found in the schema blocks."
    code = code.replace(FIELD_CLASS_HEADER, stimulator_code.strip("\n") + "\n\n\n" + FIELD_CLASS_HEADER, 1)
    for header in SETUP_TABLE_USERS:
        if header in code:
            code = code.replace(
                header,
                header
                + "    experiment_stimulator_table = ExperimentStimulator  # Stimulator delay\n"
                + "    experiment_imaging_setup_table = ExperimentImagingSetup  # Scan width, light artifact rows\n",
                1)
    return code


def add_setup_tables_to_notebook(text, stimulator_text):
    """Insert the setup table cells before Field (Field needs them)."""
    assert NOTEBOOK_FIELD_CELL in text, "Field cell not found in the notebook blocks."
    return text.replace(NOTEBOOK_FIELD_CELL, stimulator_text.strip("\n") + "\n\n" + NOTEBOOK_FIELD_CELL, 1)


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
    imports, codes, motion_code, refine_code, stimulator_code = [], [], None, None, None
    for name in blocks:
        block_imports, code = split_imports(read_block(djimaging_dir, 'schemas', name))
        for line in block_imports:
            if 'djimaging.schemas.blocks' not in line and line not in imports:
                imports.append(line)
        if name == 'motion':
            motion_code = code
        elif name == 'motion_refine':
            refine_code = code
        elif name == 'stimulator':
            stimulator_code = code
        else:
            codes.append(code)

    code = "\n\n\n".join(codes)
    if refine_code is not None:
        assert motion_code is not None, "The motion_refine block requires the motion block."
        motion_code = add_refine_to_motion_schema(motion_code, refine_code)
    if motion_code is not None:
        code = add_motion_to_schema(code, motion_code)
    if stimulator_code is not None:
        code = add_setup_tables_to_schema(code, stimulator_code)
    code = link_optional_tables(code, blocks)
    return "\n".join(merge_imports(imports)) + "\n\n\n" + add_channels_to_schema(code, channels) + "\n"


def merge_imports(lines, max_length=120):
    """Merge single-line `from x import a, b` imports of the same module (blocks import e.g. `core` each), in the order
    of their first occurrence. Long imports are wrapped in parentheses."""
    merged, names = [], {}
    for line in lines:
        match = re.fullmatch(r'from (\S+) import ([\w ,]+)', line.strip())
        if match is None:
            merged.append(line)
            continue
        module = match.group(1)
        if module not in names:
            names[module] = []
            merged.append(module)  # Placeholder, replaced below
        names[module] += [n.strip() for n in match.group(2).split(',') if n.strip() not in names[module]]

    result = []
    for item in merged:
        if item not in names:
            result.append(item)
            continue
        line = f"from {item} import {', '.join(names[item])}"
        if len(line) > max_length:
            line = f"from {item} import (\n    " + ",\n    ".join(names[item]) + ",\n)"
        result.append(line)
    return result


def split_core_notebook(core_text):
    """Split the core notebook block into the parts of 01_setup (data import), 02_roi_masks and 03_traces."""
    assert NOTEBOOK_ROIS_SECTION in core_text, "ROI section not found in the core notebook block."
    assert NOTEBOOK_TRACES_SECTION in core_text, "Traces section not found in the core notebook block."
    i_rois, i_traces = core_text.index(NOTEBOOK_ROIS_SECTION), core_text.index(NOTEBOOK_TRACES_SECTION)
    assert i_rois < i_traces, "The ROI section must come before the traces section in the core notebook block."
    return core_text[:i_rois], core_text[i_rois:i_traces], core_text[i_traces:]


def notebook_title(text):
    """Title of a notebook part: its first top-level markdown heading."""
    match = re.search(r'^# # (.+)$', text, flags=re.MULTILINE)
    assert match is not None, "A notebook part must start with a top-level heading (# # Title)."
    return match.group(1).strip()


def notebook_parts(djimaging_dir, blocks, channels=DEFAULT_CHANNELS, raw_params=None):
    """Bodies of the numbered notebooks as a list of (name, title, percent-format text), in the order they are run:
    setup (data import of core, setup tables before Field, Channel, blocks with notebook='setup'), roi_masks,
    traces (motion before the traces of core), then one per block with its own notebook, in the order of BLOCKS."""
    setup, rois, traces = split_core_notebook(read_block(djimaging_dir, 'notebooks', 'core'))

    if 'stimulator' in blocks:
        setup = add_setup_tables_to_notebook(setup, read_block(djimaging_dir, 'notebooks', 'stimulator'))
    raw_params = validate_raw_params(dict(DEFAULT_RAW_PARAMS if raw_params is None else raw_params))
    setup = add_channels_to_notebook(setup, channels)
    setup = setup.replace(FROM_RAW_DATA_PLACEHOLDER, repr(raw_params['from_raw_data'])) \
        .replace(IGOR_ROI_MASKS_PLACEHOLDER, repr(raw_params['igor_roi_masks']))

    if 'motion' in blocks:
        motion_text = read_block(djimaging_dir, 'notebooks', 'motion')
        if 'motion_refine' in blocks:
            motion_text = add_refine_to_motion_notebook(
                motion_text, read_block(djimaging_dir, 'notebooks', 'motion_refine'))
        traces = add_motion_to_notebook(traces, motion_text)

    shared = dict(setup=setup, rois=rois, traces=traces)
    own = []
    for name in blocks:
        if name in ('core', 'stimulator', 'motion', 'motion_refine'):  # Inserted above
            continue
        text = read_block(djimaging_dir, 'notebooks', name).strip("\n")
        where = BLOCKS[name].get('notebook', 'own')
        if where == 'own':
            own.append((name, notebook_title(text), text))
        else:
            assert where in shared, f"Unknown notebook '{where}' of block '{name}'."
            shared[where] = shared[where].strip("\n") + "\n\n" + text

    traces_title = 'Motion correction, traces and responses' if 'motion' in blocks else 'Traces and responses'
    return [('setup', 'Setup and data import', shared['setup'].strip("\n")),
            ('roi_masks', 'ROI masks', shared['rois'].strip("\n")),
            ('traces', traces_title, shared['traces'].strip("\n"))] + own


def numbered_notebooks(parts):
    """[(name, title, body)] -> [(file name without extension, e.g. 01_setup, title, body)]"""
    return [(f'{i:02d}_{name}', title, body) for i, (name, title, body) in enumerate(parts, 1)]


def build_notebooks(djimaging_dir, blocks, project, package=None, schema_module=None, channels=DEFAULT_CHANNELS,
                    raw_params=None):
    """Combine the notebook blocks to the notebooks of a project: {file name without extension: percent-format text}.

    The numbered notebooks (01_setup, 02_roi_masks, 03_traces, then one per further block, see `notebook_parts`) start
    with the same preamble (load config, import and activate the schema); only 01_setup creates the schema and shows
    the diagram. `drop_schema` (the clean-up section) is not numbered, so that it is never run by accident.
    """
    package = package_name(project) if package is None else package
    schema_module = f'{package}.schemas.{project}_schema' if schema_module is None else schema_module
    preamble = read_block(djimaging_dir, 'notebooks', 'preamble').strip("\n")
    overview = read_block(djimaging_dir, 'notebooks', 'overview').strip("\n")

    numbered = numbered_notebooks(notebook_parts(djimaging_dir, blocks, channels, raw_params))

    def notebook_list(current):
        return "\n".join(f"# {i}. `{stem}`: {title}" + (" (this notebook)" if stem == current else "")
                         for i, (stem, title, _) in enumerate(numbered, 1))

    def head(title, current, create_schema):
        return preamble.replace(NOTEBOOK_TITLE_PLACEHOLDER, title) \
            .replace(NOTEBOOK_LIST_PLACEHOLDER, notebook_list(current)) \
            .replace(CREATE_SCHEMA_PLACEHOLDER, repr(create_schema))

    notebooks = {}
    for i, (stem, title, body) in enumerate(numbered):
        first = i == 0
        parts = [head(f"{stem[:2]}: {title}", stem, create_schema=first)] + ([overview] if first else []) + [body]
        notebooks[stem] = "\n\n".join(parts) + "\n"
    notebooks['drop_schema'] = "\n\n".join(
        [head('Drop schema', None, create_schema=False),
         read_block(djimaging_dir, 'notebooks', 'cleanup').strip("\n")]) + "\n"

    return {stem: text.replace(SCHEMA_MODULE_PLACEHOLDER, schema_module).replace(PACKAGE_PLACEHOLDER, package)
            .replace(BLOCKS_PLACEHOLDER, ", ".join(blocks)).replace(PROJECT_PLACEHOLDER, project)
            for stem, text in notebooks.items()}


def build_notebook_text(djimaging_dir, blocks, schema_module, project, channels=DEFAULT_CHANNELS, raw_params=None):
    """All notebooks of `build_notebooks` in one text (the numbered ones in order, then drop_schema)."""
    notebooks = build_notebooks(djimaging_dir, blocks, project, package=schema_module.split('.')[0],
                                schema_module=schema_module, channels=channels, raw_params=raw_params)
    return "\n\n".join(text.strip("\n") for text in notebooks.values()) + "\n"


# ----------------------------------------------------------------------------------------------------------------------
# Standalone project
# ----------------------------------------------------------------------------------------------------------------------
def package_name(project):
    return f'{PACKAGE_PREFIX}{project}'


def default_project_dir(djimaging_dir, project):
    """<parent folder of the djimaging repo>/djimaging-projects/<project>"""
    return os.path.join(os.path.dirname(os.path.abspath(djimaging_dir)), PROJECTS_DIR_NAME, project)


def is_inside(path, folder):
    path, folder = os.path.normcase(os.path.abspath(path)), os.path.normcase(os.path.abspath(folder))
    try:
        return os.path.commonpath([path, folder]) == folder
    except ValueError:  # E.g. different drives on Windows
        return False


def make_command(project, blocks, channels=DEFAULT_CHANNELS, raw_params=None, target_dir=None):
    """The make_project command with every option given explicitly, so that it creates the same project again without
    asking (it is recorded in the README of the project)."""
    raw_params = validate_raw_params(dict(DEFAULT_RAW_PARAMS if raw_params is None else raw_params))
    selected = [b for b in blocks if b != 'core'] or ['core']
    argv = ['python', 'make_project.py', project, '--blocks', ','.join(selected)]
    if validate_channels(channels) is None:
        argv += ['--legacy-userinfo-stacks']
    else:
        argv += ['--n-functional', str(channels[0]), '--n-background', str(channels[1])]
    argv += ['--from-raw-data', 'yes' if raw_params['from_raw_data'] else 'no',
             '--igor-roi-masks', raw_params['igor_roi_masks']]
    if target_dir is not None:
        argv += ['--target-dir', target_dir]
    return shlex.join(argv)


def format_command(command, prefix=''):
    """The command on several lines (one option per line, shell line continuations), each line starting with
    `prefix`, e.g. '# ' for comments."""
    return prefix + f" \\\n{prefix}    ".join(re.split(r' (?=--)', command))


def djimaging_version(djimaging_dir):
    """Commit (and branch) of the djimaging repo, '-dirty' if it has uncommitted changes."""
    def git(*args):
        return subprocess.run(['git', '-C', str(djimaging_dir), *args], capture_output=True, text=True,
                              check=True, timeout=10).stdout.strip()
    try:
        return f"commit {git('describe', '--always', '--dirty')} on branch {git('rev-parse', '--abbrev-ref', 'HEAD')}"
    except (OSError, subprocess.SubprocessError):
        return "unknown commit"


def djimaging_extras(blocks):
    extras = []
    for name in blocks:
        extras += [e for e in BLOCKS[name].get('extras', []) if e not in extras]
    return extras


def djimaging_source_path(djimaging_dir, project_dir, max_up=3):
    """Path to the djimaging repo for [tool.uv.sources]: relative to the project if it is close (at most `max_up`
    folders up, e.g. ../../djimaging for the default folder), else absolute."""
    djimaging_dir, project_dir = os.path.realpath(djimaging_dir), os.path.realpath(project_dir)  # realpath: long names
    try:
        path = os.path.relpath(djimaging_dir, project_dir)
    except ValueError:  # Different drives on Windows
        path = djimaging_dir
    if path.split(os.sep).count(os.pardir) > max_up:
        path = djimaging_dir
    return path.replace(os.sep, '/')


def toml_str(value):
    return json.dumps(value)  # A JSON string is a valid TOML basic string


def build_pyproject(project, blocks, djimaging_path):
    package = package_name(project)
    extras = djimaging_extras(blocks)
    djimaging_dep = f"djimaging[{','.join(extras)}]" if extras else 'djimaging'
    return f"""[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"

[project]
name = {toml_str(package)}
version = "0.0.1"
description = {toml_str(f'djimaging project {project}: DataJoint schema, tables and notebooks')}
requires-python = ">=3.10"
dependencies = [
    {toml_str(djimaging_dep)},
]

[project.optional-dependencies]
dev = [
    "pytest>=9.0.3",
    "flake8>=7.4.1",
]

[tool.setuptools.packages.find]
where = ["."]
include = [{toml_str(package + '*')}]

[tool.pytest.ini_options]
filterwarnings = [
    "error:fetch\\\\(\\\\) is deprecated.*:DeprecationWarning",
    "error:ERD is deprecated.*:DeprecationWarning",
]

[tool.uv.sources]
# The local djimaging repo, editable. Remove this to install djimaging from the package index instead.
djimaging = {{ path = {toml_str(djimaging_path)}, editable = true }}
"""


GITIGNORE = """\
# Python
__pycache__/
*.py[cod]
*.egg-info/
.eggs/
build/
dist/
.pytest_cache/
.mypy_cache/
.coverage
htmlcov/

# Environments
.venv/
venv/
env/
.env

# Jupyter
.ipynb_checkpoints/

# DataJoint config files contain credentials: never commit them
djconfig/
dj_*conf*.json
*_conf.json

# Large or derived data: keep it in the DataJoint stores, not in git
*.pkl
*.pickle
*.h5
*.npy
*.npz

# IDEs and OS
.idea/
.vscode/
.DS_Store
Thumbs.db
"""

GITATTRIBUTES = "* text=auto\n"

FLAKE8 = """\
[flake8]
max-line-length = 127
exclude = .git,__pycache__,.venv,build
"""


def build_settings(project):
    return f'''"""Settings shared by all notebooks of the project {project}."""
import getpass
import os

# DataJoint config file, made from djimaging/djconfig/djconf_template.json. It contains credentials: never commit it.
CONFIG_FILE = os.path.join(os.path.expanduser('~'), 'datajoint', f'dj_{{getpass.getuser()}}_conf.json')

# Name of the database schema, it should start with 'ageuler_'. The default contains your login name, so that every
# user gets their own schema. For a schema that several users populate together, use a fixed name instead, e.g.
# 'ageuler_{project}'. All notebooks of this project use this name.
SCHEMA_NAME = f'ageuler_{{getpass.getuser()}}_{project}'
'''


def build_schema_file(djimaging_dir, project, blocks, channels=DEFAULT_CHANNELS, command=None):
    package = package_name(project)
    command_lines = [line + ("  # noqa: E501" if len(line) > 120 else "")  # E.g. a long --target-dir
                     for line in format_command(command, prefix='#   ').splitlines()] if command else []
    header = (f"# Schema of the djimaging project {project}, generated by make_project.py.\n"
              f"# Blocks: {', '.join(blocks)}.\n"
              f"# {describe_channels(channels)}\n"
              + ("# Command:\n" + "\n".join(command_lines) + "\n" if command else "")
              + f"# Remove the comment to use the table templates of {package}/tables/__init__.py\n"
              + f"# from {package} import tables as project_tables\n")
    example_table = (
        "\n\n"
        "# Remove the comments to add the example table (project_tables.ExampleTableTemplate) to the schema\n"
        "# @schema\n"
        "# class ExampleTable(project_tables.ExampleTableTemplate):\n"
        "#     field_table = Field\n")
    return header + build_schema_text(djimaging_dir, blocks, channels) + example_table


def build_schema_test(project):
    package = package_name(project)
    return f'''"""The schema can be imported and defines its tables. No database connection is needed."""
import datajoint as dj

from {package}.schemas import {project}_schema


def test_schema_defines_tables():
    assert isinstance({project}_schema.schema, dj.Schema)
    for name in ('UserInfo', 'Experiment', 'Field', 'Presentation', 'RoiMask', 'Roi', 'Traces'):
        assert isinstance(getattr({project}_schema, name), type), name
'''


def kernel_setup_commands(project):
    """Shell commands that install the project and register its environment as a Jupyter kernel."""
    return ["uv sync", "source .venv/bin/activate", f"python -m ipykernel install --user --name={project}"]


def next_steps(project, project_dir):
    """What to do after creating the project, printed at the end."""
    commands = '\n'.join(f"    {c}" for c in [f'cd "{project_dir}"'] + kernel_setup_commands(project))
    return (f"\nNext steps (details in README.md there):\n"
            f"1. Install the project and register its Jupyter kernel:\n{commands}\n"
            f"2. Restart Jupyter by refreshing the page in the browser.\n"
            f"3. Check {package_name(project)}/settings.py (DataJoint config file and schema name).\n"
            f"4. Open the notebooks in {package_name(project)}/notebooks and select the kernel '{project}'.\n")


def build_readme(djimaging_dir, project_dir, project, blocks, notebook_titles, command):
    """README of the project. `notebook_titles`: [(file name without extension, title)] of the numbered notebooks."""
    package = package_name(project)
    djimaging_path = djimaging_source_path(djimaging_dir, project_dir)
    extras = djimaging_extras(blocks)
    pip_target = f'"{djimaging_path}[{",".join(extras)}]"' if extras else f'"{djimaging_path}"'
    block_rows = "\n".join(f"| `{name}` | {BLOCKS[name]['description']} |" for name in blocks)
    notebook_rows = "\n".join(f"| `{stem}` | {title} |" for stem, title in notebook_titles)
    return f"""# {project}

TODO: describe the project.

Created on {datetime.date.today().isoformat()} with `make_project.py` of djimaging ({djimaging_version(djimaging_dir)}):

```bash
{format_command(command)}
```

## Install

With [uv](https://docs.astral.sh/uv/), using the local djimaging repo (`{djimaging_path}`, see `[tool.uv.sources]` in
`pyproject.toml`):

```bash
{chr(10).join(kernel_setup_commands(project))}
```

`uv sync --extra dev` also installs pytest and flake8. On Windows, activate with `.venv\\Scripts\\activate`.

Or with pip, djimaging first (editable), in an environment of your choice:

```bash
pip install -e {pip_target}
pip install -e ".[dev]"
python -m ipykernel install --user --name={project}
```

Then restart Jupyter by refreshing the page in the browser, open a notebook and select the kernel `{project}`.

## Configuration

`{package}/settings.py` sets the DataJoint config file (`CONFIG_FILE`, made from
`djimaging/djconfig/djconf_template.json`) and the schema name (`SCHEMA_NAME`) for all notebooks. Config files contain
credentials: never commit them (they are in `.gitignore`).

## Schema

`{package}/schemas/{project}_schema.py`, built from these table blocks of djimaging:

| Block | Tables |
|---|---|
{block_rows}

Own table templates go to `{package}/tables/` (see the example `my_table.py`) and are bound in the schema.

## Notebooks

The notebooks are in `{package}/notebooks/`. Run them in this order:

| Notebook | Content |
|---|---|
{notebook_rows}

`drop_schema` drops the schema with all its data. `connect_to_schema` connects to a schema without its definition and
`scan_for_corrupted_recordings` finds recordings that turned into noise.

## Tests

```bash
pytest -v tests
```
"""


def percent_to_ipynb(text):
    import jupytext
    nb = jupytext.reads(text, fmt='py:percent')
    nb.metadata['kernelspec'] = dict(display_name='Python 3 (ipykernel)', language='python', name='python3')
    return jupytext.writes(nb, fmt='ipynb')


def build_static_notebook(djimaging_dir, name, package):
    """A static example notebook, with the schema name taken from the settings of the project."""
    with open(os.path.join(djimaging_dir, 'djimaging', 'notebooks', name), 'r', encoding='utf8') as f:
        nb = json.load(f)
    for cell in nb['cells']:
        source = ''.join(cell['source'])
        if STATIC_NOTEBOOK_SCHEMA_NAME in source:
            source = source.replace(STATIC_NOTEBOOK_SCHEMA_NAME,
                                    f"from {package} import settings\n\nschema_name = settings.SCHEMA_NAME")
            cell['source'] = source.splitlines(keepends=True)
    return json.dumps(nb, indent=1, ensure_ascii=False) + "\n"


def build_project_files(djimaging_dir, project_dir, project, blocks, channels=DEFAULT_CHANNELS, raw_params=None,
                        command=None):
    """All files of a new project: {path relative to the project folder (with /): text}."""
    package = package_name(project)
    command = make_command(project, blocks, channels, raw_params) if command is None else command
    notebooks = build_notebooks(djimaging_dir, blocks, project, package=package, channels=channels,
                                raw_params=raw_params)
    notebook_titles = [(stem, title) for stem, title, _ in numbered_notebooks(
        notebook_parts(djimaging_dir, blocks, channels, raw_params))]
    with open(os.path.join(djimaging_dir, 'djimaging', 'tables', '_new_table_example.py'), 'r', encoding='utf8') as f:
        example_table = f.read()

    files = {
        'README.md': build_readme(djimaging_dir, project_dir, project, blocks, notebook_titles, command),
        'pyproject.toml': build_pyproject(project, blocks, djimaging_source_path(djimaging_dir, project_dir)),
        '.gitignore': GITIGNORE,
        '.gitattributes': GITATTRIBUTES,
        '.flake8': FLAKE8,
        f'{package}/__init__.py': f'"""djimaging project {project}."""\n',
        f'{package}/settings.py': build_settings(project),
        f'{package}/schemas/__init__.py': '',
        f'{package}/schemas/{project}_schema.py': build_schema_file(djimaging_dir, project, blocks, channels, command),
        f'{package}/tables/__init__.py': 'from .my_table import ExampleTableTemplate  # noqa: F401\n',
        f'{package}/tables/my_table.py': example_table,
        'tests/test_schema.py': build_schema_test(project),
    }

    try:
        import jupytext  # noqa: F401
        has_jupytext = True
    except ImportError:
        has_jupytext = False
        print("jupytext is not installed (pip install jupytext), the notebooks are written as percent-format .py files.")
    for stem, text in notebooks.items():
        if has_jupytext:
            files[f'{package}/notebooks/{stem}.ipynb'] = percent_to_ipynb(text)
        else:
            files[f'{package}/notebooks/{stem}.py'] = text
    for name in STATIC_NOTEBOOKS:
        files[f'{package}/notebooks/{name}'] = build_static_notebook(djimaging_dir, name, package)
    return files


def create_project(djimaging_dir, project_dir, project, blocks, channels=DEFAULT_CHANNELS, raw_params=None,
                   command=None):
    """Write the files of a new project to `project_dir`. Existing files are never overwritten, they are skipped.
    Returns the created and the skipped files (relative paths)."""
    assert not is_inside(project_dir, djimaging_dir), \
        f"The project folder {project_dir} must be outside the djimaging repo {djimaging_dir}."
    files = build_project_files(djimaging_dir, project_dir, project, blocks, channels, raw_params, command)

    created, skipped = [], []
    for relpath, text in files.items():
        path = os.path.join(project_dir, *relpath.split('/'))
        if os.path.exists(path):
            print(f"{path} already exists, skip.")
            skipped.append(relpath)
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf8', newline='') as f:
            f.write(text)
        created.append(relpath)

    print(f"Project {project}: created {len(created)} files, skipped {len(skipped)} existing files in {project_dir}")
    print(next_steps(project, project_dir))
    return created, skipped


if __name__ == '__main__':
    main()
