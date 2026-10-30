import argparse
import ast
import os
import shutil
import sys

# Table blocks that can be combined to a personal schema and a matching tutorial notebook.
# Schema parts live in djimaging/schemas/blocks/<name>.py, notebook parts in djimaging/notebooks/blocks/<name>.py.
# The order of this dict is the order of the blocks in the generated files.
BLOCKS = {
    'core': dict(
        description='UserInfo, Experiment, Field, Stimulus, Presentation, RoiMask, Roi, Traces, Snippets, Averages',
        requires=[]),
    'basic': dict(
        description='ChirpQI, ChirpFeatures, OsDsIndexes',
        requires=['core']),
    'location': dict(
        description='OpticDisk, RelativeFieldLocation, RetinalFieldLocation',
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
NOTEBOOK_TEMPLATE_NAME = 'tutorial_my_schema'  # Replaced by tutorial_<user>_schema


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Create a personal folder with a schema and a tutorial notebook from table blocks.')
    parser.add_argument('user')
    parser.add_argument(
        '--blocks', type=str, default=None,
        help=f'Comma-separated list of table blocks, e.g. "basic,rf_sta". Required blocks are added automatically. '
             f'Available: {", ".join(BLOCKS)}. Default: ask interactively.')
    parser.add_argument('--all', action='store_true', help='Use all table blocks.')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    user = args.user
    assert len(user) > 0, 'Enter valid username'

    blocks = resolve_blocks(select_blocks(args))
    print(f"Selected blocks: {', '.join(blocks)}")

    djimaging_dir = os.path.split(os.path.abspath(__file__))[0]
    user_dir = create_user_dir(djimaging_dir, user)
    copy_notebooks(djimaging_dir, user_dir, user)
    copy_table(djimaging_dir, user_dir)
    copy_schema(djimaging_dir, user_dir, user, blocks)
    copy_tutorial_notebook(djimaging_dir, user_dir, user, blocks)


def select_blocks(args):
    """Get requested blocks from the arguments or by asking the user."""
    if args.all:
        return list(BLOCKS)
    if args.blocks is not None:
        return [b.strip() for b in args.blocks.split(',') if b.strip()]
    if not sys.stdin.isatty():
        return ['core']

    print('Select the table blocks for your schema (core is always included):')
    selected = []
    for name, info in BLOCKS.items():
        if name == 'core':
            continue
        answer = input(f"  Include '{name}' ({info['description']})? [y/N] ")
        if answer.strip().lower() in ('y', 'yes'):
            selected.append(name)
    return selected


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


def create_user_dir(djimaging_dir, user):
    assert os.path.isdir(os.path.join(djimaging_dir, 'djimaging'))
    all_user_dir = os.path.join(djimaging_dir, 'djimaging', 'user')
    make_dir_if_new(all_user_dir)  # Make if not exists
    open(os.path.join(all_user_dir, '__init__.py'), 'a').close()
    user_dir = os.path.join(all_user_dir, user)
    make_dir_if_new(user_dir)
    open(os.path.join(user_dir, '__init__.py'), 'a').close()
    make_dir_if_new(os.path.join(user_dir, 'schemas'))
    open(os.path.join(user_dir, 'schemas', '__init__.py'), 'a').close()
    make_dir_if_new(os.path.join(user_dir, 'tables'))
    open(os.path.join(user_dir, 'tables', '__init__.py'), 'a').close()
    make_dir_if_new(os.path.join(user_dir, 'notebooks'))

    return user_dir


def copy_notebooks(djimaging_dir, user_dir, username):
    """Copy the static example notebooks. The tutorial notebook is generated from blocks, see below."""
    files = os.listdir(os.path.join(djimaging_dir, 'djimaging/notebooks'))
    notebooks = [nb for nb in files
                 if os.path.isfile(os.path.join(djimaging_dir, 'djimaging/notebooks', nb)) and nb.endswith('.ipynb')
                 and nb != f'{NOTEBOOK_TEMPLATE_NAME}.ipynb']
    for nb in notebooks:
        src = os.path.join(djimaging_dir, 'djimaging/notebooks', nb)
        dst = os.path.join(user_dir, 'notebooks', nb.replace('my_schema', f'{username}_schema'))

        if not os.path.isfile(dst):
            shutil.copy(src, dst)
            replace_schema_name(dst, username)


def replace_schema_name(filepath, username):
    with open(filepath, 'r') as file:
        filedata = file.read()

    filedata = filedata.replace('djimaging.schemas.tutorial_schema',
                                f'djimaging.user.{username}.schemas.{username}_schema')

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


def build_schema_text(djimaging_dir, blocks):
    """Combine the schema blocks to one schema file. Imports between blocks are replaced by the block's code."""
    imports, codes = [], []
    for name in blocks:
        block_imports, code = split_imports(read_block(djimaging_dir, 'schemas', name))
        for line in block_imports:
            if 'djimaging.schemas.blocks' not in line and line not in imports:
                imports.append(line)
        codes.append(code)

    return "\n".join(imports) + "\n\n\n" + "\n\n\n".join(codes) + "\n"


def build_notebook_text(djimaging_dir, blocks, schema_module):
    """Combine the notebook blocks in the order of the blocks, followed by the clean-up section."""
    parts = [read_block(djimaging_dir, 'notebooks', name).strip("\n") for name in blocks]
    parts.append(read_block(djimaging_dir, 'notebooks', 'cleanup').strip("\n"))
    text = "\n\n".join(parts) + "\n"
    return text.replace(SCHEMA_MODULE_PLACEHOLDER, schema_module).replace(BLOCKS_PLACEHOLDER, ", ".join(blocks))


def copy_schema(djimaging_dir, user_dir, username, blocks):
    dst = os.path.join(user_dir, f'schemas/{username}_schema.py')

    if os.path.isfile(dst):
        print(f"Schema {dst} already exists, skip.")
        return

    new_import = \
        f"# Remove comments to import templates that listed in user/{username}/tables/__init__.py \n" + \
        f"# from djimaging.user.{username} import tables as user_tables\n"

    new_table = \
        "\n\n" + \
        "# Remove comments to add table to schema\n" + \
        "# @schema\n" + \
        "# class ExampleTable(ExampleTableTemplate):\n" + \
        "#     field_table = Field\n"

    header = f"# Generated by make_user.py. Blocks: {', '.join(blocks)}\n"

    with open(dst, 'w', encoding='utf8') as f:
        f.write(header + new_import + build_schema_text(djimaging_dir, blocks) + new_table)


def copy_tutorial_notebook(djimaging_dir, user_dir, username, blocks):
    dst = os.path.join(user_dir, 'notebooks', f'tutorial_{username}_schema.ipynb')

    if os.path.isfile(dst):
        print(f"Notebook {dst} already exists, skip.")
        return

    text = build_notebook_text(
        djimaging_dir, blocks, schema_module=f'djimaging.user.{username}.schemas.{username}_schema')

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


def copy_table(djimaging_dir, user_dir):
    src = os.path.join(djimaging_dir, 'djimaging/tables/_new_table_example.py')
    dst = os.path.join(user_dir, 'tables/my_table.py')

    if not os.path.isfile(dst):
        shutil.copy(src, dst)

        with open(os.path.join(user_dir, 'tables/__init__.py'), "a") as f:
            f.write("from .my_table import ExampleTableTemplate\n")


def make_dir_if_new(path):
    if not os.path.exists(path):
        os.mkdir(path)


if __name__ == '__main__':
    main()
