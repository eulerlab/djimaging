![Build status](https://github.com/eulerlab/djimaging/actions/workflows/python-app.yml/badge.svg)

# djimaging

2P imaging data joint tables and schemas

## Getting started

### Create a MySQL account

Ask IT to create a MySQL user account for you. They will send you a username and activation password.
> ❗ You probably don't want your own local MySQL server, so don't try to set one up but get access to the shared one.

### Create a remote docker container

Create a docker container and install this package and its requirements in it.

Download the package:

```bash
git clone https://github.com/eulerlab/djimaging.git
````

Install the package as an editable package using [uv](https://docs.astral.sh/uv/):

```bash
cd djimaging
uv pip install -e .
```

If you want to use autorois or receptive fields, install with the optional extras:
```bash
uv pip install -e ".[autorois]"
uv pip install -e ".[rf]"
uv pip install -e ".[autorois,rf]"
```

By default, `draw_roi_mask` / `load_default_autorois_models` only offer the latest U-Net (plus the
correlation-based model). Other models have to be requested explicitly and need additional extras
(`autorois-cellpose` for the finetuned Cellpose3 model and `autorois-mediar` for the finetuned MEDIAR model):
```bash
uv pip install -e ".[autorois,autorois-cellpose,autorois-mediar]"
```
```python
autorois_models = load_default_autorois_models(models=['UNet', 'Cellpose3_cyto3', 'MEDIAR'])  # also: 'Cellpose3_cyto2', 'UNet_v1', 'CorrRoiMask'
```

> ❗ To test if the package was successfully installed, e.g.
> open a jupyter notebook in your container and call <code>import djimaging</code>.

### Create a config file

Create a datajoint config <code>*.json</code> file,
e.g. based on the template <code>djimaging/djconfig/djconf_template.json</code>.
Fill out the missing values; if you don't know how, ask someone in your group.
> ❗ Never upload this personal config file to GitHub.

DataJoint 2 requires MySQL 8 with `utf8mb4`/`utf8mb4_bin` and named stores.
The template uses a reference-only DataJoint store rooted at
`/gpfs01/euler/data` for externally managed acquisition, stimulus, and model
files. DataJoint does not write `<filepath>` values, although filesystem tools
such as ROI-mask editors may still require write access to adjacent folders.
Traces, snippets, response vectors, field/presentation channel averages, ROI masks,
and small analysis arrays/objects use `<blob>` and are stored directly in the
database. Large stimulus and population matrices, full spatiotemporal receptive
fields, high-resolution channel averages, correlation maps, and model objects
use `<npy@processed>` or `<blob@processed>` in the separate writable
`processed` store. Paths inserted into `<filepath@reference>` fields must be
relative to the common reference root.
JSON does not support comments, so these store roles are documented here rather
than inside the config template.

### Create a project folder

Inside the root folder <code>djimaging</code> (not in <code>djimaging/djimaging</code>)
run the make project script by calling

```bash
python3 make_project.py your_project_name
```

This creates the git-ignored folder <code>djimaging/djimaging/user/your_project_name/</code> containing

- <code>schemas/your_project_name_schema.py</code>: the schema of your project, built from the table blocks you select,
- <code>notebooks/populate_your_project_name_schema.ipynb</code>: a notebook to connect to and populate this schema,
- <code>notebooks/</code>: the static example notebooks (e.g. <code>connect_to_schema.ipynb</code>),
- <code>tables/my_table.py</code>: a copy of <code>_new_table_example.py</code> as a starting point for your own table templates.

> ❗ Per default everything in this folder will not be under version control.
> Consider adding an expectation for your files here, but do not upload them to the shared repository.
> Do not upload personal config files.

#### Choosing table blocks

Instead of copying a fixed schema, <code>make_project.py</code> assembles the schema of your project (and the notebook) from
**table blocks**. A block is a small group of related tables, so you only get the tables
(and the notebook sections) you need.

| Block        | Tables                                                                                  | Requires |
|--------------|-----------------------------------------------------------------------------------------|----------|
| `core`       | UserInfo, Experiment, Field, Stimulus, Presentation, RoiMask, Roi, Traces, Snippets, Averages | –        |
| `basic`      | ChirpQI, ChirpFeatures, OsDsIndexes                                                     | `core`   |
| `location`   | OpticDisk, RelativeFieldLocation, RetinalFieldLocation                                  | `core`   |
| `classifier` | Baden16TracesV2, ClassifierV2, CelltypeAssignmentV2                                     | `core`   |
| `rf_sta`     | FastSta, SplitRF, FitGauss2DRF, FitDoG2DRF                                              | `core`   |
| `rf_glm`     | RfGlm, SplitRfGlm, FitGauss2DRfGlm, FitDoG2DRfGlm                                       | `core`   |

The project name becomes part of the import path <code>djimaging.user.&lt;project&gt;</code>, so it may only contain
letters, digits and underscores and must not start with a digit; anything else is rejected.
The notebook uses <code>ageuler_&lt;login name&gt;_&lt;project&gt;</code> as the schema name.

`core` is always included, and any block a selected block requires is added automatically.

```bash
python3 make_project.py your_project_name                         # interactive: asks y/N for each block
python3 make_project.py your_project_name --blocks basic,rf_sta   # explicit list (core is added for you)
python3 make_project.py your_project_name --all                   # every block
```

Without `--blocks`/`--all`, the script asks interactively; when there is no terminal (e.g. in a script or CI)
it falls back to `core` only. Unknown block names raise an error that lists the available blocks.

Existing files are never overwritten: if your schema or populate notebook already exists, the script prints a
message and skips it. To regenerate them with a different selection, move or delete the old files first.
If `jupytext` is not installed (<code>pip install jupytext</code>), the notebook is written as a percent-format
<code>.py</code> file instead of an <code>.ipynb</code>; it can be converted later with
<code>jupytext --to ipynb</code>.

#### How blocks work

Every block has two matching parts:

- **Schema part**: <code>djimaging/schemas/blocks/&lt;name&gt;.py</code> subclasses the table templates under
  <code>@schema</code> and binds their abstract table attributes. A block imports the block it builds on
  (e.g. <code>from djimaging.schemas.blocks.core import *</code>), so it is also a valid, importable schema on its own.
- **Notebook part**: <code>djimaging/notebooks/blocks/&lt;name&gt;.py</code> is a percent-format
  (<code># %%</code>) notebook section. <code>cleanup.py</code> is always appended last; the placeholders
  <code>__SCHEMA_MODULE__</code> and <code>__BLOCKS__</code> are filled in with your schema module and the selected blocks.

To generate your files, <code>make_project.py</code>

1. resolves the selection (adds required blocks, removes duplicates, orders them as in <code>BLOCKS</code>),
2. concatenates the schema parts into one file: the inter-block imports are dropped (the block code is already
   included above), all other imports are de-duplicated and moved to the top,
3. concatenates the notebook parts and converts them to an <code>.ipynb</code> via jupytext.

The generated schema starts with a <code># Generated by make_project.py. Blocks: ...</code> header and ends with a
commented-out example showing how to add your own table (see <code>tables/my_table.py</code>).
The ready-made schemas in <code>djimaging/schemas/</code> (<code>core_schema</code>, <code>full_rgc_schema</code>,
<code>rf_glm_schema</code>, <code>tutorial_schema</code>) are still available to copy; <code>core_schema</code>
re-exports the <code>core</code> block.

#### Adding a new block

1. Add the schema part <code>djimaging/schemas/blocks/&lt;name&gt;.py</code>. Start with an import of the blocks it
   depends on, e.g. <code>from djimaging.schemas.blocks.core import *</code>, then define the tables.
   Imports must be top-level, since they are parsed to merge the blocks.
2. Add the notebook part <code>djimaging/notebooks/blocks/&lt;name&gt;.py</code> with the sections that populate and
   inspect your tables.
3. Register the block in <code>BLOCKS</code> in <code>make_project.py</code> with a `description` and its `requires`.
   The order of that dict is the order in the generated files, so list a block after the blocks it requires.

### Clean up

When you no longer need your test schema, make sure you drop it by
calling <code>schema.drop()</code> and confirm by entering <code>yes</code>.

> ⚠️ Make sure you only drop your own schema! <code>schema.drop()</code> will show you the name of the schema.
> If you are not sure about the schema's origin, don't drop it!

## Local testing

Install the test environment with DataJoint 2.3.x:

```bash
uv pip install pytest "setuptools<81" -e .
```

Run unit tests:

```bash
pytest -q tests --ignore=tests/integration
```

Run integration tests against a temporary MySQL 8.0.43 server:

```bash
docker run --rm --name djimaging-mysql-test \
  -e MYSQL_ROOT_PASSWORD=datajoint -e MYSQL_ROOT_HOST=% \
  -p 3307:3306 -d mysql:8.0.43 \
  --character-set-server=utf8mb4 --collation-server=utf8mb4_bin
until docker exec djimaging-mysql-test mysqladmin ping -h 127.0.0.1 -pdatajoint; do sleep 1; done

DJ_TEST_MYSQL=1 DJ_HOST=127.0.0.1 DJ_PORT=3307 \
DJ_USER=root DJ_PASS=datajoint pytest -q tests/integration

docker stop djimaging-mysql-test
```
