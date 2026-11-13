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

### AutoROIs models

By default, `draw_roi_mask` / `load_default_autorois_models` offer the latest U-Net (plus the
correlation-based model), which only need the `autorois` extra. Other models have to be requested explicitly:
```python
autorois_models = load_default_autorois_models(models=['UNet', 'Cellpose3_cyto3', 'MEDIAR'])  # also: 'Cellpose3_cyto2', 'UNet_v1', 'CorrRoiMask'
```
The finetuned Cellpose3 (`Cellpose3_cyto2`, `Cellpose3_cyto3`) and MEDIAR models are **not** part of the package
dependencies, because their requirements conflict with the up-to-date core dependencies:
`mediar` (fork `sarmueller/MEDIAR`) pins `monai==1.4.0`, which needs `numpy<2`, and `cellpose` 3.x needs
`numpy<2.1` (older `numpy`/`numba` for everyone). Adding them breaks `uv sync`.
If you need them, install them yourself, preferably in a separate environment, e.g.
`uv pip install "cellpose>=3.1.1,<4"` or `uv pip install git+https://github.com/sarmueller/MEDIAR.git`
(possibly with `monai>=1.5` and `--no-deps` to keep your `numpy`). The MEDIAR wrapper is untested with `monai>=1.5`.

How ROI masks get into the database, the ROI canvas, corrections, shrink to fit and motion mode:
see [ROI_MASKS.md](ROI_MASKS.md).

> ⚠️ **Motion detection and motion QC are experimental** (`motion` / `motion_refine` blocks: `MotionDetection`,
> `MotionRoiMask`, `MotionTrust`, motion-corrected `Traces`; also `HighResAlignment`). They were validated on a
> limited set of recordings (xy RGC fields of four experiments and one z-stack), and thresholds and the z estimates
> are not calibrated beyond that. Check the plots for your data before relying on them. Method, validation and
> known limitations: [docs/motion_qc.md](docs/motion_qc.md).

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

### Create a project

Inside the root folder <code>djimaging</code> (not in <code>djimaging/djimaging</code>)
run the make project script by calling

```bash
python3 make_project.py your_project_name
```

This creates a standalone project next to the djimaging repo, in
<code>&lt;parent folder of djimaging&gt;/djimaging-projects/your_project_name/</code>
(choose another folder with <code>--target-dir PATH</code>; it must be outside the djimaging repo).
The project is an installable package (<code>dji_your_project_name</code>) that is ready to be a git repository:

```
djimaging-projects/your_project_name/
  README.md               the exact make_project command, djimaging commit, blocks and notebooks
  pyproject.toml          depends on djimaging (+ the extras the blocks need), [tool.uv.sources] points to your
                          local djimaging repo (editable); pytest fails on DataJoint v1 fetch()/ERD warnings
  .gitignore              Python, environments, notebook checkpoints, DataJoint config files, large data files
  .gitattributes, .flake8
  tests/test_schema.py    the schema can be imported and defines its tables (no database needed)
  dji_your_project_name/
    settings.py           CONFIG_FILE and SCHEMA_NAME, used by all notebooks
    schemas/your_project_name_schema.py   the schema, built from the table blocks you select
    tables/my_table.py    copy of _new_table_example.py, a starting point for your own table templates
    notebooks/
      01_setup.ipynb      data import (UserInfo, Channel, RawDataParams, Experiment, setup tables, Field, Stimulus,
                          Presentation, experiment history), creates the schema and shows its diagram
      02_roi_masks.ipynb  ROI masks (AutoROIs, corrections, ROI canvas) and Roi
      03_traces.ipynb     motion correction (motion block), traces, preprocessing, snippets, averages
      04_<block>.ipynb    one numbered notebook per further block, in the order of the table below
      drop_schema.ipynb   drops the schema (not numbered, so it is never run by accident)
      connect_to_schema.ipynb, scan_for_corrupted_recordings.ipynb
```

Then install the project, e.g. with <code>uv sync --extra dev</code> in the project folder (or
<code>pip install -e "&lt;djimaging&gt;[autorois]"</code> and <code>pip install -e ".[dev]"</code>), check
<code>settings.py</code> and run the notebooks in their order. Every notebook connects on its own (config, schema,
<code>activate_schema</code>), so a notebook can be rerun without the ones before; only <code>01_setup</code> creates
the schema. <code>settings.SCHEMA_NAME</code> defaults to <code>ageuler_&lt;login name&gt;_&lt;project&gt;</code>; use
a fixed name for a schema that several users populate together.

> ❗ Do not commit personal config files (the generated <code>.gitignore</code> excludes the usual names).

Older projects in the git-ignored <code>djimaging/djimaging/user/&lt;project&gt;/</code> folder keep working
(imported as <code>djimaging.user.&lt;project&gt;</code>), but <code>make_project.py</code> no longer creates them.

#### Choosing table blocks

Instead of copying a fixed schema, <code>make_project.py</code> assembles the schema of your project (and the notebook) from
**table blocks**. A block is a small group of related tables, so you only get the tables
(and the notebook sections) you need.

| Block        | Tables                                                                                  | Requires |
|--------------|-----------------------------------------------------------------------------------------|----------|
| `core`       | UserInfo, Experiment, Field, Stimulus, Presentation, RoiMask, Roi, Traces, Snippets, Averages | –        |
| `history`    | ExpHistory, ExpHistoryField, ExpHistoryStimLog, QdsPyLogFile, QdsPyLog, PresentationLog (see below) | `core`   |
| `basic`      | ChirpQI, ChirpFeatures, OsDsIndexes                                                     | `core`   |
| `location`   | OpticDisk, RelativeFieldLocation, RetinalFieldLocation                                  | `core`   |
| `highres`    | HighResAlignment (shift and match of the HighRes scan vs. the ROI mask; populates HighRes) | `core` |
| `classifier` | Baden16TracesV2, ClassifierV2, CelltypeAssignmentV2                                     | `basic`  |
| `rf_sta`     | FastSta, SplitRF, FitGauss2DRF, FitDoG2DRF                                              | `core`   |
| `rf_glm`     | RfGlm, SplitRfGlm, FitGauss2DRfGlm, FitDoG2DRfGlm                                       | `core`   |
| `stimulator` | Stimulator, ExperimentStimulator, ImagingSetup, ExperimentImagingSetup (before Field)  | `core`   |
| `mouse_cam`  | MouseCamGeometry                                                                        | `stimulator` |
| `mb_direction` | MovingBarVersion, MbDirectionMap, MbRetinalPrefDir                                   | `basic`, `stimulator` |
| `rf_offset`  | RelativeRoiLocationWrtField, RfOffset, RfRoiOffset                                      | `location`, `rf_sta`, `stimulator` |

The `classifier` block reads the moving bar features of the RGC classifier (time component and DS p-value) from
`OsDsIndexes` (block `basic`), so both use the same moving bar analysis (Baden et al. 2016: SVD tuning curve,
DS/OS index as resultant vector length, permutation test). They match the training data of the classifier only with
the per-trial snippet baseline of Baden et al. 2016 (`snippet_base_dt` in `Stimulus`: 5 x 0.128 s for the moving bar,
8 x 0.128 s for the chirp; the defaults of `add_movingbar` and `add_chirp`), `OsDsIndexes._n_shuffles >= 1000` and
moving bar snippets at 7.8125 Hz; `Baden16TracesV2` checks this. Without the per-trial baseline, slow drift shared by the
consecutive opposite directions of the moving bar inflates orientation selectivity.

The `stimulator` blocks hold all setup-dependent parameters, per setup and period and assigned to each experiment:
stimulus geometry (how the QDSpy screen is projected onto the retina, used for moving bar directions on the retina,
RF positions relative to ROIs and mouse cam orientation) and stimulator delay (`Stimulator`), and scan width and
light artifact rows (`ImagingSetup`). `make_project.py` inserts the `stimulator` tables before `Field` and binds them
in `Field`, `Presentation` and `HighRes`, which then take delay, pixel size and artifact rows from them (without the
block, the `setupid` constants of `setup_utils` are used as before). Every assignment has a confidence (`high`,
`medium`, `low`) and a reason; conventions are documented in `djimaging/utils/stimulator.py`. With the `history`
block, moving bar and mouse cam versions are read from the QDSpy logs (`PresentationLog`).

The project name becomes part of the package name <code>dji_&lt;project&gt;</code> and the module
<code>&lt;project&gt;_schema</code>, so it may only contain letters, digits and underscores and must not start with a
digit; anything else is rejected.

`core` is always included, and any block a selected block requires is added automatically.

```bash
python3 make_project.py your_project_name                         # interactive: asks y/N for each block
python3 make_project.py your_project_name --blocks basic,rf_sta   # explicit list (core is added for you)
python3 make_project.py your_project_name --all                   # every block (except motion_refine, see below)
python3 make_project.py your_project_name --blocks basic --refined-roi-mask   # also the optional motion_refine block
```

`motion_refine` (`RefinedRoiMask`, refines the field ROI mask borders in motion mode) is not asked interactively and not part
of `--all`, because it made no measurable difference to the traces. Force it with `--refined-roi-mask` (or by
listing it in `--blocks`); it requires, and thereby adds, the `motion` block.

Without `--blocks`/`--all`, the script asks interactively; when there is no terminal (e.g. in a script or CI)
it falls back to `core` only. Unknown block names raise an error that lists the available blocks.

Existing files are never overwritten: if a file of the project already exists, the script prints a message and
skips it. To regenerate files with a different selection, move or delete the old ones first.
The project README records the command with every option given explicitly (also the answers given interactively),
so running it again creates the same project.
If `jupytext` is not installed (<code>pip install jupytext</code>), the notebooks are written as percent-format
<code>.py</code> files instead of <code>.ipynb</code>; they can be converted later with
<code>jupytext --to ipynb</code>.

#### Experiment history and listed stimuli (`history` block)

`ExpHistory` registers every ScanM recording of every experiment, independent of stimuli and fields: z-stacks, high
resolution, optic disk and blood vessel recordings, test recordings, recordings that were stopped early and recordings
of stimuli that are not in the `Stimulus` table. It scans the `pre_data_dir` and `raw_data_dir` of `UserInfo` and
the folders in `ExpHistory.extra_data_dirs`, links `.smp`/`.smh` and `.h5` files of the same recording (by the
original file name stored in the `.h5`, the file name, or the time stamp; see `link_method`), and stores the scan
parameters needed for the light exposure history (start, duration, zoom, pixel size, pixel dwell time, stage position,
setup, number of triggers). Laser power and stimulus intensity are not stored digitally. The pixel size is computed
from the dated scan width of `ExperimentImagingSetup` if `ExpHistory.experiment_imaging_setup_table` is bound
(as for Field, Presentation and HighRes), otherwise from the setupid constant; the scan width used is stored in
`scan_width_um`. `ExpHistoryField` links
recordings to fields, `ExpHistoryStimLog` assigns the QDSpy stimuli (log entries) to the recordings during which they
were shown (the clock offset between the QDSpy and ScanM computers is estimated).

A recording is **listed** if the stimulus part of its file name is one of the `_`-separated aliases of a stimulus in
`Stimulus` (`djimaging.utils.stimulus_listing`). Only listed recordings become `Presentation`s, and everything with
stimulus-specific data (`Traces`, `Snippets`, `Averages`, analyses) depends on `Presentation`. So unlisted (e.g.
unpublished) stimuli are visible in the history with their names, but never get traces.
`ExpHistory().listed(Stimulus)` shows the status of every recording.

#### Imaging channels (functional and background)

The imaging channels are defined in a `Channel` table (`ch_id`, `ch_name`, `is_functional`, `roi_default`), which every
generated schema contains. The standard recording has one functional channel (Ch0) and one background channel (Ch1,
e.g. SR-101 as marker for blood vessels):

```bash
python3 make_project.py your_project_name --blocks basic                                   # 1 functional, 1 background
python3 make_project.py your_project_name --n-functional 2 --n-background 0                # two-color imaging
python3 make_project.py your_project_name --legacy-userinfo-stacks                         # old design, see below
```

Without `--n-functional` and `--n-background`, the script asks interactively (default 1 and 1; the defaults are also
used when there is no terminal). The channels are named `wDataCh0`, `wDataCh1`, ... (functional channels first); the
generated `01_setup` notebook fills `Channel` in an extra cell, where you can edit the names and the `functional` list.

- **Functional channels** (`is_functional`): traces are only extracted for these. **Background channels** have no
  traces, but they are loaded in the ROI canvas and the field plots.
- With **one functional channel**, keys, tables and plots are the same as in older schemas. With **several functional
  channels**, the generated schema sets `Traces._use_channels = True`, and `ch_id` becomes part of the primary key of
  `Traces` and of all tables that depend on it (`PreprocessTraces`, `Snippets`, `Averages`, response features, ...),
  so these are computed separately per channel. This changes primary keys, so the number of functional channels
  cannot be changed in a populated schema; create a new schema and repopulate.
- Traces must be computed from the stack (`compute_from_stack=True` in `RawDataParams`) with several functional channels.
- `UserInfo` has `_use_channels = True` and `channel_table = Channel`: the channels replace the columns
  `data_stack_name` and `alt_stack_name`, which are then not part of `UserInfo`.
  Tables that need a main and an alternative stack (field plots, `Roi`, `Brightness`, ...) use the `roi_default`
  channel and the first other channel (`UserInfo().get_stack_names(key)`).

**ROI canvas:** `draw_roi_mask` loads all channels of `Channel`. Every channel is available
as background (`stack: <name>[mean]` / `[std]`). One channel is the *target* for ROI detection (cc tool, diagnostics,
auto shift and AutoROIs); it is preselected by `roi_default` in `Channel`
(`Channel().add_channels(..., roi_channel='wDataCh1')`, default: the first functional channel), can be overridden with
`draw_roi_mask(roi_channel=...)` and changed in the canvas ("Target"). AutoROIs only use the target channel (plus one
other channel for the two-channel U-Net v0.1.0); multi-color AutoROIs are not supported. The ROI mask is shared by
all channels.

**Old schemas** keep working, because the templates default to the old behavior: `UserInfoTemplate._use_channels` and
`TracesTemplate._use_channels` are `False`, so the stacks are defined by `data_stack_name` (functional channel) and
`alt_stack_name` in `UserInfo`. To generate a new schema like this, use `--legacy-userinfo-stacks`.
To use the `Channel` table in a hand-written schema, define `class Channel(core.ChannelTemplate)` and set
`_use_channels = True` and `channel_table = Channel` on `UserInfo` (and on `Traces` for several functional channels,
and `FeaturesTemplate` for clustering features per channel). The `Channel` table and `UserInfo` must be created
together, so this requires a new schema.

Notes:

- Tables that aggregate over ROIs and are not listed above should be checked before use with several functional channels.

#### How blocks work

Every block has two matching parts:

- **Schema part**: <code>djimaging/schemas/blocks/&lt;name&gt;.py</code> subclasses the table templates under
  <code>@schema</code> and binds their abstract table attributes. A block imports the block it builds on
  (e.g. <code>from djimaging.schemas.blocks.core import *</code>), so it is also a valid, importable schema on its own.
- **Notebook part**: <code>djimaging/notebooks/blocks/&lt;name&gt;.py</code> is a percent-format
  (<code># %%</code>) notebook section that starts with a top-level heading (<code># # Title</code>), which becomes
  the title of its notebook. The `notebook` key of the block in <code>BLOCKS</code> says where it goes: `'own'`
  (default) a numbered notebook of its own, `'setup'`, `'rois'` or `'traces'` appended to <code>01_setup</code>,
  <code>02_roi_masks</code> or <code>03_traces</code>. The `core` part is split at its <code># # Core: ROIs</code> and
  <code># # Core: Traces and responses</code> headings into these three notebooks; `stimulator` (before `Field`),
  `motion` (before `Traces`) and `motion_refine` (before `MotionRoiMask`) are inserted at fixed places.
  Every notebook starts with <code>preamble.py</code> (connect, import and activate the schema), <code>01_setup</code>
  also gets <code>overview.py</code> (diagram), and <code>cleanup.py</code> becomes <code>drop_schema</code>.
  Placeholders such as <code>__SCHEMA_MODULE__</code>, <code>__PACKAGE__</code> and <code>__BLOCKS__</code> are filled
  in. Use <code>plot_if_data</code> (from <code>djimaging.utils.dj_utils</code>, imported by the preamble) for plots, so
  that *Run all* works on empty tables. A notebook must only use names it defines itself or gets from the preamble.

To generate your files, <code>make_project.py</code>

1. resolves the selection (adds required blocks, removes duplicates, orders them as in <code>BLOCKS</code>),
2. concatenates the schema parts into one file: the inter-block imports are dropped (the block code is already
   included above), all other imports are de-duplicated, merged per module and moved to the top,
3. builds the numbered notebooks from the notebook parts and converts them to <code>.ipynb</code> via jupytext,
4. writes the project files (README, pyproject, settings, ...); `extras` of the selected blocks become extras of the
   djimaging dependency.

The generated schema starts with a header (blocks, channels, the make_project command) and ends with a
commented-out example showing how to add your own table (see <code>tables/my_table.py</code>).
The ready-made schemas in <code>djimaging/schemas/</code> (<code>core_schema</code>, <code>full_rgc_schema</code>,
<code>rf_glm_schema</code>, <code>tutorial_schema</code>) are still available to copy; <code>core_schema</code>
re-exports the <code>core</code> block.

#### Adding a new block

1. Add the schema part <code>djimaging/schemas/blocks/&lt;name&gt;.py</code>. Start with an import of the blocks it
   depends on, e.g. <code>from djimaging.schemas.blocks.core import *</code>, then define the tables.
   Imports must be top-level, since they are parsed to merge the blocks.
2. Add the notebook part <code>djimaging/notebooks/blocks/&lt;name&gt;.py</code> with the sections that populate and
   inspect your tables, starting with a top-level heading (<code># # Title</code>).
3. Register the block in <code>BLOCKS</code> in <code>make_project.py</code> with a `description` and its `requires`,
   and if needed `notebook` (default `'own'`) and `extras` (extras of djimaging it needs, e.g. `['rf']`).
   The order of that dict is the order in the generated files (and of the numbered notebooks), so list a block after
   the blocks it requires.

### Clean up

When you no longer need your test schema, make sure you drop it by
calling <code>schema.drop()</code> (e.g. in the <code>drop_schema</code> notebook of your project) and confirm by
entering <code>yes</code>.

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
