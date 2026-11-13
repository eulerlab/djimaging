# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`djimaging` is a library of reusable **DataJoint 2** table templates (and helpers) for 2-photon imaging pipelines of the Euler lab. It targets a shared MySQL 8 server; there is no local DB setup. Python >=3.10, dependencies managed with [uv](https://docs.astral.sh/uv/) (`uv.lock`, `.venv` present).

## Commands

```bash
uv pip install -e .                  # install editable; extras: ".[autorois]", ".[rf]", ".[dev]"
pytest -v tests --ignore=tests/integration       # unit tests (what the python-app CI runs; also runs flake8)
pytest tests/utils/test_x.py::test_name          # single test
DJ_TEST_MYSQL=1 pytest -q tests/integration      # MySQL integration tests; needs a MySQL server
                                                 # (env DJ_HOST/DJ_PORT/DJ_USER/DJ_PASS, defaults 127.0.0.1:3306 root/datajoint)
python3 make_project.py <project> [--blocks basic,rf_sta | --all] [--target-dir PATH]   # standalone project (default ../djimaging-projects/<project>): installable package dji_<project> with schema, settings.py and numbered notebooks 01_setup, 02_roi_masks, 03_traces, 04_<block>... built from the chosen table blocks; never overwrites files
```

`pyproject.toml` turns DataJoint v1-style `fetch()` and `ERD` deprecation warnings into **errors** in pytest — use the v2 API (`fetch1`, `to_dicts`/`to_arrays`, etc.).

## Architecture

- **`djimaging/tables/<area>/`** — the core of the repo: table *templates* (e.g. `core`, `response`, `receptivefield`, `classifier`, `classifier_v2`, `clustering`, `color_flicker`, `ipl`, `location`, `motion_correction`, `spike_estimation`, `misc`). Each is an abstract `dj.Manual/Imported/Computed/Lookup` subclass named `*Template`. Its `definition` is a property, and upstream dependencies are declared as abstract properties (e.g. `field_table`, `userinfo_table`) that are filled in by the concrete schema. `_new_table_example.py` shows the canonical pattern (`make` fetches via `fetch1`, delegates computation to a function outside `make`, then inserts).
- **`djimaging/schemas/`** — *example* schemas that wire templates together by subclassing them under `@schema` and binding the abstract table attributes (`core_schema`, `full_rgc_schema`, `rf_glm_schema`, `tutorial_schema`). Users copy these into their own project, or generate a custom schema with `make_project.py` from the table blocks in `djimaging/schemas/blocks/` (`core`, `history`, `motion`, `motion_refine`, `basic`, `location`, `classifier`, `rf_sta`, `rf_glm`, `stimulator`, `mouse_cam`, `mb_direction`, `rf_offset`; `core_schema` re-exports `blocks/core`). The `history` block (`ExpHistory`, `ExpHistoryField`, `ExpHistoryStimLog`, QDSpy logs) registers every recording, independent of stimuli; whether a recording is *listed* (may become a `Presentation` and get traces) is decided only by `djimaging.utils.stimulus_listing`. The optional `motion_refine` block (requires `motion`) inserts `RefinedRoiMask` before `MotionRoiMask` and sets `MotionRoiMask.refined_roi_mask_table`; without it `MotionRoiMask` starts from the drawn `RoiMask` mask. The `motion` block switches core into motion mode by code substitution in `make_project.py`: motion tables are inserted before `Traces`, `RoiMask` gets `_presentation_masks = False` (no `RoiMaskPresentation`; `MotionRoiMask` replaces it) and `Traces` gets `_include_motion_correction = True`; method and validation in `docs/motion_qc.md`. Each schema block has a matching percent-format notebook part in `djimaging/notebooks/blocks/` (plus `preamble`, `overview`, `cleanup`), so the generated notebooks only contain cells for tables in the schema: `core` is split into `01_setup`/`02_roi_masks`/`03_traces`, other blocks go where their `notebook` key in `BLOCKS` says (`'own'` = a numbered notebook of their own, titled by the part's first `# # ` heading); every notebook must be runnable on its own (shared helpers like `plot_if_data` come from `djimaging.utils.dj_utils` via the preamble). Register new blocks in `BLOCKS` in `make_project.py` (`description`, `requires`, optional `notebook`, `extras`); schemas are created with `dj.Schema()` and activated later through `djimaging.utils.dj_utils.activate_schema`. The core chain is UserInfo → RawDataParams → Experiment → Field → Stimulus/Presentation → ROI/Traces → Snippets/Averages → analyses.
- **Setup-dependent parameters** (stimulator delay, scan width, light artifact rows) come from `Stimulator`/`ImagingSetup` via `ExperimentStimulator`/`ExperimentImagingSetup` when bound (`RecordingSetupMixin` in `tables/core/setup_params.py`; `make_project.py` inserts the `stimulator` block before `Field`), else from the `setupid` constants in `utils/scanm/setup_utils.py`.
- **Stimulus geometry** comes only from `Stimulator`/`ExperimentStimulator` (`djimaging/tables/stimulator/`, conventions and documented setup transforms in `djimaging/utils/stimulator.py`); no `setupid` branches for stimulus geometry. Noise stimuli must declare `stim_dict['array_to_screen']`.
- **`djimaging/utils/`** — computation and IO helpers kept separate from tables (trace/snippet/filter/mask utils, `motion/` (registration, structural-channel cleaning, z proxy, trust, per-ROI shifts), `scanm/` and `qdspy/` file readers, `receptive_fields/`, `dj_storage.py` for object/array references, `dj_utils.py`).
- **`djimaging/autorois/`** — optional ROI detection (U-Net, finetuned Cellpose3, MEDIAR, correlation maps, autoshift); needs the `autorois` extra; the Cellpose3 and MEDIAR models additionally need `cellpose` / `mediar`, which are deliberately not dependencies (numpy conflicts, see README). The default model set is only the latest UNet (+ CorrRoiMask); others must be requested via `load_default_autorois_models(models=[...])`. Training code lives in the separate `autorois` repo; here is inference only. Model weights (`Resources/AutoROIs/models/{UNET,CELLPOSE3,MEDIAR}_v0.2.0`) are in the `reference` store.
- **`djimaging/user/`** — legacy personal project folders (git-ignored, imported as `djimaging.user.<project>`); still supported, but `make_project.py` now creates standalone projects outside the repo.

## DataJoint 2 storage conventions

Configured via a JSON config (template: `djimaging/djconfig/djconf_template.json`, never commit personal configs). Store roles:
- `reference` store (`<filepath@reference>`): read-only references to externally managed acquisition/stimulus/model files; paths must be relative to the common reference root.
- Small arrays/objects (traces, snippets, masks, small analyses) → `<blob>` (in the DB).
- Large arrays/objects (stimulus/population matrices, full STRFs, correlation maps, models) → `<npy@processed>` / `<blob@processed>` in the writable `processed` store.

## Notes

- `scikit-learn` is pinned to `==1.7.2` and numpy >=2 because saved classifier pickles depend on them; don't bump casually.
- `scanm-support` is installed from a git URL (see `[tool.uv.sources]`).
- Main branch of the repo for PRs is `main`.
