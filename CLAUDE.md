# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`djimaging` is a library of reusable **DataJoint 2** table templates (and helpers) for 2-photon imaging pipelines of the Euler lab. It targets a shared MySQL 8 server; there is no local DB setup. Python >=3.10, dependencies managed with [uv](https://docs.astral.sh/uv/) (`uv.lock`, `.venv` present).

## Commands

```bash
uv pip install -e .                  # install editable; extras: ".[autorois]", ".[autorois-cellpose]", ".[autorois-mediar]", ".[rf]", ".[dev]"
pytest -v tests --ignore=tests/integration       # unit tests (what the python-app CI runs; also runs flake8)
pytest tests/utils/test_x.py::test_name          # single test
DJ_TEST_MYSQL=1 pytest -q tests/integration      # MySQL integration tests; needs a MySQL server
                                                 # (env DJ_HOST/DJ_PORT/DJ_USER/DJ_PASS, defaults 127.0.0.1:3306 root/datajoint)
python3 make_user.py <username> [--blocks basic,rf_sta | --all]   # scaffold git-ignored djimaging/user/<username> with a schema + tutorial .ipynb built from the chosen table blocks (needs jupytext)
```

`pyproject.toml` turns DataJoint v1-style `fetch()` and `ERD` deprecation warnings into **errors** in pytest — use the v2 API (`fetch1`, `to_dicts`/`to_arrays`, etc.).

## Architecture

- **`djimaging/tables/<area>/`** — the core of the repo: table *templates* (e.g. `core`, `response`, `receptivefield`, `classifier`, `classifier_v2`, `clustering`, `color_flicker`, `ipl`, `location`, `motion_correction`, `spike_estimation`, `misc`). Each is an abstract `dj.Manual/Imported/Computed/Lookup` subclass named `*Template`. Its `definition` is a property, and upstream dependencies are declared as abstract properties (e.g. `field_table`, `userinfo_table`) that are filled in by the concrete schema. `_new_table_example.py` shows the canonical pattern (`make` fetches via `fetch1`, delegates computation to a function outside `make`, then inserts).
- **`djimaging/schemas/`** — *example* schemas that wire templates together by subclassing them under `@schema` and binding the abstract table attributes (`core_schema`, `full_rgc_schema`, `rf_glm_schema`, `tutorial_schema`). Users copy these into their own project, or generate a custom schema with `make_user.py` from the table blocks in `djimaging/schemas/blocks/` (`core`, `basic`, `location`, `classifier`, `rf_sta`, `rf_glm`; `core_schema` re-exports `blocks/core`). Each schema block has a matching percent-format notebook part in `djimaging/notebooks/blocks/` (plus `cleanup`), so the generated notebook only contains cells for tables in the schema. Register new blocks in `BLOCKS` in `make_user.py`; schemas are created with `dj.Schema()` and activated later through `djimaging.utils.dj_utils.activate_schema`. The core chain is UserInfo → RawDataParams → Experiment → Field → Stimulus/Presentation → ROI/Traces → Snippets/Averages → analyses.
- **`djimaging/utils/`** — computation and IO helpers kept separate from tables (trace/snippet/filter/mask utils, `scanm/` and `qdspy/` file readers, `receptive_fields/`, `dj_storage.py` for object/array references, `dj_utils.py`).
- **`djimaging/autorois/`** — optional ROI detection (U-Net, finetuned Cellpose3, MEDIAR, correlation maps, autoshift); needs the `autorois` extra, plus `autorois-cellpose` / `autorois-mediar` for those models. The default model set is only the latest UNet (+ CorrRoiMask); others must be requested via `load_default_autorois_models(models=[...])`. Training code lives in the separate `autorois` repo; here is inference only. Model weights (`Resources/AutoROIs/models/{UNET,CELLPOSE3,MEDIAR}_v0.2.0`) are in the `reference` store.
- **`djimaging/user/`** — personal per-user folders, not under version control.

## DataJoint 2 storage conventions

Configured via a JSON config (template: `djimaging/djconfig/djconf_template.json`, never commit personal configs). Store roles:
- `reference` store (`<filepath@reference>`): read-only references to externally managed acquisition/stimulus/model files; paths must be relative to the common reference root.
- Small arrays/objects (traces, snippets, masks, small analyses) → `<blob>` (in the DB).
- Large arrays/objects (stimulus/population matrices, full STRFs, correlation maps, models) → `<npy@processed>` / `<blob@processed>` in the writable `processed` store.

## Notes

- `scikit-learn` is pinned to `==1.7.2` and numpy >=2 because saved classifier pickles depend on them; don't bump casually.
- `scanm-support` is installed from a git URL (see `[tool.uv.sources]`).
- Main branch of the repo for PRs is `master`.
