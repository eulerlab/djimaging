You are a single agent working on ONE module of the djimaging test campaign. Work alone: do not spawn
subagents, workflows or parallel agents. Finish one module, then stop.

## Goal
Raise unit-test coverage of one module, find real bugs with the new tests, and fix them. Each bug becomes
its own reviewable commit. Match the style of the previous campaign (PR 91, branch `fix/test-found-bugs`).

## Step 0: orient (cheap, do not skip)
1. Read `CLAUDE.md` and `tools/test_campaign.md` (ledger: which modules are done, in progress, or skipped,
   and known suspicions). Do not read anything else yet.
2. Create a branch from the current campaign branch: `git switch -c test/<module-name>`.
3. Use the project venv: `.venv/Scripts/python.exe` (Windows). Never touch the `X:` drive; it is a shared server.

## Step 1: pick the module
- Run `python tools/coverage_gaps.py --reuse` if a `.coverage` file exists, else `python tools/coverage_gaps.py`
  once (about 6 minutes; do this at most once per session). It prints the top modules by missed statements.
- Take the highest-ranked module with ledger status `todo` that you can test WITHOUT a database. Modules whose
  logic lives in DataJoint `make()` methods need MySQL: test only their pure helper functions (or the
  functions they delegate to in `djimaging/utils/`), and skip the rest with a ledger note.
- For a very large module (for example `roi_mask.py`), limit the session to its pure functions. Do not try
  to cover everything.
- Set the module's ledger row to `wip` right away.
- To see what is untested in that module: `python tools/coverage_gaps.py --module <path>` (prints missed line
  ranges, instant). Then read only those line ranges of the module, not the whole file.

## Step 2: write tests
- Put them in `tests/` mirroring the module path, in the style of the neighbouring test files.
- Synthetic data only. No MySQL, no network, no real test data (if you use `DJIMAGING_TEST_DATA`, skip when missing).
- Use the DataJoint 2 API (`fetch1`, `to_dicts`, `to_arrays`); v1 `fetch()` and `ERD` raise errors in pytest.
- Test behaviour from the docstring, the callers and the maths, not from what the code happens to do.
  Check edge cases: empty input, single element, NaN, integer dtype, unequal lengths, off-by-one at
  boundaries, 2-D vs 3-D, numpy-2 scalar conversion (`float()`/`int()` of 1-element arrays).
- Iterate with targeted runs (`pytest tests/path/test_x.py -q`), never the full suite.

## Step 3: handle suspected bugs
For each failing or suspicious behaviour:
1. Reproduce it with a 3-line script. Read the callers and docstring: is this plausibly intended?
2. Confirmed bug: write a test asserting the CORRECT behaviour, marked
   `@pytest.mark.xfail(strict=True, reason="BUG: <one line>")`.
3. Unsure whether it is intended: do not add an xfail. Add one line under "Known suspicions" in
   `tools/test_campaign.md`.
4. Commit all new tests plus xfails as ONE test-only commit (nothing under `djimaging/` changed):
   `tests: <module> helpers`.

## Step 4: fix the bugs
Fix at most 8 bugs per session. If there are more, leave their xfails in place and add a ledger row
`needs-fix` with the count.
- One bug per commit: the smallest possible library change plus removal of that bug's xfail marker.
- Inline comments only where the correct code is not self-evident.
- Commit message format (subject: `<module>: <what is fixed>`):
```
Bug: <what is wrong>
Repro: <input -> observed vs expected>
How to judge: <why this is a bug and not intended behaviour>
Impact: <which tables or functions; crash or silent; would stored values change on repopulation>
Confidence: <high, or medium with the reason it might be intended>

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
```
- If a fix would change behaviour a caller depends on, or the intended behaviour is a judgement call,
  do not fix it: keep the xfail, and note it in the ledger.

## Step 5: verify and finish
1. Run the full unit suite once: `pytest -q tests --ignore=tests/integration --ignore=tests/autorois`.
   Everything must pass; no XPASS (a fixed bug whose marker remains).
2. Run flake8 on the files you changed with CI settings:
   `flake8 <files> --max-complexity=10 --max-line-length=127`.
3. Run `python tools/coverage_gaps.py --module <path>` and record the new coverage.
4. Update the ledger row to `done` (or `skip` with a reason): coverage before and after, number of bugs
   found and fixed. Commit the ledger as `tests: ledger <module>`.
5. Do not push and do not open a PR. Do not commit `.coverage` files. Do not edit existing tests except
   removing the xfail marker of a bug you fixed. Do not change dependencies.

## Budget rules
- One module per session. Stop when the module's pure-function coverage plateaus; do not chase 100%.
- At most one full coverage run (Step 1) and one full test run (Step 5) per session.
- Read line ranges, not whole large files. Do not re-read files you already read.
- Final report to the user, at most 15 lines: module, coverage before to after, bugs found, bugs fixed
  (commit hashes, one line each, with confidence), bugs left as xfail or suspicion, anything blocked.
