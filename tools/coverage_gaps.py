"""Compact coverage-gap report for the test campaign (see tools/test_campaign.md).

Usage (from the repo root, with the project venv active):
    python tools/coverage_gaps.py              # run unit tests under coverage (~2-3 min), print the top gaps
    python tools/coverage_gaps.py --reuse      # skip the test run, reuse the last .coverage data
    python tools/coverage_gaps.py --module djimaging/utils/snippet_utils.py
                                               # missed line ranges of ONE module (reuses .coverage data)
    python tools/coverage_gaps.py --module djimaging/utils/snippet_utils.py --tests tests/utils/test_snippet_utils.py
                                               # seconds. Prints the module's coverage before (last full run), from the
                                               # new tests alone, and combined (= the "after" figure). Never touches .coverage
    python tools/coverage_gaps.py --top 25 --all   # also list modules the ledger marks done/skip

Output is intentionally short: one line per module, ranked by missed statements.
Modules marked `done` or `skip` in tools/test_campaign.md are hidden unless --all is given.
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "tools" / "test_campaign.md"
EXCLUDE = ("djimaging/user/", "djimaging/notebooks/", "djimaging/autorois/", "djimaging/djconfig/",
           "djimaging/schemas/", "_new_table_example", "__init__.py")
DATA_FILE = ROOT / ".coverage"
JSON_FILE = Path(tempfile.gettempdir()) / "djimaging_cov_report.json"  # not .coverage.*: coverage treats that as data


def run_tests():
    cmd = [sys.executable, "-m", "coverage", "run", "--source=djimaging", "-m", "pytest", "-q", "-x",
           "tests", "--ignore=tests/integration", "--ignore=tests/autorois", "-p", "no:cacheprovider"]
    print("running:", " ".join(cmd[2:]), file=sys.stderr)
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    print((res.stdout.strip().splitlines() or [""])[-1], file=sys.stderr)
    if res.returncode != 0:
        print(res.stdout[-3000:], file=sys.stderr)
        sys.exit("test run failed; fix the failure (or run with --reuse after a successful run)")


def to_json(data_file=DATA_FILE):
    subprocess.run([sys.executable, "-m", "coverage", "json", "-q", f"--data-file={data_file}", "-o", str(JSON_FILE)],
                   cwd=ROOT, check=True)
    return json.loads(JSON_FILE.read_text())["files"]


def run_targeted(tests):
    """Run only `tests` under coverage into a throw-away data file; return that file."""
    data_file = Path(tempfile.gettempdir()) / "djimaging_cov_targeted"
    data_file.unlink(missing_ok=True)
    cmd = [sys.executable, "-m", "coverage", "run", f"--data-file={data_file}", "--source=djimaging",
           "-m", "pytest", "-q", "-p", "no:cacheprovider", *tests]
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    print((res.stdout.strip().splitlines() or [""])[-1], file=sys.stderr)
    if res.returncode != 0:
        print(res.stdout[-3000:], file=sys.stderr)
        sys.exit("targeted tests failed")
    return data_file


def ledger_status():
    status = {}
    if LEDGER.exists():
        for line in LEDGER.read_text(encoding="utf-8").splitlines():
            m = re.search(r"(djimaging/[\w/]+\.py)\s*\|\s*(done|skip|todo|wip)", line)
            if m:
                status[m.group(1)] = m.group(2)
    return status


def ranges(lines):
    out, start, prev = [], None, None
    for n in sorted(lines):
        if start is None:
            start = prev = n
        elif n == prev + 1:
            prev = n
        else:
            out.append((start, prev)); start = prev = n
    if start is not None:
        out.append((start, prev))
    return ",".join(f"{a}" if a == b else f"{a}-{b}" for a, b in out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--module")
    ap.add_argument("--tests", nargs="+", help="with --module: measure coverage from only these test files")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()

    if a.tests and not a.module:
        sys.exit("--tests needs --module")
    before = None
    if a.tests:
        new_file = run_targeted(a.tests)
        data_file = new_file
        if DATA_FILE.exists():
            before = {k.replace("\\", "/"): v for k, v in to_json(DATA_FILE).items()}
            comb = Path(tempfile.gettempdir()) / "djimaging_cov_combined"
            comb.unlink(missing_ok=True)
            import shutil
            base_copy = Path(tempfile.gettempdir()) / "djimaging_cov_baseline"
            shutil.copyfile(DATA_FILE, base_copy)
            subprocess.run([sys.executable, "-m", "coverage", "combine", "-q", "--keep", f"--data-file={comb}",
                            str(base_copy), str(new_file)], cwd=ROOT, check=True)
            data_file = comb
    else:
        data_file = DATA_FILE
        if not a.reuse and not a.module:
            run_tests()
        elif not DATA_FILE.exists():
            sys.exit("no .coverage data yet: run `python tools/coverage_gaps.py` once first")
    files = {k.replace("\\", "/"): v for k, v in to_json(data_file).items()}

    if a.module:
        f = files.get(a.module.replace("\\", "/"))
        if not f:
            sys.exit(f"{a.module} not in coverage data")
        s = f["summary"]
        if before is not None:
            b = before.get(a.module.replace("\\", "/"), {}).get("summary")
            if b:
                print(f"before (last full run): {b['percent_covered']:.0f}% ({b['missing_lines']} missed)")
            alone = json.loads(subprocess.run([sys.executable, "-m", "coverage", "json", "-q", f"--data-file={new_file}",
                                               "-o", "-"], cwd=ROOT, capture_output=True, text=True).stdout)
            af = {k.replace("\\", "/"): v for k, v in alone["files"].items()}.get(a.module.replace("\\", "/"))
            if af:
                print(f"new tests alone: {af['summary']['percent_covered']:.0f}%")
            print("after (combined): ", end="")
        print(f"{a.module}: {s['percent_covered']:.0f}% ({s['missing_lines']} of {s['num_statements']} statements missed)")
        print("missed lines:", ranges(f["missing_lines"]))
        return

    status = ledger_status()
    rows = []
    for name, f in files.items():
        if any(x in name for x in EXCLUDE):
            continue
        s = f["summary"]
        if s["missing_lines"] == 0:
            continue
        st = status.get(name, "todo")
        if st in ("done", "skip") and not a.all:
            continue
        rows.append((s["missing_lines"], s["percent_covered"], s["num_statements"], name, st))
    rows.sort(reverse=True)
    tot = sum(f["summary"]["num_statements"] for n, f in files.items() if not any(x in n for x in EXCLUDE))
    miss = sum(f["summary"]["missing_lines"] for n, f in files.items() if not any(x in n for x in EXCLUDE))
    print(f"in scope: {100 * (tot - miss) / tot:.0f}% covered, {miss} statements missed. Top {a.top} modules by missed statements:")
    print(f"{'missed':>6} {'cov%':>5} {'stmts':>5}  module (ledger status)")
    for m, p, n, name, st in rows[:a.top]:
        print(f"{m:>6} {p:>4.0f}% {n:>5}  {name} ({st})")


if __name__ == "__main__":
    main()
