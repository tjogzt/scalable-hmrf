"""Matched-restart-budget comparison (BayesSpace/SC-MEB, five seeds).

Author : Tao Zhu <zhutao@tjh.tjmu.edu.cn>
Created: 2026-10-02
Updated: 2026-10-20

Inputs
------
data/dlpfc12/<slice>/ and the R baseline scripts.
Configuration: config/defaults.yaml, loaded through hmrf.config.

Outputs
-------
results/matched_budget/ CSVs.

Usage
-----
    python 07_run_matched_budget.py [options]
"""

import os
import sys
from pathlib import Path

for _parent in Path(__file__).resolve().parents:
    if (_parent / "config" / "defaults.yaml").exists():
        REPO_ROOT = _parent
        break
else:  # pragma: no cover - repository layout guard
    raise RuntimeError("repository root not found: config/defaults.yaml missing")

sys.path.insert(0, str(REPO_ROOT / "src"))

from hmrf.config import load_config  # noqa: E402
from hmrf.paths import data_dir, results_dir  # noqa: E402

CFG = load_config()
_ROOT = str(REPO_ROOT)  # backwards-compatible alias used by legacy code paths

import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

CODE = os.path.join(ROOT, "experiments/17_dlpfc12_benchmark/code")
SLICES = CFG["datasets"]["dlpfc12"]["slices"]
BS_SEEDS = tuple(CFG["experiments"]["h9"]["bayesspace_seeds"])
SCMEB_SEEDS = tuple(CFG["experiments"]["h9"]["scmeb_seeds"])

ENV = {**os.environ, "HMRF_ROOT": ROOT,
       "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}


def run_one(script, sid, seed, tag):
    t0 = time.time()
    r = subprocess.run(["Rscript", os.path.join(CODE, script), sid, str(seed)],
                       capture_output=True, text=True, env=ENV, cwd=ROOT)
    dt = time.time() - t0
    ok = r.returncode == 0
    line = (r.stdout.strip().splitlines() or [""])[-1]
    return {"tag": tag, "ok": ok, "rc": r.returncode, "dt": round(dt, 1),
            "out": line, "err": "" if ok else r.stderr[-400:]}


def phase_bs():
    jobs = [("run_bayesspace_multi.R", sid, s, f"bs-{sid}-{s}")
            for sid in SLICES for s in BS_SEEDS]
    results = []
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = [ex.submit(run_one, *j) for j in jobs]
        for f in as_completed(futs):
            r = f.result()
            results.append(r)
            print(f"[{r['dt']}s] {r['tag']} ok={r['ok']} {r['out']}", flush=True)
    return results


def phase_scmeb():
    jobs = [("run_scmeb_multi.R", sid, s, f"scmeb-{sid}-{s}")
            for sid in SLICES for s in SCMEB_SEEDS]
    results = []
    with ThreadPoolExecutor(max_workers=3) as ex:  # each SC.MEB uses 5 internal cores
        futs = [ex.submit(run_one, *j) for j in jobs]
        for f in as_completed(futs):
            r = f.result()
            results.append(r)
            print(f"[{r['dt']}s] {r['tag']} ok={r['ok']} {r['out']}", flush=True)
    return results


def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    allr = []
    if phase in ("bs", "all"):
        print("== phase: BayesSpace x5 seeds ==", flush=True)
        allr += phase_bs()
    if phase in ("scmeb", "all"):
        print("== phase: SC.MEB x3 seeds ==", flush=True)
        allr += phase_scmeb()
    fails = [r for r in allr if not r["ok"]]
    print(f"\nDONE matched-budget: {len(allr)} jobs, {len(fails)} failures, "
          f"wall={round(time.time()-t0,1)}s", flush=True)
    for r in fails:
        print("FAIL:", r["tag"], r["rc"], r["err"][:300], flush=True)


if __name__ == "__main__":
    main()
