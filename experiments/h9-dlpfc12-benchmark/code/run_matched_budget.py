#!/usr/bin/env python3
"""H9 matched-budget driver: BayesSpace x5 seeds and SC.MEB x3 seeds on all 12 DLPFC slices.

Usage: /opt/anaconda3/bin/python3 run_matched_budget.py [phase]
  phase: bs | scmeb | all (default all)
"""
import os as _os
import pathlib as _pl


def _find_root(_start):
    _p = _pl.Path(_start).resolve()
    for _anc in [_p] + list(_p.parents):
        if (_anc / "src" / "hmrf_solvers.py").exists():
            return str(_anc)
    raise RuntimeError("project root not found; run inside the repository")


ROOT = _os.environ.get("HMRF_ROOT") or _find_root(__file__)

import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

CODE = _os.path.join(ROOT, "experiments/h9-dlpfc12-benchmark/code")
SLICES = ["151507", "151508", "151509", "151510", "151669", "151670",
          "151671", "151672", "151673", "151674", "151675", "151676"]
BS_SEEDS = (100, 101, 102, 103, 104)
SCMEB_SEEDS = (100, 101, 102, 103, 104)

ENV = {**_os.environ, "HMRF_ROOT": ROOT,
       "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}


def run_one(script, sid, seed, tag):
    t0 = time.time()
    r = subprocess.run(["Rscript", _os.path.join(CODE, script), sid, str(seed)],
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
