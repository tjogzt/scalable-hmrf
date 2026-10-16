# --- portable project root (no hardcoded user paths) ---
import os as _os
import pathlib as _pl


def _find_root(_start):
    _p = _pl.Path(_start).resolve()
    for _anc in [_p] + list(_p.parents):
        if (_anc / "src" / "hmrf_solvers.py").exists():
            return str(_anc)
    raise RuntimeError("project root not found; run inside the repository")


_ROOT = _os.environ.get("HMRF_ROOT") or _find_root(__file__)
# ------------------------------------------------------
"""H2d timing/iteration audit (review R6): iteration counts, per-iteration wall time,
cap-hitting runs, and a per-iteration cost fit. Produces the data backing the
revised Table 1 runtime rows and its footnote.

Usage: python timing_analysis.py
"""
import sys

import numpy as np
import pandas as pd

RES = _os.path.join(_ROOT, "experiments/h2d-multiseed-metrics/results/h2d_results.csv")
OUT = _os.path.join(_ROOT, "experiments/h2d-multiseed-metrics/results/h2d_timing_table.csv")

df = pd.read_csv(RES)
df["per_iter_ms"] = df["t_s"] / df["iters"] * 1e3

rows = []
for S, g in df.groupby("S"):
    per_inst = g.groupby("inst_seed")["t_s"].sum()  # 3-restart total per instance
    rows.append(dict(
        S=S, n=S * S, runs=len(g),
        iters_mean=round(g["iters"].mean(), 1), iters_sd=round(g["iters"].std(ddof=1), 1),
        iters_min=int(g["iters"].min()), iters_max=int(g["iters"].max()),
        cap_runs=int((g["iters"] >= 300).sum()),
        t3_median=round(float(per_inst.median()), 2), t3_min=round(float(per_inst.min()), 2),
        t3_max=round(float(per_inst.max()), 2),
        per_iter_ms_median=round(float(g["per_iter_ms"].median()), 2),
        per_iter_ms_max=round(float(g["per_iter_ms"].max()), 2),
        ari_max=round(float(g["ari"].max()), 4),
    ))

tab = pd.DataFrame(rows)
print(tab.to_string(index=False))
tab.to_csv(OUT, index=False)

# per-iteration cost fit (linear through origin)
x = df["n"].to_numpy(dtype=float)
y = df["per_iter_ms"].to_numpy(dtype=float)
a = float((x * y).sum() / (x * x).sum())
yhat = a * x
r2 = 1.0 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
print(f"\nper-iteration cost fit: {a:.3e} ms/spot (through origin), R^2 = {r2:.4f}")
print("Hardware: Apple M4 Max (16 CPU cores), 128 GB RAM, macOS; single process, NumPy/SciPy (Accelerate BLAS), no GPU.")

# cap-hitting detail (the 562.5K footnote)
cap = df[(df["S"] == 750) & (df["iters"] >= 300)]
print(f"\nS=750 cap-hitting runs: {len(cap)} of {len(df[df['S'] == 750])}")
if len(cap):
    print(cap[["inst_seed", "restart", "t_s", "iters", "ari"]].to_string(index=False))
print("DONE")
