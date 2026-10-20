"""Analyze the matched-budget runs (best-of-5 vs median).

Author : Tao Zhu <zhutao@tjh.tjmu.edu.cn>
Created: 2026-10-02
Updated: 2026-10-20

Inputs
------
results/matched_budget/ CSVs.
Configuration: config/defaults.yaml, loaded through hmrf.config.

Outputs
-------
Console summary and results/matched_budget_summary.csv.

Usage
-----
    python 10_analyze_matched_budget.py [options]
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

import glob
import sys

import numpy as np
import pandas as pd
from scipy import stats

RES = str(results_dir("17_dlpfc12_benchmark"))
SLICES = CFG["datasets"]["dlpfc12"]["slices"]
DONOR = CFG["datasets"]["dlpfc12"]["donors"]

rows = []
for m in ("bayesspace", "scmeb"):
    for f in glob.glob(os.path.join(RES, "matched_budget", m, "*_seed*_result.csv")):
        rows.append(pd.read_csv(f, dtype={"slice": str}))
mb = pd.concat(rows, ignore_index=True)

ours = pd.read_csv(os.path.join(RES, "h9_ours_runs.csv"), dtype={"slice": str})
ours = ours[ours["beta"] == 0.7].drop_duplicates(subset=["slice", "init"], keep="first")


def agg(df, val="ari"):
    return df.groupby("slice")[val].agg(best="max", median="median").reset_index()


ours_a = agg(ours).rename(columns={"best": "ours_best5", "median": "ours_med5"})
bs_a = agg(mb[mb["method"] == "BayesSpace"]).rename(columns={"best": "bs_best5", "median": "bs_med5"})
sc_a = agg(mb[mb["method"] == "SC.MEB"]).rename(columns={"best": "sc_best3", "median": "sc_med3"})
tab = ours_a.merge(bs_a, on="slice").merge(sc_a, on="slice")
tab["donor"] = tab["slice"].map(DONOR)

print("== per-slice (matched budgets) ==")
print(tab.round(4).to_string(index=False))

print("\n== paired Wilcoxon over 12 slices (ours best-of-5 vs baseline best-of-B) ==")
for col, name in (("bs_best5", "vs BayesSpace best-of-5"), ("sc_best3", "vs SC.MEB best-of-3")):
    d = (tab["ours_best5"] - tab[col]).to_numpy()
    w = stats.wilcoxon(d)
    print(f"{name}: mean diff {d.mean():+.4f}, median {np.median(d):+.4f}, "
          f"wins {int((d > 0).sum())}/12, Wilcoxon W={w.statistic:.1f}, p={w.pvalue:.4f}")

print("\n== per-donor mean diffs (ours best-of-5 - baseline) ==")
for col in ("bs_best5", "sc_best3"):
    dd = tab.groupby("donor").apply(lambda g: (g["ours_best5"] - g[col]).mean())
    print(f"{col}: " + ", ".join(f"{k} {v:+.4f}" for k, v in dd.items()))

print("\n== BayesSpace seed dispersion (5 seeds per slice) ==")
disp = (mb[mb["method"] == "BayesSpace"].groupby("slice")["ari"]
        .agg(["min", "max", "mean", "std"]).round(4))
print(disp.to_string())

out = os.path.join(RES, "matched_budget", "matched_budget_summary.csv")
tab.to_csv(out, index=False)
print("\nDONE ->", out)
