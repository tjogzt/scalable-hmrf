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
"""Matched-budget comparison (review R3): our method vs BayesSpace (5 seeds) and
SC.MEB (3 seeds) under equal restart budgets, plus donor-blocked views.

Ours: beta=0.7, 5 k-means++ inits (the same 5-restart budget) from h9_ours_runs.csv.
Baselines: results/matched_budget/{bayesspace,scmeb}/*_seed*_result.csv.
Outputs: results/matched_budget/matched_budget_summary.csv + printed tables.
Usage: python analyze_matched_budget.py
"""
import glob
import sys

import numpy as np
import pandas as pd
from scipy import stats

RES = _os.path.join(_ROOT, "experiments/h9-dlpfc12-benchmark/results")
SLICES = ["151507", "151508", "151509", "151510", "151669", "151670",
          "151671", "151672", "151673", "151674", "151675", "151676"]
DONOR = {"151507": "Br5292", "151508": "Br5292", "151509": "Br5292", "151510": "Br5292",
         "151669": "Br5595", "151670": "Br5595", "151671": "Br5595", "151672": "Br5595",
         "151673": "Br8100", "151674": "Br8100", "151675": "Br8100", "151676": "Br8100"}

rows = []
for m in ("bayesspace", "scmeb"):
    for f in glob.glob(_os.path.join(RES, "matched_budget", m, "*_seed*_result.csv")):
        rows.append(pd.read_csv(f, dtype={"slice": str}))
mb = pd.concat(rows, ignore_index=True)

ours = pd.read_csv(_os.path.join(RES, "h9_ours_runs.csv"), dtype={"slice": str})
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

out = _os.path.join(RES, "matched_budget", "matched_budget_summary.csv")
tab.to_csv(out, index=False)
print("\nDONE ->", out)
