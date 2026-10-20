"""Donor-blocked linear mixed model for method differences.

Author : Tao Zhu <zhutao@tjh.tjmu.edu.cn>
Created: 2026-10-02
Updated: 2026-10-20

Inputs
------
results/h9_benchmark_metrics.csv.
Configuration: config/defaults.yaml, loaded through hmrf.config.

Outputs
-------
Console LMM statistics.

Usage
-----
    python 12_donor_lmm.py [options]
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

import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

RES = str(results_dir("17_dlpfc12_benchmark") / "h9_benchmark_metrics.csv")
DONOR = CFG["datasets"]["dlpfc12"]["donors"]

df = pd.read_csv(RES, dtype={"slice": str})
df["donor"] = df["slice"].map(DONOR)
df = df[df["method"] != "Oracle"].copy()  # Oracle is the reference, not a comparator
base = [m for m in df["method"].unique() if "Ours" in m][0]
baselines = [m for m in df["method"].unique() if m != base]
print("methods:", list(df["method"].unique()), "| reference method:", base)

out_rows = []
for metric in ("ari", "nmi", "bf1"):
    sub = df[["slice", "method", "donor", metric]].rename(columns={metric: "y"})
    pt = sub.pivot_table(index="donor", columns="method", values="y", aggfunc="mean").round(4)
    print(f"\n===== {metric}: per-donor means =====")
    print(pt.to_string())
    # per-donor paired diffs vs the reference method
    wide = sub.pivot_table(index="slice", columns="method", values="y")
    wide["donor"] = [DONOR[s] for s in wide.index]
    for m in baselines:
        diff = wide[base] - wide[m]
        per_donor = diff.groupby(wide["donor"]).mean().round(4)
        n_pos = int((diff > 0).sum())
        print(f"  vs {m}: per-donor mean diff = {dict(per_donor)} | slices positive {n_pos}/12")
        for donor, v in per_donor.items():
            out_rows.append(dict(metric=metric, comparison=f"ours-{m}", donor=donor,
                                 mean_diff=round(float(v), 4), n_slices=4))
    # LMM: y ~ method + (1|donor)
    model = smf.mixedlm(f"y ~ C(method, Treatment('{base}'))", sub, groups=sub["donor"])
    try:
        fit = model.fit(reml=True)
        params = fit.params
        print(f"  LMM (donor random intercept):")
        for name, val in params.items():
            if name == "Group Var":
                continue
            p = fit.pvalues.get(name, float("nan"))
            print(f"    {name}: beta={val:+.4f} (p={p:.3f})")
            out_rows.append(dict(metric=metric, comparison=f"lmm:{name}", donor="ALL",
                                 mean_diff=round(float(val), 4), n_slices=12, p=round(float(p), 4)))
    except Exception as e:  # noqa: BLE001
        print("  LMM failed:", e)

out = os.path.join(_ROOT, "experiments/17_dlpfc12_benchmark/results/donor_lmm.csv")
pd.DataFrame(out_rows).to_csv(out, index=False)
print("\nDONE ->", out)
