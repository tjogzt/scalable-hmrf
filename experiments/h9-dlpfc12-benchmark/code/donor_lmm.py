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
"""Donor-blocked re-analysis of the 12-slice paired comparisons (review R8, R1 W7).

The 12 DLPFC sections come from three donors (4 serial sections each):
  Br5292 = {151507..151510}, Br5595 = {151669..151672}, Br8100 = {151673..151676}
(Maynard et al. 2021; spatialLIBD). Sections are treated as replicates in the paper's
Wilcoxon tests; this script reports donor-level means, per-donor paired differences,
and a linear mixed model with donor as a random intercept.

Usage: python donor_lmm.py
"""
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

RES = _os.path.join(_ROOT, "experiments/h9-dlpfc12-benchmark/results/h9_benchmark_metrics.csv")
DONOR = {"151507": "Br5292", "151508": "Br5292", "151509": "Br5292", "151510": "Br5292",
         "151669": "Br5595", "151670": "Br5595", "151671": "Br5595", "151672": "Br5595",
         "151673": "Br8100", "151674": "Br8100", "151675": "Br8100", "151676": "Br8100"}

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

out = _os.path.join(_ROOT, "experiments/h9-dlpfc12-benchmark/results/donor_lmm.csv")
pd.DataFrame(out_rows).to_csv(out, index=False)
print("\nDONE ->", out)
