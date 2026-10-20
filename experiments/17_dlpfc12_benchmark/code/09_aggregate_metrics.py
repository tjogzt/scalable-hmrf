"""Aggregate per-slice metrics into the manuscript benchmark table.

Author : Tao Zhu <zhutao@tjh.tjmu.edu.cn>
Created: 2026-10-02
Updated: 2026-10-20

Inputs
------
results/h9_benchmark_metrics.csv.
Configuration: config/defaults.yaml, loaded through hmrf.config.

Outputs
-------
results/h9_summary.csv.

Usage
-----
    python 09_aggregate_metrics.py [options]
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

RES = str(results_dir("17_dlpfc12_benchmark"))
SEVEN = set(CFG["experiments"]["h9"]["seven_slices"])

ours = pd.read_csv(os.path.join(RES, "h9_ours_summary.csv"), dtype={"slice": str})

rows = []
for r in ours.itertuples():
    rows.append(dict(slice=r.slice, method="Ours(best)", k="", ari=r.ari_max, nmi=r.nmi_max,
                     bf1=r.bf1_max, t_s=r.t_runs_s + r.t_pipe_s, note="best of 5 inits x 5 betas"))
    rows.append(dict(slice=r.slice, method="Ours(median)", k="", ari=r.ari_median, nmi=np.nan,
                     bf1=np.nan, t_s=r.t_runs_s + r.t_pipe_s, note="median over 25 runs"))
    rows.append(dict(slice=r.slice, method="Oracle", k="", ari=r.oracle, nmi=np.nan,
                     bf1=np.nan, t_s=np.nan, note="kmeans++ on truth, beta=%.1f" % r.oracle_beta))

def add_baseline(path, method):
    if not os.path.exists(path):
        print("MISSING:", path)
        return
    df = pd.read_csv(path, dtype={"slice": str})
    for r in df.itertuples():
        rows.append(dict(slice=r.slice, method=method, k=getattr(r, "k", ""), ari=r.ari,
                         nmi=getattr(r, "nmi", np.nan),
                         bf1=getattr(r, "bf1", np.nan),
                         t_s=getattr(r, "t_s", np.nan), note=""))

add_baseline(os.path.join(RES, "bayesspace/bayesspace_all.csv"), "BayesSpace")
add_baseline(os.path.join(RES, "scmeb/scmeb_all.csv"), "SC.MEB")
add_baseline(os.path.join(RES, "graphst/graphst_all.csv"), "GraphST")

tab = pd.DataFrame(rows)
tab.to_csv(os.path.join(RES, "h9_benchmark_slices.csv"), index=False)
print("\n=== Per-slice ARI pivot ===")
piv = tab.pivot_table(index="slice", columns="method", values="ari")
print(piv.round(4).to_string())

summ = []
for m, g in tab.groupby("method", sort=False):
    rec = dict(method=m, ari_mean=round(g.ari.mean(), 4), ari_sd=round(g.ari.std(), 4),
               ari_med=round(g.ari.median(), 4), nmi_mean=round(g.nmi.mean(), 4) if g.nmi.notna().any() else np.nan,
               t_med=round(g.t_s.median(), 1) if g.t_s.notna().any() else np.nan, n=len(g))
    g7 = g[g.slice.isin(SEVEN)]; g5 = g[~g.slice.isin(SEVEN)]
    rec["ari7_mean"] = round(g7.ari.mean(), 4); rec["ari5_mean"] = round(g5.ari.mean(), 4)
    summ.append(rec)
summ = pd.DataFrame(summ)
summ.to_csv(os.path.join(RES, "h9_benchmark_summary.csv"), index=False)
print("\n=== Method-level summary ===")
print(summ.to_string(index=False))
print("\nAGG DONE")
