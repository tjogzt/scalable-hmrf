# --- portable project root (no hardcoded user paths) ---
import os as _os, pathlib as _pl
def _find_root(_start):
    _p = _pl.Path(_start).resolve()
    for _anc in [_p] + list(_p.parents):
        if (_anc / "src" / "hmrf_solvers.py").exists():
            return str(_anc)
    raise RuntimeError("project root not found; run inside the repository")
_ROOT = _os.environ.get("HMRF_ROOT") or _find_root(__file__)
# ------------------------------------------------------
"""H9 汇总：把 本方法/BayesSpace/SC.MEB/GraphST 的 12 切片结果合并成基准表。
输出:
  results/h9_benchmark_slices.csv  - 每切片每方法的 ARI/NMI/时间
  results/h9_benchmark_summary.csv - 方法级 mean±sd（全体 + 7层子集 + 5层子集）
"""
import os
import numpy as np
import pandas as pd

ROOT = _ROOT
RES = os.path.join(ROOT, "experiments/h9-dlpfc12-benchmark/results")
SEVEN = {"151507", "151508", "151509", "151510", "151673", "151674", "151675", "151676"}

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
print("\n=== 每切片 ARI 透视 ===")
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
print("\n=== 方法级汇总 ===")
print(summ.to_string(index=False))
print("\nAGG DONE")
