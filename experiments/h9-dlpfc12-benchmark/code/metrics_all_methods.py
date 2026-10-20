"""A1+A2: Unified metrics (ARI / NMI / boundary-F1) for ALL methods on the 12 DLPFC slices,
plus paired Wilcoxon signed-rank tests (ours vs each baseline).

- Uses the SAME metric definitions as the main pipeline (run_ours.py: boundary_f1_edges, nmi).
- Label sources (all 12 slices):
  ours(best)  : results/labels/<sid>.npz  ['best']
  oracle      : results/labels/<sid>.npz  ['oracle']
  BayesSpace  : results/bayesspace/<sid>_labels.csv  (barcode,pred)
  SC.MEB      : results/scmeb/<sid>_labels.csv       (index,pred)
  GraphST     : results/graphst/<sid>_labels.csv     (barcode,pred)
Outputs: results/h9_benchmark_metrics.csv, results/h9_benchmark_stats.csv
"""
import os as _os, pathlib as _pl
def _find_root(_start):
    _p = _pl.Path(_start).resolve()
    for _anc in [_p] + list(_p.parents):
        if (_anc / "src" / "hmrf_solvers.py").exists():
            return str(_anc)
    raise RuntimeError("project root not found; run inside the repository")
_ROOT = _os.environ.get("HMRF_ROOT") or _find_root(__file__)
# ------------------------------------------------------
import sys, csv, gzip
import numpy as np
import pandas as pd
sys.path.insert(0, _os.path.join(_ROOT, "src"))
from hmrf_solvers import adjusted_rand_index  # noqa

DATA = _os.path.join(_ROOT, "data/dlpfc12")
RES = _os.path.join(_ROOT, "experiments/h9-dlpfc12-benchmark/results")
ALL = ["151507", "151508", "151509", "151510", "151669", "151670",
       "151671", "151672", "151673", "151674", "151675", "151676"]


def nmi(a, b, k):
    a = np.asarray(a).ravel().astype(np.int64); b = np.asarray(b).ravel().astype(np.int64)
    ct = np.bincount(a * k + b, minlength=k * k).reshape(k, k).astype(np.float64)
    n = ct.sum(); p = ct / n
    pi = p.sum(1); pj = p.sum(0)
    denom = (pi[:, None] * pj[None, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(p > 0, p * np.log(p / (denom + 1e-300) + 1e-300), 0.0)
    I = t.sum()
    Ha = -np.sum(pi[pi > 0] * np.log(pi[pi > 0]))
    Hb = -np.sum(pj[pj > 0] * np.log(pj[pj > 0]))
    h = (Ha + Hb) / 2.0
    return 1.0 if h <= 0 else float(I / h)


def boundary_f1_edges(pred, true, edges):
    bp = pred[edges[:, 0]] != pred[edges[:, 1]]
    bt = true[edges[:, 0]] != true[edges[:, 1]]
    den = int(bp.sum()) + int(bt.sum())
    if den == 0:
        return 1.0
    return float(2.0 * int((bp & bt).sum()) / den)


def factorize(x):
    return np.unique(np.asarray(x), return_inverse=True)[1]


def load_truth(sid):
    spots = pd.read_csv(_os.path.join(DATA, sid, "spots.csv"))
    lab = spots["label"].astype(str)
    classes = sorted(lab.unique())
    truth = np.array([classes.index(t) for t in lab])
    rows = spots["row"].values.astype(np.int64); cols = spots["col"].values.astype(np.int64)
    pos2i = {(r, c): i for i, (r, c) in enumerate(zip(rows, cols))}
    e = []
    for i, (r, c) in enumerate(zip(rows, cols)):
        for dr, dc in ((0, 2), (0, -2), (1, -1), (1, 1), (-1, -1), (-1, 1)):
            j = pos2i.get((r + dr, c + dc))
            if j is not None:
                e.append((i, j))
    return truth, np.array(e), spots["barcode"].astype(str).values


def load_pred(sid, method, barcodes, n):
    if method in ("Ours(best)", "Oracle"):
        key = "best" if method == "Ours(best)" else "oracle"
        d = np.load(_os.path.join(RES, "labels", f"{sid}.npz"))
        return d[key].astype(int)
    if method == "SC.MEB":
        df = pd.read_csv(_os.path.join(RES, "scmeb", f"{sid}_labels.csv"))
        return df["pred"].values.astype(int)
    df = pd.read_csv(_os.path.join(RES, ({"BayesSpace": "bayesspace", "GraphST": "graphst"}[method]),
                                 f"{sid}_labels.csv"))
    mp = dict(zip(df["barcode"].astype(str), df["pred"].values))
    return np.array([mp.get(b, -1) for b in barcodes])


METHODS = ["Ours(best)", "Oracle", "BayesSpace", "SC.MEB", "GraphST"]
rows = []
for sid in ALL:
    truth, edges, barcodes = load_truth(sid)
    for m in METHODS:
        pred = load_pred(sid, m, barcodes, len(truth))
        mask = pred >= 0
        t = factorize(truth[mask]); p = factorize(pred[mask])
        k = max(t.max(), p.max()) + 1
        rows.append(dict(slice=sid, method=m,
                         ari=round(float(adjusted_rand_index(t, p)), 4),
                         nmi=round(nmi(t, p, k), 4),
                         bf1=round(boundary_f1_edges(p, t, np.array([(i, j) for i, j in edges if mask[i] and mask[j]])), 4),
                         n=int(mask.sum())))

tab = pd.DataFrame(rows)
tab.to_csv(_os.path.join(RES, "h9_benchmark_metrics.csv"), index=False)
print("=== per-method mean±sd (12 slices) ===")
summ = tab.groupby("method").agg(ari_mean=("ari", "mean"), ari_sd=("ari", "std"),
                                 nmi_mean=("nmi", "mean"), nmi_sd=("nmi", "std"),
                                 bf1_mean=("bf1", "mean"), bf1_sd=("bf1", "std")).round(4)
print(summ.to_string())

# ---- A2: paired Wilcoxon (ours vs baselines) ----
from scipy.stats import wilcoxon
piv = tab.pivot(index="slice", columns="method", values="ari")
stat_rows = []
for m in ["BayesSpace", "SC.MEB", "GraphST"]:
    for ours in ["Ours(best)"]:
        d = piv[ours] - piv[m]
        w = wilcoxon(d.values, alternative="two-sided")
        wins = int((d > 0).sum()); losses = int((d < 0).sum())
        stat_rows.append(dict(pair=f"{ours} vs {m}", median_diff=round(float(np.median(d)), 4),
                              mean_diff=round(float(d.mean()), 4), wins=wins, losses=losses,
                              cliff_delta=round((wins - losses) / len(d), 3),
                              wilcoxon_p=round(float(w.pvalue), 4)))
stat = pd.DataFrame(stat_rows)
stat.to_csv(_os.path.join(RES, "h9_benchmark_stats.csv"), index=False)
print("\n=== paired Wilcoxon (n=12, two-sided) ===")
print(stat.to_string(index=False))
print("\nMETRICS DONE")
