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
"""H4 final: 5 起点 × beta 网格；按 ELBO 选优 + 报告 max/median ARI。"""
import sys, time, csv, json
sys.path.insert(0, _os.path.join(_ROOT, "src"))
import numpy as np
import pandas as pd
import h5py
from hmrf_solvers import em_mf_gaussian_graph, mean_field_graph, gaussian_log_unaries, adjusted_rand_index

DATA = _os.path.join(_ROOT, "data/dlpfc")
K, N_PC = 7, 50
BETAS = (0.3, 0.5, 0.7, 1.0)
N_INIT = 5

f = h5py.File(f"{DATA}/151673_raw_feature_bc_matrix.h5", "r")
M = f["matrix"]
n_genes, n_spots = int(M["shape"][0]), int(M["shape"][1])
data = np.array(M["data"], dtype=np.float32); indices = np.array(M["indices"], dtype=np.int64); indptr = np.array(M["indptr"], dtype=np.int64)
barcodes = [b.decode() for b in f["matrix"]["barcodes"]]; f.close()
X = np.zeros((n_spots, n_genes), dtype=np.float32)
for s in range(n_spots):
    a, b = indptr[s], indptr[s+1]; X[s, indices[a:b]] = data[a:b]
pos = pd.read_csv(f"{DATA}/tissue_positions_list.txt", header=None, index_col=0); pos.index = pos.index.astype(str)
pos.columns = ["in_tissue", "row", "col", "px_row", "px_col"]
lab = pd.read_csv(f"{DATA}/cluster_labels.csv"); lab.columns = [c.strip() for c in lab.columns]
lab["barcode"] = lab[lab.columns[0]].astype(str).str.replace("^151673_", "", regex=True)
lab = lab.set_index("barcode")["ground_truth"]; lab = lab[~lab.isna()]
in_t = pos[pos["in_tissue"] == 1]
common = in_t.index.intersection(lab.index)
bc_idx = {b: i for i, b in enumerate(barcodes)}
X = X[[bc_idx[b] for b in common]]
rows = in_t.loc[common, "row"].values.astype(np.int64); cols = in_t.loc[common, "col"].values.astype(np.int64)
truth_str = lab.loc[common].values
classes = sorted(set(truth_str)); truth = np.array([classes.index(t) for t in truth_str])
n = X.shape[0]
lx = np.log1p(X / (X.sum(1, keepdims=True) + 1e-9) * 1e4)
del X
m = lx.mean(0); v = lx.var(0); use = m > 1e-3
lm = np.log10(m[use]); lv = np.log10(np.maximum(v[use], 1e-12))
coef = np.polyfit(lm, lv, 2); resid = lv - np.polyval(coef, lm)
gidx = np.where(use)[0][np.argsort(resid)[::-1][:3000]]
Y = lx[:, gidx].astype(np.float64)
Yz = (Y - Y.mean(0, keepdims=True)) / (Y.std(0, keepdims=True) + 1e-6)
del Y
U, S, Vt = np.linalg.svd(Yz, full_matrices=False)
PC = U[:, :N_PC] * S[:N_PC]
PC = PC - PC.mean(0, keepdims=True); PC = PC / (PC.std(0, keepdims=True) + 1e-6)

pos2i = {(r, c): i for i, (r, c) in enumerate(zip(rows, cols))}
edges = []
for i, (r, c) in enumerate(zip(rows, cols)):
    for dr, dc in ((0,2),(0,-2),(1,-1),(1,1),(-1,-1),(-1,1)):
        j = pos2i.get((r+dr, c+dc))
        if j is not None: edges.append((i, j))
edges = np.array(edges, dtype=np.int64)

def kmeans_pp(Xm, k, rng):
    c = [int(rng.integers(0, len(Xm)))]
    for _ in range(k - 1):
        d2 = ((Xm[:, None, :] - Xm[c][None, :, :]) ** 2).sum(2).min(1)
        c.append(int(rng.choice(len(Xm), p=(d2 + 1e-12) / (d2.sum() + 1e-12))))
    mu = Xm[c].copy()
    for _ in range(30):
        lab = np.argmin(((Xm[:, None, :] - mu[None, :, :]) ** 2).sum(2), axis=1)
        for kk in range(k):
            mm = lab == kk
            if mm.any(): mu[kk] = Xm[mm].mean(0)
    return lab

# oracle 上限
mu_or = np.array([PC[truth == k].mean(0) for k in range(K)])
log_u = gaussian_log_unaries(PC, mu_or, 0.8)
best_o = 0.0
for b in (0.3, 0.5, 0.7, 1.0):
    qo, _ = mean_field_graph(log_u, b, edges, max_iter=500, tol=1e-3)
    best_o = max(best_o, adjusted_rand_index(truth, np.argmax(qo, axis=1)))
print(f"ORACLE best ARI={best_o:.4f}", flush=True)

rows = []
for b in BETAS:
    ari_list, elbo_list = [], []
    for t in range(N_INIT):
        rng = np.random.default_rng(100 + t)
        init = kmeans_pp(PC, K, rng)
        q, mu, sig, iters, elbo = em_mf_gaussian_graph(PC, K, b, edges, restarts=1, max_iter=300, tol=1e-4, seed=t, init_labels=init)
        ari_list.append(adjusted_rand_index(truth, np.argmax(q, axis=1)))
        elbo_list.append(elbo)
    best_i = int(np.argmax(elbo_list))
    rows.append(dict(beta=b, ari_elbo_best=round(ari_list[best_i], 4), ari_max=round(max(ari_list), 4),
                     ari_median=round(float(np.median(ari_list)), 4), elbo_best=round(elbo_list[best_i], 1)))
    print(rows[-1], flush=True)
with open(_os.path.join(_ROOT, "experiments/h4-dlpfc/results/h4_final.csv"), "w", newline="") as fw:
    w = csv.DictWriter(fw, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
json.dump(dict(oracle_best=best_o), open(_os.path.join(_ROOT, "experiments/h4-dlpfc/results/h4_final_meta.json"), "w"))
print("DONE")
