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
"""H4 诊断：oracle 上限 / 纯 GMM 基线 / z-score 特征 / 更好初始化。"""
import sys, time
sys.path.insert(0, _os.path.join(_ROOT, "src"))
import numpy as np
import pandas as pd
import h5py
from hmrf_solvers import em_mf_gaussian_graph, mean_field_graph, gaussian_log_unaries, adjusted_rand_index

DATA = _os.path.join(_ROOT, "data/dlpfc")
K = 7
# 载入（与 run.py 相同）
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
keep = [bc_idx[b] for b in common]
X = X[keep]
rows = in_t.loc[common, "row"].values.astype(np.int64); cols = in_t.loc[common, "col"].values.astype(np.int64)
truth_str = lab.loc[common].values
classes = sorted(set(truth_str)); truth = np.array([classes.index(t) for t in truth_str])
n = X.shape[0]
spot_tot = X.sum(1, keepdims=True) + 1e-9
lx = np.log1p(X / spot_tot * 1e4)
mean_g = lx.mean(0); var_g = lx.var(0)
disp = np.where(mean_g > 1e-3, var_g / np.maximum(mean_g, 1e-9), 0.0)
hvg = np.argsort(disp)[::-1][:2000]
Y = lx[:, hvg].astype(np.float64)
del X
pos2i = {(r, c): i for i, (r, c) in enumerate(zip(rows, cols))}
edges = []
for i, (r, c) in enumerate(zip(rows, cols)):
    for dr, dc in ((0,2),(0,-2),(1,-1),(1,1),(-1,-1),(-1,1)):
        j = pos2i.get((r+dr, c+dc))
        if j is not None: edges.append((i, j))
edges = np.array(edges, dtype=np.int64)
print(f"n={n} edges={len(edges)}", flush=True)

def ari_mf(Ym, beta):
    q, mu, sig, iters, elbo = em_mf_gaussian_graph(Ym, K, beta, edges, restarts=1, max_iter=300, tol=1e-4, seed=9)
    return adjusted_rand_index(truth, np.argmax(q, axis=1)), elbo, sig

# (a) oracle 上限：真标签均值 + 真 σ 下纯 MF
mu_or = np.zeros((K, Y.shape[1]))
for k in range(K):
    mu_or[k] = Y[truth == k].mean(0)
log_u = gaussian_log_unaries(Y, mu_or, 0.21).astype(np.float64)
for b in (0.5, 1.0, 1.5):
    qo, it = mean_field_graph(log_u, b, edges, max_iter=500, tol=1e-3)
    print(f"ORACLE beta={b}: ARI={adjusted_rand_index(truth, np.argmax(qo, axis=1)):.4f} iters={it}", flush=True)

# (b) 纯 GMM（beta=0，即无空间先验）
for seed in (9, 10):
    q, mu, sig, iters, elbo = em_mf_gaussian_graph(Y, K, 0.0, edges, restarts=1, max_iter=300, tol=1e-4, seed=seed)
    print(f"GMM beta=0 seed={seed}: ARI={adjusted_rand_index(truth, np.argmax(q, axis=1)):.4f}", flush=True)

# (c) z-score 特征后重跑 VI-EM
Yz = (Y - Y.mean(0, keepdims=True)) / (Y.std(0, keepdims=True) + 1e-6)
for b in (0.7, 1.3):
    q, mu, sig, iters, elbo = em_mf_gaussian_graph(Yz, K, b, edges, restarts=1, max_iter=300, tol=1e-4, seed=9)
    print(f"ZSCORE beta={b}: ARI={adjusted_rand_index(truth, np.argmax(q, axis=1)):.4f}", flush=True)

# (d) 更多起点 + 更大 beta（原始特征）
for b in (1.7, 2.2):
    q, mu, sig, iters, elbo = em_mf_gaussian_graph(Y, K, b, edges, restarts=3, max_iter=300, tol=1e-4, seed=9)
    print(f"RAW beta={b} r3: ARI={adjusted_rand_index(truth, np.argmax(q, axis=1)):.4f}", flush=True)
print("DONE")
