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
"""H10 标签复现（多配置）：同种子确定性重放指定 (K,beta,restart)，max_iter=500 并报告收敛。
用法: python save_best_labels.py 8:0.7:2 25:0.7:2
输出: results/h10_labels_K{K}_b{beta}_r{r}.npz（每个配置一个文件）
"""
import sys, os, time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree

ROOT = _ROOT
XEN = _os.environ.get("HMRF_XENIUM_DIR", "")
if not XEN or not _os.path.isdir(XEN):
    raise SystemExit("环境变量 HMRF_XENIUM_DIR 未设置或无效：请指向 Xenium outs 目录（10x Genomics 公开数据，见 README）")
OUT = os.path.join(ROOT, "experiments/h10-xenium574k/results")
sys.path.insert(0, os.path.join(ROOT, "src"))
from hmrf_solvers import gaussian_log_unaries, adjusted_rand_index  # noqa

CONFIGS = []
for a in sys.argv[1:]:
    K_, b_, r_ = a.split(":")
    CONFIGS.append((int(K_), float(b_), int(r_)))
assert CONFIGS, "用法: save_best_labels.py K:beta:restart [K:beta:restart ...]"

rng = np.random.default_rng(49)
t0 = time.time()

import scanpy as sc
ad = sc.read_10x_h5(os.path.join(XEN, "cell_feature_matrix.h5"))
Xc = ad.X.tocsr()
cells = pd.read_csv(os.path.join(XEN, "cells.csv.gz"))
coords = cells[["x_centroid", "y_centroid"]].values.astype(np.float64)
ref_gc = pd.read_csv(os.path.join(XEN, "analysis/clustering/gene_expression_graphclust/clusters.csv"))
lab_gc = np.full(len(cells), -1, dtype=np.int64); lab_gc[ref_gc["Barcode"].values - 1] = ref_gc["Cluster"].values
ref_k8 = pd.read_csv(os.path.join(XEN, "analysis/clustering/gene_expression_kmeans_8_clusters/clusters.csv"))
lab_k8 = np.full(len(cells), -1, dtype=np.int64); lab_k8[ref_k8["Barcode"].values - 1] = ref_k8["Cluster"].values

n = len(cells)
lib = np.asarray(Xc.sum(1)).ravel(); lib[lib == 0] = 1.0
Xc = sp.diags((1e4 / lib).astype(np.float32)) @ Xc.tocsr().astype(np.float32)
Y = np.log1p(Xc.toarray()); del Xc
Y -= Y.mean(0, keepdims=True)
sd = Y.std(0, keepdims=True); sd[sd < 1e-6] = 1.0; Y /= sd
G = (Y.T @ Y).astype(np.float64) / n
w, V = np.linalg.eigh(G); V = V[:, ::-1][:, :30]
PC = (Y @ V).astype(np.float32); PC -= PC.mean(0, keepdims=True); PC /= (PC.std(0, keepdims=True) + 1e-9)
del Y, G
print(f"features/PCA {time.time()-t0:.1f}s", flush=True)

tree = cKDTree(coords); dist, nn = tree.query(coords, k=7)
nn_idx = nn[:, 1:]
sym = sp.coo_matrix((np.ones(n * 6, np.int8), (np.repeat(np.arange(n), 6), nn_idx.ravel())), shape=(n, n)).tocsr()
sym = ((sym + sym.T) > 0).astype(np.float32).tocsr()
print(f"kNN {time.time()-t0:.1f}s", flush=True)

def kmeanspp_labels(PC, K, seed):
    r = np.random.default_rng(seed); n_ = PC.shape[0]
    c = [int(r.integers(n_))]; d2 = ((PC - PC[c[0]]) ** 2).sum(1)
    for _ in range(1, K):
        nxt = int(r.choice(n_, p=d2 / d2.sum())); c.append(nxt)
        d2 = np.minimum(d2, ((PC - PC[nxt]) ** 2).sum(1))
    C = PC[c]; out = np.empty(n_, dtype=np.int64)
    for s in range(0, n_, 100000):
        ch = PC[s:s + 100000]
        dd = (ch ** 2).sum(1)[:, None] + (C ** 2).sum(1)[None, :] - 2.0 * (ch @ C.T)
        out[s:s + 100000] = dd.argmin(1)
    return out

def run_cfg(K, beta, restart, max_iter=500):
    ref = lab_gc if K > 10 else lab_k8
    init = kmeanspp_labels(PC, K, seed=100 + restart)
    lab = np.asarray(init, dtype=np.int64).copy()
    mu = np.zeros((K, 30))
    for k in range(K):
        m = lab == k
        mu[k] = PC[m].mean(0) if m.any() else rng.standard_normal(30)
    log_u = gaussian_log_unaries(PC, mu, 1.0)
    sigma_cap = float(np.sqrt((PC ** 2).mean())) or 1.0
    q = np.exp(log_u - log_u.max(1, keepdims=True)); q /= q.sum(1, keepdims=True)
    sigma = 1.0; nbr = None; dq = np.inf; it = 0
    for it in range(max_iter):
        nbr = sym @ q
        logq = log_u + beta * nbr
        logq -= logq.max(1, keepdims=True)
        new = np.exp(logq); new /= new.sum(1, keepdims=True)
        dq = float(np.abs(new - q).max()); q = new
        wv = q.astype(np.float64)
        mu = (wv.T @ PC) / (wv.sum(0)[:, None] + 1e-12)
        ysq = (PC ** 2).sum(1)
        sig2 = (wv.T @ ysq) / (wv.sum(0) * 30 + 1e-12) - (mu ** 2).sum(1) / 30
        sigma = float(np.sqrt(max(sig2.max(), 0.04))); sigma = min(sigma, sigma_cap)
        log_u = gaussian_log_unaries(PC, mu, sigma)
        if dq < 1e-4 and it > 10:
            break
    pred = q.argmax(1)
    ari = adjusted_rand_index(ref[ref >= 0], pred[ref >= 0])
    print(f"K={K} beta={beta} r={restart}: iters={it+1} dq_final={dq:.2e} ARI={ari:.4f} "
          f"sigma={sigma:.3f} {time.time()-t0:.1f}s", flush=True)
    np.savez_compressed(os.path.join(OUT, f"h10_labels_K{K}_b{beta:g}_r{restart}.npz"),
                        pred=pred.astype(np.int16), lab_gc=lab_gc.astype(np.int16),
                        lab_k8=lab_k8.astype(np.int16), coords=coords.astype(np.float32),
                        K=K, beta=beta)
    return ari

for (K, beta, r) in CONFIGS:
    run_cfg(K, beta, r)
print("ALL DONE", flush=True)
