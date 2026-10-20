"""Validate the fast sparse path against the dense reference.

Author : Tao Zhu <zhutao@tjh.tjmu.edu.cn>
Created: 2026-10-02
Updated: 2026-10-20

Inputs
------
Xenium inputs and configuration.
Configuration: config/defaults.yaml, loaded through hmrf.config.

Outputs
-------
Console assertions and timing report.

Usage
-----
    python 04_validate_fast_path.py [options]
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

XEN = os.environ.get("HMRF_XENIUM_DIR", "")
if not XEN or not os.path.isdir(XEN):
    raise SystemExit("env var HMRF_XENIUM_DIR is unset or invalid: point it at the Xenium outs directory (10x Genomics public data, see README)")
sys.path.insert(0, os.path.join(ROOT, "src"))
from hmrf.solvers import em_mf_gaussian_graph, gaussian_log_unaries, adjusted_rand_index

N = CFG["datasets"]["xenium"]["smoke_n"]
rng = np.random.default_rng(49)

import scanpy as sc
ad = sc.read_10x_h5(os.path.join(XEN, "cell_feature_matrix.h5"))
Xc = ad.X.tocsr()
cells = pd.read_csv(os.path.join(XEN, "cells.csv.gz"))
coords = cells[["x_centroid", "y_centroid"]].values.astype(np.float64)
ref_k8 = pd.read_csv(os.path.join(XEN, "analysis/clustering/gene_expression_kmeans_8_clusters/clusters.csv"))
lab_k8 = np.full(len(cells), -1, dtype=np.int64); lab_k8[ref_k8["Barcode"].values - 1] = ref_k8["Cluster"].values

idx = np.sort(rng.choice(len(cells), N, replace=False))
Xc = Xc[idx]; coords = coords[idx]; ref = lab_k8[idx]

lib = np.asarray(Xc.sum(1)).ravel(); lib[lib == 0] = 1.0
Xc = sp.diags((1e4 / lib).astype(np.float32)) @ Xc.tocsr().astype(np.float32)
Y = np.log1p(Xc.toarray()); Y -= Y.mean(0, keepdims=True)
sd = Y.std(0, keepdims=True); sd[sd < 1e-6] = 1.0; Y /= sd
G = (Y.T @ Y).astype(np.float64) / N
w, V = np.linalg.eigh(G); V = V[:, ::-1][:, :30]
PC = (Y @ V).astype(np.float32); PC -= PC.mean(0, keepdims=True); PC /= (PC.std(0, keepdims=True) + 1e-9)

tree = cKDTree(coords); dist, nn = tree.query(coords, k=7)
nn_idx = nn[:, 1:]
# Unique undirected edges -> bidirectional directed table (each edge exactly twice, canonical convention)
E = np.unique(np.stack([np.minimum(np.repeat(np.arange(N), 6), nn_idx.ravel()),
                        np.maximum(np.repeat(np.arange(N), 6), nn_idx.ravel())], 1), axis=0)
edges = np.vstack([E, E[:, ::-1]])
# Symmetric sparse
sym = sp.coo_matrix((np.ones(N * 6, np.int8), (np.repeat(np.arange(N), 6), nn_idx.ravel())), shape=(N, N)).tocsr()
sym = ((sym + sym.T) > 0).astype(np.float32).tocsr()
print(f"n={N} undirected={len(E)} directed={len(edges)} symnnz={sym.nnz}")

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

K, beta = 8, 1.0
init = kmeanspp_labels(PC, K, seed=100)
print("init ARI =", round(float(adjusted_rand_index(ref, init)), 4))

t = time.time()
q1, mu1, s1, it1, e1 = em_mf_gaussian_graph(PC, K, beta, edges, max_iter=200, tol=1e-4, restarts=1, seed=100, init_labels=init)
t1 = time.time() - t
lab1 = q1.argmax(1); ari1 = adjusted_rand_index(ref, lab1)
print(f"ORIGINAL: iters={it1} ari={ari1:.4f} sigma={s1:.3f} elbo={e1:.1f} t={t1:.1f}s")

def em_knn_fast(y2d, K, beta, nbr_sum, max_iter=200, tol=1e-4, sigma_floor=0.2, init_labels=None, seed=0):
    rr = np.random.default_rng(seed)
    n_, d_ = y2d.shape
    lab = np.asarray(init_labels, dtype=np.int64).copy()
    mu = np.zeros((K, d_))
    for k in range(K):
        m = lab == k
        mu[k] = y2d[m].mean(0) if m.any() else rr.standard_normal(d_)
    log_u = gaussian_log_unaries(y2d, mu, 1.0)
    sigma_cap = float(np.sqrt((y2d ** 2).mean())) or 1.0
    q = np.exp(log_u - log_u.max(1, keepdims=True)); q /= q.sum(1, keepdims=True)
    sigma = 1.0; nbr = None
    for it in range(max_iter):
        nbr = nbr_sum(q)
        logq = log_u + beta * nbr
        logq -= logq.max(1, keepdims=True)
        new = np.exp(logq); new /= new.sum(1, keepdims=True)
        dq = np.abs(new - q).max(); q = new
        w = q.astype(np.float64)
        mu = (w.T @ y2d) / (w.sum(0)[:, None] + 1e-12)
        ysq = (y2d ** 2).sum(1)
        sig2 = (w.T @ ysq) / (w.sum(0) * d_ + 1e-12) - (mu ** 2).sum(1) / d_
        sigma = float(np.sqrt(max(sig2.max(), sigma_floor ** 2))); sigma = min(sigma, sigma_cap)
        log_u = gaussian_log_unaries(y2d, mu, sigma)
        if dq < tol and it > 10:
            break
    elbo = float((q * log_u).sum() + 0.5 * beta * (q * nbr).sum() - (q * np.log(q + 1e-300)).sum())
    return q, mu, sigma, it + 1, elbo

t = time.time()
q2, mu2, s2, it2, e2 = em_knn_fast(PC, K, beta, lambda q: np.asarray(sym @ q.astype(np.float32), dtype=q.dtype), init_labels=init, seed=100)
t2 = time.time() - t
lab2 = q2.argmax(1); ari2 = adjusted_rand_index(ref, lab2)
print(f"FAST:     iters={it2} ari={ari2:.4f} sigma={s2:.3f} elbo={e2:.1f} t={t2:.1f}s")
print(f"dARI = {abs(ari1-ari2):.5f} | label agreement = {(lab1==lab2).mean():.4f}")
