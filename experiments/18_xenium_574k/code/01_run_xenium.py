"""Run VI-HMRF on the 574k-cell Xenium breast section (smoke/full modes).

Author : Tao Zhu <zhutao@tjh.tjmu.edu.cn>
Created: 2026-10-02
Updated: 2026-10-20

Inputs
------
Xenium output directory from HMRF_XENIUM_DIR.
Configuration: config/defaults.yaml, loaded through hmrf.config.

Outputs
-------
results/h10_*.csv and label NPZ files.

Usage
-----
    python 01_run_xenium.py [options]
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
OUT = str(results_dir("18_xenium_574k"))
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, "src"))
from hmrf.solvers import gaussian_log_unaries, adjusted_rand_index  # noqa

MODE = sys.argv[1] if len(sys.argv) > 1 else "smoke"
SMOKE_N = CFG["datasets"]["xenium"]["smoke_n"]
rng = np.random.default_rng(49)


def em_mf_gaussian_knn(y2d, K, beta, nn_idx, max_iter=200, tol=1e-4, sigma_floor=0.2,
                       init_labels=None):
    """kNN version of VI-EM: returns (q, mu, sigma, iters, elbo). Neighbor sum = q[nn_idx].sum(1)."""
    n, d = y2d.shape
    lab = np.asarray(init_labels, dtype=np.int64).copy()
    mu = np.zeros((K, d))
    for k in range(K):
        m = lab == k
        mu[k] = y2d[m].mean(0) if m.any() else rng.standard_normal(d)
    log_u = gaussian_log_unaries(y2d, mu, 1.0)
    sigma_cap = float(np.sqrt((y2d ** 2).mean())) or 1.0
    q = np.exp(log_u - log_u.max(1, keepdims=True)); q /= q.sum(1, keepdims=True)
    sigma = 1.0
    it = 0
    for it in range(max_iter):
        nbr = q[nn_idx].sum(1)                      # (n,K) neighbor sums (fast path)
        logq = log_u + beta * nbr
        logq -= logq.max(1, keepdims=True)
        new = np.exp(logq); new /= new.sum(1, keepdims=True)
        dq = np.abs(new - q).max(); q = new
        w = q.astype(np.float64)
        mu = (w.T @ y2d) / (w.sum(0)[:, None] + 1e-12)
        ysq = (y2d ** 2).sum(1)
        sig2 = (w.T @ ysq) / (w.sum(0) * d + 1e-12) - (mu ** 2).sum(1) / d
        sigma = float(np.sqrt(max(sig2.max(), sigma_floor ** 2)))
        sigma = min(sigma, sigma_cap)
        log_u = gaussian_log_unaries(y2d, mu, sigma)
        if dq < tol and it > 10:
            break
    elbo = float((q * log_u).sum() + 0.5 * beta * (q * nbr).sum() - (q * np.log(q + 1e-300)).sum())
    return q, mu, sigma, it + 1, elbo


t0 = time.time()
# ---------- 1. Data loading ----------
try:
    import scanpy as sc
    ad = sc.read_10x_h5(os.path.join(XEN, "cell_feature_matrix.h5"))
    Xc = ad.X.tocsr()
    barc = np.array([int(b) for b in ad.obs_names])
    print(f"h5 load: {Xc.shape} {time.time()-t0:.1f}s", flush=True)
except Exception as e:
    print("h5 failed, fallback mtx:", repr(e), flush=True)
    import scipy.io as sio
    with gzip.open(os.path.join(XEN, "cell_feature_matrix/matrix.mtx.gz"), "rb") as fh:
        M = sio.mmread(fh).tocsr()
    Xc = M.T.tocsr()
    barc = np.arange(1, Xc.shape[0] + 1)

cells = pd.read_csv(os.path.join(XEN, "cells.csv.gz"))
print("barcode==cell_id:", bool(np.array_equal(barc, cells["cell_id"].values)), flush=True)
coords = cells[["x_centroid", "y_centroid"]].values.astype(np.float64)

ref_gc = pd.read_csv(os.path.join(XEN, "analysis/clustering/gene_expression_graphclust/clusters.csv"))
lab_gc = np.full(len(cells), -1, dtype=np.int64)
lab_gc[ref_gc["Barcode"].values - 1] = ref_gc["Cluster"].values
ref_k8 = pd.read_csv(os.path.join(XEN, "analysis/clustering/gene_expression_kmeans_8_clusters/clusters.csv"))
lab_k8 = np.full(len(cells), -1, dtype=np.int64)
lab_k8[ref_k8["Barcode"].values - 1] = ref_k8["Cluster"].values
print(f"cells {len(cells)} | graphclust {(lab_gc>=0).sum()} ({lab_gc.max()+1} cl) | kmeans8 {(lab_k8>=0).sum()}", flush=True)

# ---------- 2. Subsampling (smoke) ----------
if MODE == "smoke":
    idx = np.sort(rng.choice(len(cells), SMOKE_N, replace=False))
    Xc = Xc[idx]; coords = coords[idx]; lab_gc = lab_gc[idx]; lab_k8 = lab_k8[idx]
    print(f"smoke subsample: {Xc.shape[0]}", flush=True)
n = Xc.shape[0]

# ---------- 3. Feature pipeline ----------
lib = np.asarray(Xc.sum(1)).ravel(); lib[lib == 0] = 1.0
Xc = sp.diags((1e4 / lib).astype(np.float32)) @ Xc.tocsr().astype(np.float32)
Y = np.log1p(Xc.toarray())
del Xc
Y -= Y.mean(0, keepdims=True)
sd = Y.std(0, keepdims=True); sd[sd < 1e-6] = 1.0
Y /= sd
print(f"features {time.time()-t0:.1f}s", flush=True)
G = (Y.T @ Y).astype(np.float64) / n
w, V = np.linalg.eigh(G)
V = V[:, ::-1][:, :30]
PC = (Y @ V).astype(np.float32)
PC -= PC.mean(0, keepdims=True); PC /= (PC.std(0, keepdims=True) + 1e-9)
del Y, G
print(f"PCA30 {time.time()-t0:.1f}s", flush=True)

# ---------- 4. kNN graph ----------
knn = 6
tree = cKDTree(coords)
dist, nn = tree.query(coords, k=knn + 1)
nn_idx = nn[:, 1:]                                   # (n,k) neighbor indices (may be asymmetric -> take the symmetric union)
# Symmetrization: if i and j are mutual neighbors each keeps the other; asymmetric pairs are completed
sym = sp.coo_matrix((np.ones(n * knn, dtype=np.int8),
                     (np.repeat(np.arange(n), knn), nn_idx.ravel())), shape=(n, n)).tocsr()
sym = ((sym + sym.T) > 0).tocsr()
kmax = int(sym.getnnz(1).max())
print(f"kNN symmetric: max deg {kmax}, mean {sym.getnnz(1).mean():.2f}, {time.time()-t0:.1f}s", flush=True)
sym = sym.astype(np.float32).tocsr()

def nbr_sum(q2d):
    return sym @ q2d

# Replace EM's neighbor sum
def em_knn_fast(y2d, K, beta, max_iter=200, tol=1e-4, sigma_floor=0.2, init_labels=None):
    n_, d_ = y2d.shape
    lab = np.asarray(init_labels, dtype=np.int64).copy()
    mu = np.zeros((K, d_))
    for k in range(K):
        m = lab == k
        mu[k] = y2d[m].mean(0) if m.any() else rng.standard_normal(d_)
    log_u = gaussian_log_unaries(y2d, mu, 1.0)
    sigma_cap = float(np.sqrt((y2d ** 2).mean())) or 1.0
    q = np.exp(log_u - log_u.max(1, keepdims=True)); q /= q.sum(1, keepdims=True)
    sigma = 1.0; it = 0; nbr = None
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

# ---------- 5. kmeans++ initialization ----------
def kmeanspp_labels(PC, K, seed):
    r = np.random.default_rng(seed)
    n_ = PC.shape[0]
    c = [int(r.integers(n_))]
    d2 = ((PC - PC[c[0]]) ** 2).sum(1)
    for _ in range(1, K):
        p = d2 / d2.sum()
        nxt = int(r.choice(n_, p=p)); c.append(nxt)
        d2 = np.minimum(d2, ((PC - PC[nxt]) ** 2).sum(1))
    C = PC[c]
    out = np.empty(n_, dtype=np.int64)
    for s in range(0, n_, 100000):
        ch = PC[s:s + 100000]
        dd = (ch ** 2).sum(1)[:, None] + (C ** 2).sum(1)[None, :] - 2.0 * (ch @ C.T)
        out[s:s + 100000] = dd.argmin(1)
    return out

# ---------- 6. MF-VI runs ----------
rows = []
BETAS = CFG["experiments"]["h10"]["betas_smoke"] if MODE == "smoke" else CFG["experiments"]["h10"]["betas_full"]
RESTARTS = CFG["experiments"]["h10"]["restarts_smoke"] if MODE == "smoke" else CFG["experiments"]["h10"]["restarts_full"]
for K, ref, refname in ((8, lab_k8, "kmeans8"), (20, lab_gc, "graphclust")):
    for beta in BETAS:
        for r in range(RESTARTS):
            init = kmeanspp_labels(PC, K, seed=100 + r)
            tr = time.time()
            q, mu, sigma, iters, elbo = em_knn_fast(PC, K, beta, max_iter=200, tol=1e-4,
                                                    init_labels=init)
            lab = q.argmax(1)
            mask = ref >= 0
            ari = adjusted_rand_index(ref[mask], lab[mask])
            dt = time.time() - tr
            rows.append(dict(K=K, ref=refname, beta=beta, restart=r, ari=round(float(ari), 4),
                             iters=iters, t_s=round(dt, 1), sigma=round(float(sigma), 3)))
            print(rows[-1], flush=True)

# ---------- 7. Summary ----------
df = pd.DataFrame(rows)
tag = MODE
df.to_csv(os.path.join(OUT, f"h10_{tag}.csv"), index=False)
if MODE == "full":
    np.savez_compressed(os.path.join(OUT, "h10_labels.npz"),
                        lab_gc=lab_gc, lab_k8=lab_k8)
json.dump(dict(mode=MODE, n=n, t_total=round(time.time() - t0, 1)),
          open(os.path.join(OUT, f"h10_{tag}_meta.json"), "w"))
best = df.sort_values("ari", ascending=False).iloc[0]
print(f"\n{'SMOKE' if MODE=='smoke' else 'FULL'} done {time.time()-t0:.1f}s | best: K={best.K} "
      f"beta={best.beta} ARI({best.ref})={best.ari} in {best.t_s}s", flush=True)
