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
"""H4: DLPFC 151673 real-data validation (7-layer labels, graph-structured VI-EM, beta grid + ELBO selection)."""
import sys, time, csv, json
sys.path.insert(0, _os.path.join(_ROOT, "src"))
import numpy as np
import pandas as pd
import h5py
from hmrf_solvers import em_mf_gaussian_graph, adjusted_rand_index

DATA = _os.path.join(_ROOT, "data/dlpfc")
SAMPLE = "151673"
K, d_hvg, BETA_GRID = 7, 2000, (0.5, 0.7, 1.0, 1.3)

t_all = time.perf_counter()
# ---- load counts ----
f = h5py.File(f"{DATA}/{SAMPLE}_raw_feature_bc_matrix.h5", "r")
M = f["matrix"]
n_genes, n_spots = tuple(int(x) for x in M["shape"])  # this file's shape is (genes, spots)
data = np.array(M["data"], dtype=np.float32)
indices = np.array(M["indices"], dtype=np.int64)
indptr = np.array(M["indptr"], dtype=np.int64)
barcodes = [b.decode() for b in f["matrix"]["barcodes"]]
f.close()
X = np.zeros((n_spots, n_genes), dtype=np.float32)
for s in range(n_spots):  # CSR is stored by spot rows (indptr length = n_spots+1)
    a, b = indptr[s], indptr[s + 1]
    X[s, indices[a:b]] = data[a:b]
print(f"counts dense: {X.shape}, time {time.perf_counter()-t_all:.1f}s", flush=True)

# ---- coordinates and labels ----
pos = pd.read_csv(f"{DATA}/tissue_positions_list.txt", header=None, index_col=0)
pos.index = pos.index.astype(str)
pos.columns = ["in_tissue", "row", "col", "px_row", "px_col"]
lab = pd.read_csv(f"{DATA}/cluster_labels.csv")
lab.columns = [c.strip() for c in lab.columns]
gt_col = "ground_truth" if "ground_truth" in lab.columns else lab.columns[-1]
lab["barcode"] = lab[lab.columns[0]].astype(str).str.replace("^151673_", "", regex=True)
lab = lab.set_index("barcode")[gt_col]
lab = lab[~lab.isna()]

in_tissue = pos[pos["in_tissue"] == 1]
common = in_tissue.index.intersection(lab.index)
print(f"in-tissue spots: {len(in_tissue)}, labeled: {len(common)}", flush=True)
bc_idx = {b: i for i, b in enumerate(barcodes)}
keep = [bc_idx[b] for b in common]
X = X[keep]
rows = in_tissue.loc[common, "row"].values.astype(np.int64)
cols = in_tissue.loc[common, "col"].values.astype(np.int64)
truth_str = lab.loc[common].values
classes = sorted(set(truth_str))
truth = np.array([classes.index(t) for t in truth_str])
n = X.shape[0]
print(f"n={n}, layers={classes}", flush=True)

# ---- HVG ----
spot_tot = X.sum(1, keepdims=True) + 1e-9
cpm = X / spot_tot * 1e4
lx = np.log1p(cpm)
mean_g = lx.mean(0)
var_g = lx.var(0)
disp = np.where(mean_g > 1e-3, var_g / np.maximum(mean_g, 1e-9), 0.0)
hvg = np.argsort(disp)[::-1][:d_hvg]
Y = lx[:, hvg].astype(np.float64)
del X, cpm, lx
print(f"HVG selected {d_hvg}", flush=True)

# ---- hexagonal lattice adjacency edge list (Visium: same row 2 columns apart, adjacent rows differ by ±1 column) ----
pos2i = {(r, c): i for i, (r, c) in enumerate(zip(rows, cols))}
edges = []
for i, (r, c) in enumerate(zip(rows, cols)):
    for dr, dc in ((0, 2), (0, -2), (1, -1), (1, 1), (-1, -1), (-1, 1)):
        j = pos2i.get((r + dr, c + dc))
        if j is not None:
            edges.append((i, j))
edges = np.array(edges, dtype=np.int64)
print(f"edges: {len(edges)} (avg {len(edges)/n:.2f}/spot)", flush=True)

# ---- VI-EM beta grid ----
rows_out = []
best = None
for beta in BETA_GRID:
    t0 = time.perf_counter()
    q, mu, sigma, iters, elbo = em_mf_gaussian_graph(Y, K, beta, edges, restarts=3, max_iter=300, tol=1e-4, seed=9)
    t = time.perf_counter() - t0
    pred = np.argmax(q, axis=1)
    ari = adjusted_rand_index(truth, pred)
    rows_out.append(dict(beta=beta, ari=round(ari, 4), elbo=round(elbo, 1), sigma_hat=round(sigma, 3),
                         iters=iters, t_s=round(t, 1)))
    print(rows_out[-1], flush=True)
    if best is None or elbo > best[1]:
        best = (beta, elbo, pred, ari)
print("BEST by ELBO:", dict(beta=best[0], elbo=round(best[1], 1), ari=round(best[3], 4)), flush=True)
with open(_os.path.join(_ROOT, "experiments/h4-dlpfc/results/h4_results.csv"), "w", newline="") as fw:
    w = csv.DictWriter(fw, fieldnames=list(rows_out[0].keys())); w.writeheader(); w.writerows(rows_out)
np.save(_os.path.join(_ROOT, "experiments/h4-dlpfc/results/h4_best_labels.npy"), best[2])
np.save(_os.path.join(_ROOT, "experiments/h4-dlpfc/results/h4_truth.npy"), truth)
print("DONE total", round(time.perf_counter() - t_all, 1), "s", flush=True)
