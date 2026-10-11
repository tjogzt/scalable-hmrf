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
"""H5b: 边界拐点 beta 选择准则 vs ELBO（NB 仿真 + DLPFC 两场景）。"""
import sys, time, csv
sys.path.insert(0, _os.path.join(_ROOT, "src"))
import numpy as np
import pandas as pd
import h5py
from simulate import gaussian_rf_domains, nb_emission_domains
from hmrf_solvers import em_mf_nb_restarts, em_mf_gaussian_graph, adjusted_rand_index

def boundary_ratio(labels, edges):
    a = labels[edges[:, 0]]; b = labels[edges[:, 1]]
    return float((a != b).mean())

def elbow_pick(betas, ratios, eps=0.12):
    rmin = min(ratios)
    # 选满足 ratio <= (1+eps)*rmin 的最小 beta（偏好弱平滑）
    cand = [b for b, r in zip(betas, ratios) if r <= (1 + eps) * rmin]
    return cand[0]

rows = []
# ---- 场景 1：NB 仿真 S=250（真最优 beta≈1.3，ELBO 误选 0.3）----
S, K, d, DISP = 250, 8, 20, 3.0
FACTORS = [0.8, 1.1, 1.5, 2.0, 2.7, 3.7, 5.0, 6.8]
truth = gaussian_rf_domains(S, K, seed=41)
y, mu_true = nb_emission_domains(truth, d=d, factors=FACTORS, disp=DISP, base_scale=1.0, seed=42)
y2d = y.reshape(-1, d).astype(np.float64)
# 网格边
edges_nb = []
pos2i = {}
k = 0
for i in range(S):
    for j in range(S):
        pos2i[(i, j)] = k; k += 1
for i in range(S):
    for j in range(S):
        for di, dj in ((1,0),(-1,0),(0,1),(0,-1)):
            t = pos2i.get((i+di, j+dj))
            if t is not None: edges_nb.append((pos2i[(i,j)], t))
edges_nb = np.array(edges_nb, dtype=np.int64)
BETAS = (0.3, 0.5, 0.7, 1.0, 1.3, 1.7)
res = []
for b in BETAS:
    q, mu, iters, elbo = em_mf_nb_restarts(y2d, K, b, S, S, disp=DISP, max_iter=300, tol=1e-3, restarts=3, seed=5)
    pred = np.argmax(q, axis=2).ravel()  # 网格版 q 为 (H,W,K)
    res.append(dict(beta=b, ari=adjusted_rand_index(truth, pred), elbo=elbo,
                    br=boundary_ratio(pred, edges_nb)))
    print("NB", res[-1], flush=True)
best_ari = max(x["ari"] for x in res)
elbo_b = max(res, key=lambda x: x["elbo"])["beta"]
elbow_b = elbow_pick([x["beta"] for x in res], [x["br"] for x in res])
print(f"NB: best-ARI beta={[x['beta'] for x in res if x['ari']==best_ari][0]} (ARI {best_ari:.3f}), ELBO->{elbo_b}, elbow->{elbow_b}", flush=True)
rows += [dict(scene="nb_sim", **x) for x in res]
rows.append(dict(scene="nb_sim", beta=f"ELBOW={elbow_b}", ari=None, elbo=None, br=None))

# ---- 场景 2：DLPFC（真最优 beta≈0.7（best-of-5），ELBO 失效）----
DATA = _os.path.join(_ROOT, "data/dlpfc")
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
rows_ = in_t.loc[common, "row"].values.astype(np.int64); cols_ = in_t.loc[common, "col"].values.astype(np.int64)
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
U, S_, Vt = np.linalg.svd(Yz, full_matrices=False)
PC = U[:, :50] * S_[:50]
PC = PC - PC.mean(0, keepdims=True); PC = PC / (PC.std(0, keepdims=True) + 1e-6)
pos2i2 = {(r, c): i for i, (r, c) in enumerate(zip(rows_, cols_))}
edges_d = []
for i, (r, c) in enumerate(zip(rows_, cols_)):
    for dr, dc in ((0,2),(0,-2),(1,-1),(1,1),(-1,-1),(-1,1)):
        j = pos2i2.get((r+dr, c+dc))
        if j is not None: edges_d.append((i, j))
edges_d = np.array(edges_d, dtype=np.int64)
res2 = []
for b in (0.3, 0.5, 0.7, 1.0):
    aris = []
    for t in range(5):
        rng = np.random.default_rng(100 + t)
        # kmeans++ init
        c = [int(rng.integers(0, n))]
        for _ in range(6):
            d2 = ((PC[:, None, :] - PC[c][None, :, :]) ** 2).sum(2).min(1)
            c.append(int(rng.choice(n, p=(d2 + 1e-12) / (d2.sum() + 1e-12))))
        mu_i = PC[c].copy()
        for _ in range(30):
            ll = np.argmin(((PC[:, None, :] - mu_i[None, :, :]) ** 2).sum(2), axis=1)
            for kk in range(7):
                mm = ll == kk
                if mm.any(): mu_i[kk] = PC[mm].mean(0)
        q, mu, sig, iters, elbo = em_mf_gaussian_graph(PC, 7, b, edges_d, restarts=1, max_iter=300, tol=1e-4, seed=t, init_labels=ll)
        aris.append(adjusted_rand_index(truth, np.argmax(q, axis=1)))
    pred = np.argmax(q, axis=1)
    res2.append(dict(beta=b, ari=max(aris), br=boundary_ratio(pred, edges_d)))
    print("DLPFC", res2[-1], flush=True)
best_ari2 = max(x["ari"] for x in res2)
elbow_b2 = elbow_pick([x["beta"] for x in res2], [x["br"] for x in res2])
print(f"DLPFC: best-ARI beta={[x['beta'] for x in res2 if x['ari']==best_ari2][0]} (ARI {best_ari2:.3f}), elbow->{elbow_b2}", flush=True)
rows += [dict(scene="dlpfc", **x) for x in res2]
with open(_os.path.join(_ROOT, "experiments/h5b-boundary-criterion/results/h5b_results.csv"), "w", newline="") as fw:
    w = csv.DictWriter(fw, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("DONE")
