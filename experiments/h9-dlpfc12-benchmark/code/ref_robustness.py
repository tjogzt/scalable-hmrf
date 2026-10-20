# --- portable project root (no hardcoded user paths) ---
import os as _os
import pathlib as _pl


def _find_root(_start):
    _p = _pl.Path(_start).resolve()
    for _anc in [_p] + list(_p.parents):
        if (_anc / "src" / "hmrf_solvers.py").exists():
            return str(_anc)
    raise RuntimeError("project root not found; run inside the repository")


_ROOT = _os.environ.get("HMRF_ROOT") or _find_root(__file__)
# ------------------------------------------------------
"""H9 reference-robustness check (review R7): how stable is the pipeline-conditioned
reference under different constructions, initialisations and beta grids?

Variants (same feature pipeline as run_ours.py):
  A truth_class_means   : truth-label class means -> unaries -> pure mean field (current construction)
  B truth_init_kmeans   : one Lloyd step from truth labels -> class means -> pure MF
  C kmeanspp_random x5  : random k-means++ labelings -> class means -> pure MF
  D pca15_truth_means   : first 15 PCs, truth-label class means -> pure MF
Beta grid: 0.3..2.0 step 0.1 (max over the grid = the reference value for that variant).
Usage: python ref_robustness.py
"""
import csv
import sys

import numpy as np

sys.path.insert(0, _os.path.join(_ROOT, "src"))
sys.path.insert(0, _os.path.join(_ROOT, "experiments", "h9-dlpfc12-benchmark", "code"))
from hmrf_solvers import adjusted_rand_index, gaussian_log_unaries, mean_field_graph
from run_ours import kmeans_pp, load_slice

GRID = [round(0.3 + 0.1 * i, 2) for i in range(18)]  # 0.3 .. 2.0


def pipeline(sid):
    X, rows, cols, truth = load_slice(sid)
    k = int(np.unique(truth).size)
    lx = np.log1p(X / (X.sum(1, keepdims=True) + 1e-9) * 1e4)
    del X
    m = lx.mean(0); v = lx.var(0); use = m > 1e-3
    lm = np.log10(m[use]); lv = np.log10(np.maximum(v[use], 1e-12))
    coef = np.polyfit(lm, lv, 2); resid = lv - np.polyval(coef, lm)
    gidx = np.where(use)[0][np.argsort(resid)[::-1][:3000]]
    Y = lx[:, gidx].astype(np.float64)
    Yz = (Y - Y.mean(0, keepdims=True)) / (Y.std(0, keepdims=True) + 1e-6)
    U, S, Vt = np.linalg.svd(Yz, full_matrices=False)
    PC = U[:, :50] * S[:50]
    PC = PC - PC.mean(0, keepdims=True); PC = PC / (PC.std(0, keepdims=True) + 1e-6)
    pos2i = {(r, c): i for i, (r, c) in enumerate(zip(rows, cols))}
    edges = []
    for i, (r, c) in enumerate(zip(rows, cols)):
        for dr, dc in ((0, 2), (0, -2), (1, -1), (1, 1), (-1, -1), (-1, 1)):
            j = pos2i.get((r + dr, c + dc))
            if j is not None:
                edges.append((i, j))
    return PC, truth, k, np.array(edges, dtype=np.int64)


def ref_best_ari(PCf, truth, k, edges, mu):
    log_u = gaussian_log_unaries(PCf, mu, 0.8)
    best, bb = 0.0, None
    for b in GRID:
        q, _ = mean_field_graph(log_u, b, edges, max_iter=500, tol=1e-3)
        a = float(adjusted_rand_index(truth, np.argmax(q, axis=1)))
        if a > best:
            best, bb = a, b
    return best, bb


rows = []
for sid in ("151673", "151507"):
    PC, truth, k, edges = pipeline(sid)
    # A: truth-class means (current construction)
    muA = np.array([PC[truth == kk].mean(0) for kk in range(k)])
    a, b = ref_best_ari(PC, truth, k, edges, muA)
    rows.append(dict(slice=sid, variant="truth_class_means", seed=-1, ari=round(a, 4), beta=b))
    # B: one Lloyd step from truth labels
    d = ((PC[:, None, :] - muA[None, :, :]) ** 2).sum(2)
    labB = d.argmin(1)
    muB = np.array([PC[labB == kk].mean(0) if (labB == kk).any() else muA[kk] for kk in range(k)])
    aB, bB = ref_best_ari(PC, truth, k, edges, muB)
    rows.append(dict(slice=sid, variant="truth_init_kmeans", seed=-1, ari=round(aB, 4), beta=bB))
    # C: random k-means++ labelings (5 seeds)
    for t in range(5):
        rng = np.random.default_rng(500 + t)
        lab = kmeans_pp(PC, k, rng)
        muC = np.array([PC[lab == kk].mean(0) if (lab == kk).any() else muA[kk] for kk in range(k)])
        aC, bC = ref_best_ari(PC, truth, k, edges, muC)
        rows.append(dict(slice=sid, variant="kmeanspp_random", seed=500 + t, ari=round(aC, 4), beta=bC))
    # D: PCA15 feature variant
    PC15 = PC[:, :15]
    muD = np.array([PC15[truth == kk].mean(0) for kk in range(k)])
    aD, bD = ref_best_ari(PC15, truth, k, edges, muD)
    rows.append(dict(slice=sid, variant="pca15_truth_means", seed=-1, ari=round(aD, 4), beta=bD))
    print(sid, "done", flush=True)

out = _os.path.join(_ROOT, "experiments/h9-dlpfc12-benchmark/results/ref_robustness.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print("\n== spread per slice ==")
for sid in ("151673", "151507"):
    aa = [r["ari"] for r in rows if r["slice"] == sid]
    print(f"{sid}: reference ARI across variants: min={min(aa):.4f} max={max(aa):.4f} mean={np.mean(aa):.4f}")
print("DONE")
