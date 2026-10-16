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
"""H3c: alpha-expansion graph cut vs mean field on an irregular cell-level kNN graph.

Addresses the review point that the exactness comparison (h3b) was only run on
regular grids. Topology mimics a cell-level kNN graph (k=6, symmetric union),
as in the Xenium pipeline; smooth domains are sampled at irregular points;
both cores receive the same fitted unaries (fair comparison).

Usage: python run.py
"""
import csv
import sys
import time

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

sys.path.insert(0, _os.path.join(_ROOT, "src"))
from simulate import gaussian_emission
from hmrf_solvers import (adjusted_rand_index, em_mf_gaussian_graph,
                          gaussian_log_unaries, mean_field_graph)
from graphcut_hmrf import alpha_expansion_graph, potts_energy_graph

N_PTS = 10000
KNN = 6
D = 20


def knn_edges(pts, k=6):
    """Symmetric-union kNN adjacency as a deduplicated edge list (i < j)."""
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=k + 1)
    pairs = set()
    for i in range(pts.shape[0]):
        for j in idx[i, 1:]:
            a, b = (i, int(j)) if i < int(j) else (int(j), i)
            if a != b:
                pairs.add((a, b))
    return np.array(sorted(pairs), dtype=np.int64)


def field_domains_on_points(pts, K, G=120, smooth=13.0, bias=0.3, seed=0):
    """Smooth random-field domains evaluated at irregular points (bilinear)."""
    rng = np.random.default_rng(seed)
    fields = rng.standard_normal((K, G, G))
    fy = np.fft.fftfreq(G)[:, None] * 2 * np.pi
    fx = np.fft.rfftfreq(G)[None, :] * 2 * np.pi
    ker = np.exp(-0.5 * (smooth ** 2) * (fx * fx + fy * fy))
    sm = np.fft.irfft2(np.fft.rfft2(fields, axes=(1, 2)) * ker[None], axes=(1, 2))
    sm = (sm - sm.mean((1, 2), keepdims=True)) / (sm.std((1, 2), keepdims=True) + 1e-8)
    if bias:
        sm = sm + rng.uniform(0.0, float(bias), size=(K, 1, 1))
    coords = (pts * (G - 1)).T
    vals = np.stack([map_coordinates(sm[kk], coords, order=1, mode="nearest") for kk in range(K)])
    return np.argmax(vals, axis=0).astype(np.int64)


def boundary_f1_edges(pred, true, e):
    bp = pred[e[:, 0]] != pred[e[:, 1]]
    bt = true[e[:, 0]] != true[e[:, 1]]
    den = int(bp.sum()) + int(bt.sum())
    return 1.0 if den == 0 else float(2.0 * int((bp & bt).sum()) / den)


def main():
    pts = np.random.default_rng(49).random((N_PTS, 2))
    e_und = knn_edges(pts, KNN)                          # unique undirected pairs (GC convention)
    e_dir = np.vstack([e_und, e_und[:, ::-1]])           # canonical directed pairs (MF convention:
    #                                                      _nbr_sum_edges expects both directions)
    deg = 2.0 * len(e_und) / N_PTS
    print(f"n={N_PTS} undirected={len(e_und)} directed={len(e_dir)} mean_degree={deg:.2f}", flush=True)
    cfgs = [
        ("wellspec", 5, 0.7, 2.0, 1.0),
        ("stress", 5, 1.0, 1.3, 2.5),
        ("wellspec", 8, 0.7, 2.0, 1.0),
        ("stress", 8, 1.0, 1.3, 2.5),
    ]
    rows = []
    for regime, K, sigma, muc, beta in cfgs:
        truth = field_domains_on_points(pts, K, seed=11 + K)
        y, mu_true = gaussian_emission(truth, K, D, mu_scale=muc, sigma=sigma, seed=12 + K)
        y2d = y.astype(np.float64)
        q, mu_em, sig_hat, iters, elbo = em_mf_gaussian_graph(y2d, K, beta, e_dir, restarts=2,
                                                              max_iter=300, tol=1e-4, seed=3)
        log_u = gaussian_log_unaries(y2d, mu_em, sig_hat)
        cost = -log_u
        t0 = time.perf_counter()
        qf, itf = mean_field_graph(log_u, beta, e_dir, max_iter=2000, tol=1e-5)
        t_mf = time.perf_counter() - t0
        mf_mode = np.argmax(qf, axis=1)
        t0 = time.perf_counter()
        gc_mode = alpha_expansion_graph(cost, beta, e_und, max_sweeps=10, seed=7)
        t_gc = time.perf_counter() - t0
        row = dict(regime=regime, K=K, sigma=sigma, mu_scale=muc, beta=beta, n=N_PTS,
                   edges=len(e_und), mean_degree=round(deg, 2),
                   ari_mf=round(adjusted_rand_index(truth, mf_mode), 4),
                   ari_gc=round(adjusted_rand_index(truth, gc_mode), 4),
                   bf1_mf=round(boundary_f1_edges(mf_mode, truth, e_und), 4),
                   bf1_gc=round(boundary_f1_edges(gc_mode, truth, e_und), 4),
                   e_mf=round(potts_energy_graph(mf_mode, cost, beta, e_und[:, 0], e_und[:, 1]), 1),
                   e_gc=round(potts_energy_graph(gc_mode, cost, beta, e_und[:, 0], e_und[:, 1]), 1),
                   t_mf_s=round(t_mf, 2), t_gc_s=round(t_gc, 2), it_mf=int(itf))
        print(row, flush=True)
        rows.append(row)
    out = _os.path.join(_ROOT, "experiments/h3c-knn-topology/results/h3c_results.csv")
    _os.makedirs(_os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("DONE")


if __name__ == "__main__":
    main()
