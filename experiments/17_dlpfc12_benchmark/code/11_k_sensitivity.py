"""K-sensitivity analysis on one DLPFC section.

Author : Tao Zhu <zhutao@tjh.tjmu.edu.cn>
Created: 2026-10-02
Updated: 2026-10-20

Inputs
------
data/dlpfc12/151673/ and configuration.
Configuration: config/defaults.yaml, loaded through hmrf.config.

Outputs
-------
results/k_sensitivity.csv.

Usage
-----
    python 11_k_sensitivity.py [options]
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

import csv
import sys

import numpy as np

sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "experiments", "h9-dlpfc12-benchmark", "code"))
from hmrf.solvers import adjusted_rand_index, em_mf_gaussian_graph
from run_ours import kmeans_pp, load_slice

SID = CFG["experiments"]["h9"]["k_sensitivity"]["slice"]
BETA = CFG["experiments"]["h9"]["k_sensitivity"]["beta"]
N_INIT = CFG["experiments"]["h9"]["k_sensitivity"]["n_init"]


def pipeline(sid):
    X, rows, cols, truth = load_slice(sid)
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
    return PC, truth, np.array(edges, dtype=np.int64)


PC, truth, edges = pipeline(SID)
rows_out = []
for K in range(5, 11):
    aris = []
    for t in range(N_INIT):
        rng = np.random.default_rng(100 + t)
        init = kmeans_pp(PC, K, rng)
        q, mu, sig, iters, elbo = em_mf_gaussian_graph(PC, K, BETA, edges, restarts=1,
                                                       max_iter=300, tol=1e-4, seed=t, init_labels=init)
        aris.append(float(adjusted_rand_index(truth, np.argmax(q, axis=1))))
    row = dict(slice=SID, K=K, beta=BETA, n_init=N_INIT,
               ari_best=round(max(aris), 4), ari_median=round(float(np.median(aris)), 4),
               ari_list="/".join(f"{a:.4f}" for a in aris))
    rows_out.append(row)
    print(row, flush=True)

out = os.path.join(_ROOT, "experiments/17_dlpfc12_benchmark/results/k_sensitivity.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
    w.writeheader()
    w.writerows(rows_out)
print("DONE")
