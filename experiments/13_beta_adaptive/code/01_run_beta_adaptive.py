"""Test adaptive beta selection on synthetic domains.

Author : Tao Zhu <zhutao@tjh.tjmu.edu.cn>
Created: 2026-10-02
Updated: 2026-10-20

Inputs
------
Synthetic instances generated in memory.
Configuration: config/defaults.yaml, loaded through hmrf.config.

Outputs
-------
CSV with ARI per beta rule.

Usage
-----
    python 01_run_beta_adaptive.py [options]
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

import sys, time, csv
sys.path.insert(0, os.path.join(_ROOT, "src"))
import numpy as np
from hmrf.simulate import gaussian_rf_domains, gaussian_emission, nb_emission_domains
from hmrf.solvers import em_mf_gaussian, em_mf_nb_restarts, adjusted_rand_index

S, K, d = 250, 8, 20
GRID = (0.3, 0.5, 0.7, 1.0, 1.3, 1.7)
rows = []
# Gaussian scenario
truth = gaussian_rf_domains(S, K, seed=11)
y, mu_true = gaussian_emission(truth, K, d, mu_scale=2.0, sigma=0.5, seed=12)
y2d = y.reshape(-1, d).astype(np.float64)
best = None; per = []
for b in GRID:
    q, mu, sh, iters, elbo = em_mf_gaussian(y2d, K, b, S, S, restarts=2, seed=3)
    ari = adjusted_rand_index(truth, np.argmax(q, axis=2))
    per.append((b, ari, elbo))
    if best is None or elbo > best[1]:
        best = (b, ari, elbo)
print("GAUSS per-beta:", [(b, round(a,3)) for b,a,e in per])
print("GAUSS ELBO-selected:", best[0], "ARI", round(best[1],4))
# NB scenario
FACTORS = [0.8, 1.1, 1.5, 2.0, 2.7, 3.7, 5.0, 6.8]
truth = gaussian_rf_domains(S, K, seed=41)
y, mu_true = nb_emission_domains(truth, d=d, factors=FACTORS, disp=3.0, base_scale=1.0, seed=42)
y2d = y.reshape(-1, d).astype(np.float64)
best2 = None; per2 = []
for b in GRID:
    q, mu, iters, elbo = em_mf_nb_restarts(y2d, K, b, S, S, disp=3.0, max_iter=300, tol=1e-3, restarts=3, seed=5)
    ari = adjusted_rand_index(truth, np.argmax(q, axis=2))
    per2.append((b, ari, elbo))
    if best2 is None or elbo > best2[1]:
        best2 = (b, ari, elbo)
print("NB per-beta:", [(b, round(a,3)) for b,a,e in per2])
print("NB ELBO-selected:", best2[0], "ARI", round(best2[1],4))
rows = [dict(emission='gauss', beta=float(b), ari=round(a,4), elbo=round(e,1)) for b,a,e in per] + \
       [dict(emission='nb', beta=float(b), ari=round(a,4), elbo=round(e,1)) for b,a,e in per2]
with open(os.path.join(_ROOT, "experiments/13_beta_adaptive/results/h5_results.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("DONE")
