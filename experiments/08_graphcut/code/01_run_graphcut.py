"""Benchmark alpha-expansion graph cuts against the Potts energy.

Author : Tao Zhu <zhutao@tjh.tjmu.edu.cn>
Created: 2026-10-02
Updated: 2026-10-20

Inputs
------
Synthetic Potts instances generated in memory.
Configuration: config/defaults.yaml, loaded through hmrf.config.

Outputs
-------
CSV with ARI and energy values.

Usage
-----
    python 01_run_graphcut.py [options]
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
from hmrf.simulate import gaussian_rf_domains, gaussian_emission
from hmrf.solvers import em_mf_gaussian, mean_field_potts, gaussian_log_unaries, adjusted_rand_index
from hmrf.graphcut import alpha_expansion, potts_energy

K, d, BETA, SIGMA, MUC = 8, 20, 0.7, 0.5, 2.0
rows = []
for S in (100, 250):
    truth = gaussian_rf_domains(S, K, seed=11)
    y, mu_true = gaussian_emission(truth, K, d, mu_scale=MUC, sigma=SIGMA, seed=12)
    y2d = y.reshape(-1, d).astype(np.float64)
    t0 = time.perf_counter()
    q, mu_em, sigma_hat, iters, elbo = em_mf_gaussian(y2d, K, BETA, S, S, restarts=1, seed=3)
    t_em = time.perf_counter() - t0
    log_u = gaussian_log_unaries(y2d, mu_em, sigma_hat).reshape(S, S, K)
    # mean field (same beta)
    t0 = time.perf_counter()
    qf, itf = mean_field_potts(log_u, BETA)
    t_mf = time.perf_counter() - t0
    mf_mode = np.argmax(qf, axis=2)
    # graph cut (cost = -log_u)
    cost = -log_u
    t0 = time.perf_counter()
    gc_mode = alpha_expansion(cost, BETA, max_sweeps=10, seed=7)
    t_gc = time.perf_counter() - t0
    row = dict(S=S, n=S * S,
               ari_mf=round(adjusted_rand_index(truth, mf_mode), 4),
               ari_gc=round(adjusted_rand_index(truth, gc_mode), 4),
               e_mf=round(potts_energy(mf_mode, cost, BETA, S, S), 1),
               e_gc=round(potts_energy(gc_mode, cost, BETA, S, S), 1),
               t_em_s=round(t_em, 2), t_mf_s=round(t_mf, 3), t_gc_s=round(t_gc, 2),
               it_mf=itf)
    print(row, flush=True)
    rows.append(row)
with open(os.path.join(_ROOT, "experiments/08_graphcut/results/h3_results.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("DONE")
