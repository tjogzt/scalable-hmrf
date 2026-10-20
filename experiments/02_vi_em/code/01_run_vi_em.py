"""Run the variational-EM solver on a synthetic Gaussian Potts model.

Author : Tao Zhu <zhutao@tjh.tjmu.edu.cn>
Created: 2026-10-02
Updated: 2026-10-20

Inputs
------
Synthetic Gaussian emissions generated in memory.
Configuration: config/defaults.yaml, loaded through hmrf.config.

Outputs
-------
CSV with recovered labels, ARI and iteration counts.

Usage
-----
    python 01_run_vi_em.py [options]
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
from hmrf.simulate import sample_potts, gaussian_emission
from hmrf.solvers import mean_field_potts, checkerboard_gibbs, adjusted_rand_index, gaussian_log_unaries, em_mf_gaussian

H = W = 40; K, d = 5, 20
SIGMA, MUC, BETA = 0.5, 2.0, 0.8
rows = []
for seed in (0, 1, 2):
    truth = sample_potts(H, W, K, beta=1.0, sweeps=3000, seed=seed)
    y, mu_true = gaussian_emission(truth, K, d, mu_scale=MUC, sigma=SIGMA, seed=seed)
    y2d = y.reshape(-1, d).astype(np.float64)
    # VI-EM
    t0 = time.perf_counter()
    q, mu_em, sigma_em, iters, elbo = em_mf_gaussian(y2d, K, BETA, H, W, restarts=3, seed=seed)
    t_viem = time.perf_counter() - t0
    vi_em_mode = np.argmax(q, axis=2)
    # VI-oracle
    log_u = gaussian_log_unaries(y2d, mu_true, SIGMA).reshape(H, W, K)
    t0 = time.perf_counter(); qo, ito = mean_field_potts(log_u, BETA); t_vio = time.perf_counter() - t0
    vio_mode = np.argmax(qo, axis=2)
    # Gibbs-oracle
    t0 = time.perf_counter()
    g_mode, _ = checkerboard_gibbs(log_u, BETA, sweeps=1500, burn=500, seed=seed)
    t_g = time.perf_counter() - t0
    row = dict(seed=seed,
               ari_viem=round(adjusted_rand_index(truth, vi_em_mode), 4),
               ari_vio=round(adjusted_rand_index(truth, vio_mode), 4),
               ari_gibbs=round(adjusted_rand_index(truth, g_mode), 4),
               ari_viem_gibbs=round(adjusted_rand_index(vi_em_mode, g_mode), 4),
               sigma_hat=round(float(sigma_em), 3), t_viem_s=round(t_viem, 2),
               t_vio_ms=round(t_vio * 1e3, 1), t_gibbs_s=round(t_g, 1), iters=iters)
    print(row); rows.append(row)
with open(os.path.join(_ROOT, "experiments/02_vi_em/results/h1b_results.csv"), "w", newline="") as f:
    wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wcsv.writeheader(); wcsv.writerows(rows)
print("DONE")