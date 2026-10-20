"""Check convergence of VI under the NB emission model.

Author : Tao Zhu <zhutao@tjh.tjmu.edu.cn>
Created: 2026-10-02
Updated: 2026-10-20

Inputs
------
Synthetic NB counts generated in memory.
Configuration: config/defaults.yaml, loaded through hmrf.config.

Outputs
-------
CSV with ARI and iteration counts.

Usage
-----
    python 01_run_nb_convergence.py [options]
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
from hmrf.simulate import gaussian_rf_domains, nb_emission_domains
from hmrf.solvers import em_mf_nb_restarts, mean_field_potts, nb_log_unaries, adjusted_rand_index

K, d, DISP = 8, 20, 3.0
FACTORS = [0.8, 1.1, 1.5, 2.0, 2.7, 3.7, 5.0, 6.8]
rows = []
for S in (250, 500):
    truth = gaussian_rf_domains(S, K, seed=41)
    y, mu_true = nb_emission_domains(truth, d=d, factors=FACTORS, disp=DISP, base_scale=1.0, seed=42)
    y2d = y.reshape(-1, d).astype(np.float64)
    log_u = nb_log_unaries(y2d, mu_true, DISP).reshape(S, S, K)
    t0 = time.perf_counter()
    qo, ito = mean_field_potts(log_u, 0.7, max_iter=1000, tol=1e-3)
    t_o = time.perf_counter() - t0
    ari_o = adjusted_rand_index(truth, np.argmax(qo, axis=2))
    for beta in (0.5, 0.7, 1.0):
        t0 = time.perf_counter()
        q, mu, iters, obj = em_mf_nb_restarts(y2d, K, beta, S, S, disp=DISP, max_iter=300, tol=1e-3, restarts=3, seed=5)
        t_vi = time.perf_counter() - t0
        ari = adjusted_rand_index(truth, np.argmax(q, axis=2))
        rows.append(dict(S=S, beta=beta, ari_oracle=round(ari_o, 4), ari_viem=round(ari, 4),
                         t_viem_s=round(t_vi, 2), iters=iters))
        print(rows[-1], flush=True)
with open(os.path.join(_ROOT, "experiments/05_nb_convergence/results/h2c_results.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("DONE")