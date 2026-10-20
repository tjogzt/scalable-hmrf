"""Probe VI convergence trajectories for the multi-seed study.

Author : Tao Zhu <zhutao@tjh.tjmu.edu.cn>
Created: 2026-10-02
Updated: 2026-10-20

Inputs
------
Synthetic instances generated in memory.
Configuration: config/defaults.yaml, loaded through hmrf.config.

Outputs
-------
CSV with per-iteration objective values.

Usage
-----
    python 02_probe_convergence.py [options]
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

import sys, time
sys.path.insert(0, os.path.join(_ROOT, "src"))
import numpy as np
from hmrf.simulate import gaussian_rf_domains, gaussian_emission
from hmrf.solvers import em_mf_gaussian, adjusted_rand_index

K, d, BETA, SIGMA, MUC = 8, 20, 0.7, 0.5, 2.0


def inst(S, iseed):
    truth = gaussian_rf_domains(S, K, seed=iseed)
    y, _ = gaussian_emission(truth, K, d, mu_scale=MUC, sigma=SIGMA, seed=iseed + 1)
    return truth, y.reshape(-1, d).astype(np.float64)


print("== A: S=100 inst=21, max_iter=3000 ==", flush=True)
truth, y2d = inst(100, 21)
for r in (0, 1, 2):
    t0 = time.perf_counter()
    q, mu, sh, iters, elbo = em_mf_gaussian(y2d, K, BETA, 100, 100, restarts=1, max_iter=3000, seed=3 + r)
    print(dict(r=r, iters=iters, ari=round(adjusted_rand_index(truth, np.argmax(q, axis=2)), 4),
               sigma_hat=round(float(sh), 3), t=round(time.perf_counter() - t0, 1)), flush=True)

print("== B: S=250 capped cases, max_iter=3000 ==", flush=True)
for iseed, r in ((51, 0), (41, 2)):
    truth, y2d = inst(250, iseed)
    t0 = time.perf_counter()
    q, mu, sh, iters, elbo = em_mf_gaussian(y2d, K, BETA, 250, 250, restarts=1, max_iter=3000, seed=3 + r)
    print(dict(iseed=iseed, r=r, iters=iters, ari=round(adjusted_rand_index(truth, np.argmax(q, axis=2)), 4),
               sigma_hat=round(float(sh), 3), t=round(time.perf_counter() - t0, 1)), flush=True)

print("== C: S=750 inst=11 escape attempts, seeds 6..11, max_iter=300 ==", flush=True)
truth, y2d = inst(750, 11)
for seed in range(6, 12):
    t0 = time.perf_counter()
    q, mu, sh, iters, elbo = em_mf_gaussian(y2d, K, BETA, 750, 750, restarts=1, max_iter=300, seed=seed)
    ari = round(adjusted_rand_index(truth, np.argmax(q, axis=2)), 4)
    print(dict(seed=seed, iters=iters, ari=ari, sigma_hat=round(float(sh), 3),
               t=round(time.perf_counter() - t0, 1)), flush=True)
print("PROBE DONE", flush=True)