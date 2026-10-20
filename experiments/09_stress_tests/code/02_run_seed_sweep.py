"""Seed sweep across stress settings and beta values.

Author : Tao Zhu <zhutao@tjh.tjmu.edu.cn>
Created: 2026-10-02
Updated: 2026-10-20

Inputs
------
Synthetic instances generated in memory.
Configuration: config/defaults.yaml, loaded through hmrf.config.

Outputs
-------
CSV with per-seed ARI and aggregate statistics.

Usage
-----
    python 02_run_seed_sweep.py [options]
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
import time

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(_ROOT, "src"))
from hmrf.simulate import gaussian_emission, gaussian_rf_domains
from hmrf.solvers import (adjusted_rand_index, em_mf_gaussian,
                          gaussian_log_unaries, mean_field_potts)
from hmrf.graphcut import alpha_expansion, potts_energy

SEEDS = tuple(CFG["experiments"]["h3b"]["seeds"])
SETTINGS = CFG["experiments"]["h3b"]["settings"]
BETAS = tuple(CFG["experiments"]["h3b"]["betas"])
S, K, d = 100, 8, 20

rows = []
for sigma, muc in SETTINGS:
    for beta in BETAS:
        for iseed in SEEDS:
            t0 = time.perf_counter()
            truth = gaussian_rf_domains(S, K, seed=iseed)
            y, mu_true = gaussian_emission(truth, K, d, mu_scale=muc, sigma=sigma, seed=iseed + 1)
            y2d = y.reshape(-1, d).astype(np.float64)
            q, mu_em, sigma_hat, iters, elbo = em_mf_gaussian(y2d, K, beta, S, S, restarts=2, seed=3)
            log_u = gaussian_log_unaries(y2d, mu_em, sigma_hat).reshape(S, S, K)
            cost = -log_u
            qf, itf = mean_field_potts(log_u, beta)
            mf_mode = np.argmax(qf, axis=2)
            gc_mode = alpha_expansion(cost, beta, max_sweeps=10, seed=7)
            ari_mf = adjusted_rand_index(truth, mf_mode)
            ari_gc = adjusted_rand_index(truth, gc_mode)
            row = dict(sigma=sigma, muc=muc, beta=beta, iseed=iseed,
                       ari_mf=round(float(ari_mf), 4), ari_gc=round(float(ari_gc), 4),
                       d_ari=round(float(ari_gc - ari_mf), 4),
                       e_mf=round(potts_energy(mf_mode, cost, beta, S, S), 1),
                       e_gc=round(potts_energy(gc_mode, cost, beta, S, S), 1), it_mf=int(itf),
                       t_s=round(time.perf_counter() - t0, 1))
            rows.append(row)
            print(row, flush=True)

out = os.path.join(_ROOT, "experiments/09_stress_tests/results/h3b_seeds_results.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

# ---- per-setting dispersion + paired analysis ----
print("\n== per-setting dispersion (d_ari = ARI_gc - ARI_mf, 5 instance seeds) ==")
summ = []
for sigma, muc in SETTINGS:
    for beta in BETAS:
        d = np.array([r["d_ari"] for r in rows if r["sigma"] == sigma and r["muc"] == muc and r["beta"] == beta])
        summ.append(dict(sigma=sigma, muc=muc, beta=beta, n=len(d),
                         d_mean=round(float(d.mean()), 4), d_sd=round(float(d.std(ddof=1)), 4),
                         d_min=round(float(d.min()), 4), d_max=round(float(d.max()), 4)))
        print(summ[-1], flush=True)

all_d = np.array([r["d_ari"] for r in rows])
wil = stats.wilcoxon(all_d)
ci = stats.t.interval(0.90, len(all_d) - 1, loc=all_d.mean(), scale=stats.sem(all_d))
print(f"\npaired over {len(all_d)} runs: mean d = {all_d.mean():+.4f}, sd = {all_d.std(ddof=1):.4f}, "
      f"max |d| = {np.abs(all_d).max():.4f}")
print(f"Wilcoxon signed-rank: W={wil.statistic}, p={wil.pvalue:.3f}")
print(f"90% CI of mean d: [{ci[0]:+.4f}, {ci[1]:+.4f}]  (equivalence margin +/-0.01: "
      f"{'within' if ci[0] > -0.01 and ci[1] < 0.01 else 'NOT within'})")
frac_lower_energy = np.mean([r["e_gc"] <= r["e_mf"] for r in rows])
print(f"graph cut energy <= mean field energy in {frac_lower_energy*100:.0f}% of runs")

summ_out = os.path.join(_ROOT, "experiments/09_stress_tests/results/h3b_seeds_summary.csv")
with open(summ_out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summ[0].keys()))
    w.writeheader()
    w.writerows(summ)
print("DONE")
