"""Compute the informativeness ceiling over the beta grid for both NB parameterizations.

Author : Tao Zhu <zhutao@tjh.tjmu.edu.cn>
Created: 2026-10-02
Updated: 2026-10-20

Inputs
------
Synthetic NB instances generated from configured factors.
Configuration: config/defaults.yaml, loaded through hmrf.config.

Outputs
-------
results/h2e_results.csv with ARI per beta and grid size.

Usage
-----
    python 01_run_beta_ceiling.py [options]
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

sys.path.insert(0, os.path.join(_ROOT, "src"))
from hmrf.simulate import gaussian_rf_domains, nb_emission_domains
from hmrf.solvers import adjusted_rand_index, mean_field_potts, nb_log_unaries

OLD_FACTORS = CFG["experiments"]["h2e"]["old_factors"]
NEW_FACTORS = CFG["experiments"]["h2e"]["new_factors"]
GRID = CFG["experiments"]["h2e"]["beta_grid"]

rows = []
for tag, factors, disp, S, d in (
    ("old_h2", OLD_FACTORS, 1.0, 250, 10),
    ("old_h2", OLD_FACTORS, 1.0, 500, 10),
    ("new_h2b", NEW_FACTORS, 3.0, 250, 20),
    ("new_h2b", NEW_FACTORS, 3.0, 500, 20),
    ("new_h2b", NEW_FACTORS, 3.0, 1000, 20),
):
    truth = gaussian_rf_domains(S, 8, seed=41)
    y, mu_true = nb_emission_domains(truth, d=d, factors=factors, disp=disp, base_scale=1.0, seed=42)
    y2d = y.reshape(-1, d).astype(np.float64)
    frac0 = float((y2d == 0).mean())
    log_u = nb_log_unaries(y2d, mu_true, disp).reshape(S, S, 8)
    for b in GRID:
        t0 = time.perf_counter()
        q, it = mean_field_potts(log_u, b, max_iter=1000, tol=1e-3)
        dt = time.perf_counter() - t0
        ari = adjusted_rand_index(truth, np.argmax(q, axis=2))
        rows.append(dict(params=tag, S=S, n=S * S, frac_zero=round(frac0, 3), d=d, beta=b,
                         ari_oracle=round(float(ari), 4), it=it, t_s=round(dt, 2)))
        print(rows[-1], flush=True)

out = os.path.join(_ROOT, "experiments/07_nb_ceiling_grid/results/h2e_results.csv")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print("== max over grid ==")
for tag in ("old_h2", "new_h2b"):
    for S in sorted({r["S"] for r in rows if r["params"] == tag}):
        rs = [r for r in rows if r["params"] == tag and r["S"] == S]
        best = max(rs, key=lambda r: r["ari_oracle"])
        print(f"{tag} S={S}: max ARI* = {best['ari_oracle']} at beta={best['beta']}")
print("DONE")
