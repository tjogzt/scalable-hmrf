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
"""H2e: informativeness ceiling over the beta grid for BOTH NB parameterizations.

Answers review items R7/M4: the abstract's 0.47 ceiling comes from the original (h2) NB
parameterization; recompute each parameterization's ceiling as the maximum over beta
grid (true parameters + pure mean field), so the ceiling is no longer a single-beta value.

Old (h2): factors in [0.25, 4.0] -> near-zero low-expression domains; disp=1.0
New (h2b): factors in [0.8, 6.8]; disp=3.0
Usage: python run.py
"""
import csv
import sys
import time

import numpy as np

sys.path.insert(0, _os.path.join(_ROOT, "src"))
from simulate import gaussian_rf_domains, nb_emission_domains
from hmrf_solvers import adjusted_rand_index, mean_field_potts, nb_log_unaries

OLD_FACTORS = [0.25, 0.5, 1.0, 2.0, 4.0, 0.35, 1.5, 3.0]
NEW_FACTORS = [0.8, 1.1, 1.5, 2.0, 2.7, 3.7, 5.0, 6.8]
GRID = [0.3, 0.5, 0.7, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0]

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

out = _os.path.join(_ROOT, "experiments/h2e-nb-ceiling-grid/results/h2e_results.csv")
_os.makedirs(_os.path.dirname(out), exist_ok=True)
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
