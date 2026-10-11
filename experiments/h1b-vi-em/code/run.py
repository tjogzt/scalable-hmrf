# --- portable project root (no hardcoded user paths) ---
import os as _os, pathlib as _pl
def _find_root(_start):
    _p = _pl.Path(_start).resolve()
    for _anc in [_p] + list(_p.parents):
        if (_anc / "src" / "hmrf_solvers.py").exists():
            return str(_anc)
    raise RuntimeError("project root not found; run inside the repository")
_ROOT = _os.environ.get("HMRF_ROOT") or _find_root(__file__)
# ------------------------------------------------------
"""H1b: VI-EM joint estimation (no oracle) vs VI-oracle vs Gibbs-oracle."""
import sys, time, csv
sys.path.insert(0, _os.path.join(_ROOT, "src"))
import numpy as np
from simulate import sample_potts, gaussian_emission
from hmrf_solvers import mean_field_potts, checkerboard_gibbs, adjusted_rand_index, gaussian_log_unaries, em_mf_gaussian

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
with open(_os.path.join(_ROOT, "experiments/h1b-vi-em/results/h1b_results.csv"), "w", newline="") as f:
    wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wcsv.writeheader(); wcsv.writerows(rows)
print("DONE")