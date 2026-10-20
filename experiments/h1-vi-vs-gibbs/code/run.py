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
"""H1: MF-VI vs Gibbs correctness comparison (Gaussian emission, oracle parameters)."""
import sys, time, csv
sys.path.insert(0, _os.path.join(_ROOT, "src"))
import numpy as np
from simulate import sample_potts, gaussian_emission
from hmrf_solvers import mean_field_potts, checkerboard_gibbs, adjusted_rand_index

H, W, K, d = 30, 30, 5, 20
rows = []
for seed in (0, 1, 2):
    truth = sample_potts(H, W, K, beta=1.0, sweeps=3000, seed=seed)
    y, mu = gaussian_emission(truth, K, d, mu_scale=1.5, sigma=0.8, seed=seed)
    # oracle unary terms
    sigma2 = 0.8 ** 2
    ysq = (y ** 2).sum(1, keepdims=True)
    log_u = -0.5 * d * np.log(2 * np.pi * sigma2) - (ysq - 2 * y @ mu.T + (mu ** 2).sum(1)[None, :]) / (2 * sigma2)
    log_u = log_u.reshape(H, W, K)
    for beta in (0.2, 0.5, 1.0):
        t0 = time.perf_counter()
        q, it = mean_field_potts(log_u, beta)
        t_vi = time.perf_counter() - t0
        vi_mode = np.argmax(q, axis=2)
        t0 = time.perf_counter()
        g_mode, _ = checkerboard_gibbs(log_u, beta, sweeps=1500, burn=500, seed=seed)
        t_g = time.perf_counter() - t0
        rows.append(dict(seed=seed, beta=beta, ari_vi=round(adjusted_rand_index(truth, vi_mode), 4),
                         ari_gibbs=round(adjusted_rand_index(truth, g_mode), 4),
                         ari_vi_gibbs=round(adjusted_rand_index(vi_mode, g_mode), 4),
                         t_vi_s=round(t_vi, 3), t_gibbs_s=round(t_g, 2), vi_iters=it))
        print(rows[-1])
# beta mismatch: true beta=0.5, model beta=1.5
truth = sample_potts(H, W, K, beta=0.5, sweeps=3000, seed=7)
y, mu = gaussian_emission(truth, K, d, mu_scale=1.5, sigma=0.8, seed=7)
sigma2 = 0.8 ** 2
ysq = (y ** 2).sum(1, keepdims=True)
log_u = -0.5 * d * np.log(2 * np.pi * sigma2) - (ysq - 2 * y @ mu.T + (mu ** 2).sum(1)[None, :]) / (2 * sigma2)
log_u = log_u.reshape(H, W, K)
q, it = mean_field_potts(log_u, 1.5)
vi_mode = np.argmax(q, axis=2)
g_mode, _ = checkerboard_gibbs(log_u, 1.5, sweeps=1500, burn=500, seed=7)
rows.append(dict(seed='mismatch', beta='0.5->1.5', ari_vi=round(adjusted_rand_index(truth, vi_mode), 4),
                 ari_gibbs=round(adjusted_rand_index(truth, g_mode), 4),
                 ari_vi_gibbs=round(adjusted_rand_index(vi_mode, g_mode), 4),
                 t_vi_s=round(t_vi if False else 0.0, 3), t_gibbs_s=0.0, vi_iters=it))
print(rows[-1])
with open(_os.path.join(_ROOT, "experiments/h1-vi-vs-gibbs/results/h1_results.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print("DONE rows=", len(rows))