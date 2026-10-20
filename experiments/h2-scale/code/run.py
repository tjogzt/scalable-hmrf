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
"""H2: scale-accuracy-time curves (kill test kt1) + NB count emission + Gibbs extrapolation comparison."""
import sys, time, csv, resource
sys.path.insert(0, _os.path.join(_ROOT, "src"))
import numpy as np
from simulate import gaussian_rf_domains, gaussian_emission, nb_emission_domains
from hmrf_solvers import em_mf_gaussian, em_mf_nb, checkerboard_gibbs, gaussian_log_unaries, adjusted_rand_index

K, d, BETA, SIGMA, MUC = 8, 20, 0.7, 0.5, 2.0
rows = []

print("== Gaussian emission scale curves ==")
for S in (100, 250, 500, 750, 1000):
    t0 = time.perf_counter()
    truth = gaussian_rf_domains(S, K, seed=11)
    y, mu_true = gaussian_emission(truth, K, d, mu_scale=MUC, sigma=SIGMA, seed=12)
    t_sim = time.perf_counter() - t0
    y2d = y.reshape(-1, d).astype(np.float64)
    t0 = time.perf_counter()
    q, mu, sigma_hat, iters, elbo = em_mf_gaussian(y2d, K, BETA, S, S, restarts=1, max_iter=150, seed=3)
    t_vi = time.perf_counter() - t0
    ari = adjusted_rand_index(truth, np.argmax(q, axis=2))
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
    n_dom = len(np.unique(truth))
    rows.append(dict(emission='gauss', S=S, n=S * S, n_domains=n_dom, ari=round(ari, 4), t_sim_s=round(t_sim, 1),
                     t_vi_s=round(t_vi, 2), t_per_iter_ms=round(t_vi / max(iters, 1) * 1e3, 1),
                     iters=iters, peak_gb=round(rss, 2)))
    print(rows[-1], flush=True)

print("== NB count emission ==")
factors = [0.25, 0.5, 1.0, 2.0, 4.0, 0.35, 1.5, 3.0]
for S in (500, 1000):
    t0 = time.perf_counter()
    truth = gaussian_rf_domains(S, K, seed=21)
    y, mu_true = nb_emission_domains(truth, d=10, factors=factors, disp=1.0, base_scale=1.0, seed=22)
    t_sim = time.perf_counter() - t0
    y2d = y.reshape(-1, 10).astype(np.float64)
    t0 = time.perf_counter()
    q, mu, iters = em_mf_nb(y2d, K, BETA, S, S, disp=1.0, max_iter=150, seed=4)
    t_vi = time.perf_counter() - t0
    ari = adjusted_rand_index(truth, np.argmax(q, axis=2))
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
    n_dom = len(np.unique(truth))
    rows.append(dict(emission='nb', S=S, n=S * S, n_domains=n_dom, ari=round(ari, 4), t_sim_s=round(t_sim, 1),
                     t_vi_s=round(t_vi, 2), t_per_iter_ms=round(t_vi / max(iters, 1) * 1e3, 1),
                     iters=iters, peak_gb=round(rss, 2)))
    print(rows[-1], flush=True)

print("== Gibbs per-sweep time extrapolation ==")
gibbs_pts = []
for S in (50, 100):
    truth = gaussian_rf_domains(S, K, seed=31)
    y, mu_true = gaussian_emission(truth, K, d, mu_scale=MUC, sigma=SIGMA, seed=32)
    log_u = gaussian_log_unaries(y.reshape(-1, d).astype(np.float64), mu_true, SIGMA).reshape(S, S, K)
    t0 = time.perf_counter()
    checkerboard_gibbs(log_u, BETA, sweeps=10, burn=0, seed=5)
    dt = (time.perf_counter() - t0) / 10
    gibbs_pts.append((S * S, dt))
    print(dict(S=S, t_per_sweep_s=round(dt, 3)), flush=True)
x = np.array([p[0] for p in gibbs_pts], dtype=float)
t = np.array([p[1] for p in gibbs_pts], dtype=float)
a, b = np.polyfit(x, t, 1)
print(f"fit: t_sweep = {a:.3e} * n + {b:.4f}")
for S in (500, 1000):
    n = S * S
    t_est = (a * n + b) * 1500  # 1500 sweeps reference
    vi_row = [r for r in rows if r['S'] == S and r['emission'] == 'gauss'][0]
    print(dict(S=S, gibbs_est_1500sweeps_s=round(t_est, 1), vi_s=vi_row['t_vi_s'], speedup_x=round(t_est / vi_row['t_vi_s'], 1)))
with open(_os.path.join(_ROOT, "experiments/h2-scale/results/h2_results.csv"), "w", newline="") as f:
    wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wcsv.writeheader(); wcsv.writerows(rows)
print("DONE")