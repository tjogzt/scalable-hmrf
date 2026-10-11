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
"""H3b: stress regime (high beta / weak unaries) graph cut vs mean field."""
import sys, time, csv
sys.path.insert(0, _os.path.join(_ROOT, "src"))
import numpy as np
from simulate import gaussian_rf_domains, gaussian_emission
from hmrf_solvers import em_mf_gaussian, mean_field_potts, gaussian_log_unaries, adjusted_rand_index
from graphcut_hmrf import alpha_expansion, potts_energy

S, K, d = 100, 8, 20
rows = []
for sigma, muc in ((0.7, 1.3), (1.0, 1.3), (0.7, 2.0)):
    for beta in (1.5, 2.5):
        truth = gaussian_rf_domains(S, K, seed=11)
        y, mu_true = gaussian_emission(truth, K, d, mu_scale=muc, sigma=sigma, seed=12)
        y2d = y.reshape(-1, d).astype(np.float64)
        q, mu_em, sigma_hat, iters, elbo = em_mf_gaussian(y2d, K, beta, S, S, restarts=2, seed=3)
        log_u = gaussian_log_unaries(y2d, mu_em, sigma_hat).reshape(S, S, K)
        cost = -log_u
        t0 = time.perf_counter()
        qf, itf = mean_field_potts(log_u, beta)
        t_mf = time.perf_counter() - t0
        mf_mode = np.argmax(qf, axis=2)
        t0 = time.perf_counter()
        gc_mode = alpha_expansion(cost, beta, max_sweeps=10, seed=7)
        t_gc = time.perf_counter() - t0
        row = dict(sigma=sigma, muc=muc, beta=beta,
                   ari_mf=round(adjusted_rand_index(truth, mf_mode), 4),
                   ari_gc=round(adjusted_rand_index(truth, gc_mode), 4),
                   e_mf=round(potts_energy(mf_mode, cost, beta, S, S), 1),
                   e_gc=round(potts_energy(gc_mode, cost, beta, S, S), 1),
                   t_mf_ms=round(t_mf * 1e3, 1), t_gc_s=round(t_gc, 2), it_mf=itf)
        print(row, flush=True)
        rows.append(row)
with open(_os.path.join(_ROOT, "experiments/h3b-stress/results/h3b_results.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("DONE")
