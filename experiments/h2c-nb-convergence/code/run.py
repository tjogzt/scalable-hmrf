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
"""H2c: NB VI-EM multi-restart + beta sweep, benchmarked against the oracle ceiling."""
import sys, time, csv
sys.path.insert(0, _os.path.join(_ROOT, "src"))
import numpy as np
from simulate import gaussian_rf_domains, nb_emission_domains
from hmrf_solvers import em_mf_nb_restarts, mean_field_potts, nb_log_unaries, adjusted_rand_index

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
with open(_os.path.join(_ROOT, "experiments/h2c-nb-convergence/results/h2c_results.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("DONE")