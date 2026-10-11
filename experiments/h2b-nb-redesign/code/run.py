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
"""H2b: 重设 NB 仿真参数（可辨识上限>0.9）后 VI-EM 对标 oracle 上限。"""
import sys, time, csv
sys.path.insert(0, _os.path.join(_ROOT, "src"))
import numpy as np
from simulate import gaussian_rf_domains, nb_emission_domains
from hmrf_solvers import em_mf_nb, mean_field_potts, nb_log_unaries, adjusted_rand_index

K, d, DISP, BETA = 8, 20, 3.0, 0.7
FACTORS = [0.8, 1.1, 1.5, 2.0, 2.7, 3.7, 5.0, 6.8]
rows = []
for S in (250, 1000):
    truth = gaussian_rf_domains(S, K, seed=41)
    y, mu_true = nb_emission_domains(truth, d=d, factors=FACTORS, disp=DISP, base_scale=1.0, seed=42)
    y2d = y.reshape(-1, d).astype(np.float64)
    frac0 = (y2d == 0).mean()
    # oracle 上限：真 mu 一元项 + 纯 MF
    log_u = nb_log_unaries(y2d, mu_true, DISP).reshape(S, S, K)
    t0 = time.perf_counter()
    qo, ito = mean_field_potts(log_u, BETA, max_iter=1000, tol=1e-3)
    t_o = time.perf_counter() - t0
    ari_o = adjusted_rand_index(truth, np.argmax(qo, axis=2))
    # VI-EM（disp 固定为真值）
    t0 = time.perf_counter()
    q, mu, iters = em_mf_nb(y2d, K, BETA, S, S, disp=DISP, max_iter=300, tol=1e-3, seed=5)
    t_vi = time.perf_counter() - t0
    ari_vi = adjusted_rand_index(truth, np.argmax(q, axis=2))
    rows.append(dict(S=S, n=S * S, frac_zero=round(float(frac0), 3), ari_oracle=round(ari_o, 4),
                     ari_viem=round(ari_vi, 4), t_oracle_s=round(t_o, 2), t_viem_s=round(t_vi, 2),
                     it_oracle=ito, it_viem=iters))
    print(rows[-1], flush=True)
with open(_os.path.join(_ROOT, "experiments/h2b-nb-redesign/results/h2b_results.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("DONE")
