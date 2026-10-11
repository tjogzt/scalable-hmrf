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
"""Batch 1 convergence probe: distinguish "slow convergence" from "stable suboptimal basin".
A) S=100 inst21 (all 3 restarts hit the 300 cap): max_iter=3000 -> does it converge, ARI;
B) S=250 inst51 r0 / inst41 r2 (hit the cap): max_iter=3000;
C) S=750 inst11 (0.9627 stable basin): extra restarts seeds 6-11, see how many restarts escaping the basin takes.
"""
import sys, time
sys.path.insert(0, _os.path.join(_ROOT, "src"))
import numpy as np
from simulate import gaussian_rf_domains, gaussian_emission
from hmrf_solvers import em_mf_gaussian, adjusted_rand_index

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