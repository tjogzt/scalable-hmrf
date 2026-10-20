"""Measure exact Gibbs sampling cost against grid size.

Author : Tao Zhu <zhutao@tjh.tjmu.edu.cn>
Created: 2026-10-02
Updated: 2026-10-20

Inputs
------
Synthetic Potts instances generated in memory.
Configuration: config/defaults.yaml, loaded through hmrf.config.

Outputs
-------
results/h6_results.csv.

Usage
-----
    python 01_measure_gibbs_timing.py [options]
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

import sys, os, time, csv, json
sys.path.insert(0, os.path.join(_ROOT, "src"))
import numpy as np
from hmrf.simulate import gaussian_rf_domains, gaussian_emission
from hmrf.solvers import gaussian_log_unaries, adjusted_rand_index

SMOKE = len(sys.argv) > 1 and sys.argv[1] == "smoke"
K, d, BETA, SIGMA, MUC = 8, 20, 0.7, 0.5, 2.0
FIT_SIZES = tuple(CFG["experiments"]["h6"]["fit_sizes"])
FIT_SWEEPS = CFG["experiments"]["h6"]["fit_sweeps"]
if SMOKE:
    FIT_SIZES = FIT_SIZES[:2]
    FIT_SWEEPS = {50: 30, 100: 20}

OUT = str(results_dir("15_gibbs_timing"))
os.makedirs(OUT, exist_ok=True)


def build_instance(S, seed_truth=11, seed_noise=12):
    truth = gaussian_rf_domains(S, K, seed=seed_truth)
    y, mu_true = gaussian_emission(truth, K, d, mu_scale=MUC, sigma=SIGMA, seed=seed_noise)
    log_u = gaussian_log_unaries(y.reshape(-1, d).astype(np.float64), mu_true, SIGMA).reshape(S, S, K)
    del y
    return truth, log_u


def run_gibbs_timed(log_u, beta, sweeps, burn, seed=5, budget_s=None, verbose_every=0):
    """Same per-step workload as the original checkerboard_gibbs; posterior mode via online counting (memory O(nK)).
    Returns (mode, total_s, achieved_sweeps, counts_sum)."""
    rng = np.random.default_rng(seed)
    H, W, Kk = log_u.shape
    Y, X = np.mgrid[0:H, 0:W]
    parity = (Y + X) % 2
    labels = np.argmax(log_u, axis=2)
    counts = np.zeros((H, W, Kk), dtype=np.int64)
    t0 = time.perf_counter()
    achieved = 0
    for s in range(sweeps):
        for p in (0, 1):
            mask = parity == p
            oh = np.zeros((H, W, Kk))
            oh[Y, X, labels] = 1.0
            cnt = np.zeros((H, W, Kk))
            for ax, sgn in [(0, 1), (0, -1), (1, 1), (1, -1)]:
                t = np.roll(oh, sgn, axis=ax)
                if ax == 0:
                    if sgn == 1: t[0, :, :] = 0.0
                    else: t[-1, :, :] = 0.0
                else:
                    if sgn == 1: t[:, 0, :] = 0.0
                    else: t[:, -1, :] = 0.0
                cnt += t
            logp = log_u + beta * cnt
            logp -= logp.max(axis=2, keepdims=True)
            pmat = np.exp(logp)
            pmat /= pmat.sum(axis=2, keepdims=True)
            g = -np.log(-np.log(rng.random((H, W, Kk)) + 1e-30))
            new = np.argmax(np.log(pmat + 1e-300) + g, axis=2)
            labels[mask] = new[mask]
        if s >= burn:
            counts[Y, X, labels] += 1
        achieved = s + 1
        if verbose_every and achieved % verbose_every == 0:
            el = time.perf_counter() - t0
            print(f"    ... {achieved}/{sweeps} sweeps, {el:.0f}s, {el/achieved*1000:.0f} ms/sweep", flush=True)
        if budget_s is not None and (time.perf_counter() - t0) > budget_s:
            print(f"    !! budget hit at {achieved}/{sweeps} sweeps", flush=True)
            break
    total = time.perf_counter() - t0
    if counts.sum() == 0:
        mode = labels.copy()
    else:
        mode = np.argmax(counts, axis=2)
    return mode, total, achieved, int(counts.sum())


def main():
    meta = {"fit_sizes": list(FIT_SIZES), "beta": BETA, "sweeps_anchor": 1500, "burn": 500}
    # ---------- Per-sweep cost (multiple sizes) ----------
    cost_rows = []
    for S in FIT_SIZES:
        truth, log_u = build_instance(S, seed_truth=31, seed_noise=32)
        _ = run_gibbs_timed(log_u, BETA, sweeps=2, burn=0, seed=9)   # warm-up
        mode, total, achieved, _ = run_gibbs_timed(log_u, BETA, sweeps=FIT_SWEEPS[S], burn=0, seed=9)
        per = total / achieved
        cost_rows.append(dict(S=S, n=S * S, sweeps_measured=achieved, per_sweep_s=round(per, 5), total_s=round(total, 2)))
        print("COST", cost_rows[-1], flush=True)
        del log_u
    n_arr = np.array([r["n"] for r in cost_rows], dtype=np.float64)
    t_arr = np.array([r["per_sweep_s"] for r in cost_rows], dtype=np.float64)
    a, b = np.polyfit(n_arr, t_arr, 1)
    pred = a * n_arr + b
    ss_res = float(((t_arr - pred) ** 2).sum()); ss_tot = float(((t_arr - t_arr.mean()) ** 2).sum())
    r2_lin = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    pA, ppow = np.polyfit(np.log(n_arr), np.log(t_arr), 1)
    predp = np.exp(pA) * n_arr ** ppow
    ss_res_p = float(((t_arr - predp) ** 2).sum())
    r2_pow = 1 - ss_res_p / ss_tot if ss_tot > 0 else 1.0
    meta["fit_linear"] = dict(a=float(a), b=float(b), r2=float(r2_lin))
    meta["fit_power"] = dict(A=float(np.exp(pA)), p=float(ppow), r2=float(r2_pow))
    for S in (500, 750, 1000):
        meta[f"fit_linear_pred_per_sweep_S{S}"] = float(a * S * S + b)
    print("FIT linear:", meta["fit_linear"], "power:", meta["fit_power"], flush=True)
    with open(os.path.join(OUT, "h6_sweepcost.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cost_rows[0].keys())); w.writeheader(); w.writerows(cost_rows)

    # ---------- Full anchors ----------
    anchors = []
    plans = [(500, 600), (750, 900), (1000, 1800)]  # (S, budget_s); target 1500 sweeps
    if SMOKE:
        plans = [(500, 120)]
    for S, budget in plans:
        truth, log_u = build_instance(S)
        sweeps = 200 if SMOKE else 1500  # use 1500 uniformly (matches the h2 extrapolation protocol)
        burn = int(sweeps * 500 / 1500)
        print(f"ANCHOR S={S} sweeps={sweeps} budget={budget}s start", flush=True)
        try:
            mode, total, achieved, _ = run_gibbs_timed(log_u, BETA, sweeps=sweeps, burn=burn,
                                                       seed=5, budget_s=budget, verbose_every=100)
            ari = round(adjusted_rand_index(truth, mode), 4)
            per = total / achieved
            rec = dict(S=S, n=S * S, sweeps_target=sweeps, sweeps_achieved=achieved, burn=burn,
                       total_s=round(total, 2), per_sweep_s=round(per, 6), ari=ari,
                       budget_hit=bool(achieved < sweeps))
            print("ANCHOR", rec, flush=True)
            anchors.append(rec)
        except Exception as e:
            print(f"ANCHOR S={S} FAILED: {type(e).__name__} {e}", flush=True)
        # incremental write
        if anchors:
            with open(os.path.join(OUT, "h6_anchors.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(anchors[0].keys())); w.writeheader(); w.writerows(anchors)
        del log_u
    meta["anchors"] = anchors
    meta["old_extrapolation_reference"] = {"S500_est_s": 204, "S1000_est_s": 817,
                                           "note": "linear extrapolation from the h2 analysis, for reference"}
    with open(os.path.join(OUT, "h6_meta.json"), "w") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
    print("DONE h6", flush=True)


if __name__ == "__main__":
    main()
