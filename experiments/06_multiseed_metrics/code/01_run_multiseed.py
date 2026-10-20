"""Multi-seed benchmark with unified metrics (ARI/NMI/boundary-F1).

Author : Tao Zhu <zhutao@tjh.tjmu.edu.cn>
Created: 2026-10-02
Updated: 2026-10-20

Inputs
------
Synthetic instances generated in memory; seeds from configuration.
Configuration: config/defaults.yaml, loaded through hmrf.config.

Outputs
-------
CSV with per-seed metrics and summary statistics.

Usage
-----
    python 01_run_multiseed.py [options]
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

import sys, os, time, csv, resource
sys.path.insert(0, os.path.join(_ROOT, "src"))
import numpy as np
from hmrf.simulate import gaussian_rf_domains, gaussian_emission
from hmrf.solvers import em_mf_gaussian, adjusted_rand_index

SMOKE = len(sys.argv) > 1 and sys.argv[1] == "smoke"
K, d, BETA, SIGMA, MUC = 8, 20, 0.7, 0.5, 2.0
if SMOKE:
    SCALES, INST_SEEDS, N_RESTART = (100,), (11, 21), 2
else:
    SCALES, INST_SEEDS, N_RESTART = (100, 250, 500, 750, 1000), (11, 21, 31, 41, 51), 3

OUT = str(results_dir("06_multiseed_metrics"))
os.makedirs(OUT, exist_ok=True)


def nmi(a, b, k):
    """NMI (arithmetic-mean normalization, same as the sklearn default)."""
    a = np.asarray(a).ravel().astype(np.int64); b = np.asarray(b).ravel().astype(np.int64)
    ct = np.bincount(a * k + b, minlength=k * k).reshape(k, k).astype(np.float64)
    n = ct.sum(); p = ct / n
    pi = p.sum(1); pj = p.sum(0)
    denom = (pi[:, None] * pj[None, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(p > 0, p * np.log(p / (denom + 1e-300) + 1e-300), 0.0)
    I = t.sum()
    Ha = -np.sum(pi[pi > 0] * np.log(pi[pi > 0]))
    Hb = -np.sum(pj[pj > 0] * np.log(pj[pj > 0]))
    h = (Ha + Hb) / 2.0
    return 1.0 if h <= 0 else float(I / h)


def grid_edges(S):
    """Undirected grid edges (right neighbor + down neighbor), integer id."""
    idx = np.arange(S * S, dtype=np.int64).reshape(S, S)
    e0 = np.concatenate([idx[:, :-1].ravel(), idx[:-1, :].ravel()])
    e1 = np.concatenate([idx[:, 1:].ravel(), idx[1:, :].ravel()])
    return e0, e1


def boundary_f1(pred, true, e0, e1):
    """Strict edge-set F1: boundary edges = adjacent pairs whose endpoint labels differ."""
    bp = pred.ravel()[e0] != pred.ravel()[e1]
    bt = true.ravel()[e0] != true.ravel()[e1]
    den = int(bp.sum()) + int(bt.sum())
    if den == 0:
        return 1.0
    return float(2.0 * int((bp & bt).sum()) / den)


def main():
    long_path = os.path.join(OUT, "h2d_results.csv")
    f_long = open(long_path, "w", newline="")
    w_long = csv.DictWriter(f_long, fieldnames=["S", "n", "inst_seed", "restart", "ari", "nmi", "bf1",
                                                "t_s", "iters", "elbo", "sigma_hat"])
    w_long.writeheader()
    rows = []
    for S in SCALES:
        e0, e1 = grid_edges(S)
        for iseed in INST_SEEDS:
            t0 = time.perf_counter()
            truth = gaussian_rf_domains(S, K, seed=iseed)
            y, mu_true = gaussian_emission(truth, K, d, mu_scale=MUC, sigma=SIGMA, seed=iseed + 1)
            t_sim = time.perf_counter() - t0
            y2d = y.reshape(-1, d).astype(np.float64)
            del y
            runs = []
            for r in range(N_RESTART):
                t0 = time.perf_counter()
                q, mu, sh, iters, elbo = em_mf_gaussian(y2d, K, BETA, S, S, restarts=1, max_iter=300, seed=3 + r)
                dt = time.perf_counter() - t0
                pred = np.argmax(q, axis=2)
                del q
                rec = dict(S=S, n=S * S, inst_seed=iseed, restart=r,
                           ari=round(adjusted_rand_index(truth, pred), 4),
                           nmi=round(nmi(truth, pred, K), 4),
                           bf1=round(boundary_f1(pred, truth, e0, e1), 4),
                           t_s=round(dt, 2), iters=iters, elbo=round(float(elbo), 1),
                           sigma_hat=round(float(sh), 3))
                runs.append(rec)
                w_long.writerow(rec); f_long.flush()
                print(rec, flush=True)
            aris = [x["ari"] for x in runs]
            eb = int(np.argmax([x["elbo"] for x in runs]))
            mx = int(np.argmax(aris))
            peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024  # macOS: bytes -> MB
            row = dict(S=S, n=S * S, inst_seed=iseed, t_sim_s=round(t_sim, 2),
                       ari_mean3=round(float(np.mean(aris)), 4), ari_sd3=round(float(np.std(aris)), 4),
                       ari_max=round(float(np.max(aris)), 4), ari_median=round(float(np.median(aris)), 4),
                       ari_elbo=round(runs[eb]["ari"], 4),
                       nmi_max=round(runs[mx]["nmi"], 4), bf1_max=round(runs[mx]["bf1"], 4),
                       nmi_elbo=round(runs[eb]["nmi"], 4), bf1_elbo=round(runs[eb]["bf1"], 4),
                       t_total3_s=round(sum(x["t_s"] for x in runs), 2),
                       iters_join="/".join(str(x["iters"]) for x in runs),
                       sigma_hat_mean=round(float(np.mean([x["sigma_hat"] for x in runs])), 3),
                       proc_peak_mb_cum=round(peak_mb, 2))
            rows.append(row)
            print("INSTANCE", row, flush=True)
    f_long.close()
    # summary table (aggregated across instances per S)
    sum_path = os.path.join(OUT, "h2d_summary.csv")
    with open(sum_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["S", "n", "n_inst",
                                          "ari_mean3_mean", "ari_mean3_sd",
                                          "ari_max_mean", "ari_max_sd",
                                          "ari_elbo_mean", "ari_elbo_sd",
                                          "nmi_max_mean", "nmi_max_sd",
                                          "bf1_max_mean", "bf1_max_sd",
                                          "t_total3_mean_s", "t_total3_sd_s"])
        w.writeheader()
        for S in SCALES:
            sub = [r for r in rows if r["S"] == S]
            w.writerow(dict(
                S=S, n=S * S, n_inst=len(sub),
                ari_mean3_mean=round(float(np.mean([r["ari_mean3"] for r in sub])), 4),
                ari_mean3_sd=round(float(np.std([r["ari_mean3"] for r in sub])), 4),
                ari_max_mean=round(float(np.mean([r["ari_max"] for r in sub])), 4),
                ari_max_sd=round(float(np.std([r["ari_max"] for r in sub])), 4),
                ari_elbo_mean=round(float(np.mean([r["ari_elbo"] for r in sub])), 4),
                ari_elbo_sd=round(float(np.std([r["ari_elbo"] for r in sub])), 4),
                nmi_max_mean=round(float(np.mean([r["nmi_max"] for r in sub])), 4),
                nmi_max_sd=round(float(np.std([r["nmi_max"] for r in sub])), 4),
                bf1_max_mean=round(float(np.mean([r["bf1_max"] for r in sub])), 4),
                bf1_max_sd=round(float(np.std([r["bf1_max"] for r in sub])), 4),
                t_total3_mean_s=round(float(np.mean([r["t_total3_s"] for r in sub])), 2),
                t_total3_sd_s=round(float(np.std([r["t_total3_s"] for r in sub])), 2)))
    print("DONE h2d; long:", long_path, "summary:", sum_path, flush=True)


if __name__ == "__main__":
    main()