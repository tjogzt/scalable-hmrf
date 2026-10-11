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
"""H2d: 多种子规模曲线 + NMI/边界F1 补全（统一 restarts=3（max_iter=300）、每起点分别记录）。

设计（回应审查意见）：
- 5 个独立实例种子 × 5 个规模档 × 3 起点；报 mean±sd、max、median、ELBO 选择值；
- 指标补全：NMI（算术平均归一化）+ 边界 F1（严格边集：相邻域对的集合 F1）；
- 全部数字落 CSV（长表 + 汇总表），修复"复核值未回写"的追溯问题。
用法: python run.py [smoke]
"""
import sys, os, time, csv, resource
sys.path.insert(0, _os.path.join(_ROOT, "src"))
import numpy as np
from simulate import gaussian_rf_domains, gaussian_emission
from hmrf_solvers import em_mf_gaussian, adjusted_rand_index

SMOKE = len(sys.argv) > 1 and sys.argv[1] == "smoke"
K, d, BETA, SIGMA, MUC = 8, 20, 0.7, 0.5, 2.0
if SMOKE:
    SCALES, INST_SEEDS, N_RESTART = (100,), (11, 21), 2
else:
    SCALES, INST_SEEDS, N_RESTART = (100, 250, 500, 750, 1000), (11, 21, 31, 41, 51), 3

OUT = _os.path.join(_ROOT, "experiments/h2d-multiseed-metrics/results")
os.makedirs(OUT, exist_ok=True)


def nmi(a, b, k):
    """NMI（arithmetic-mean 归一化，与 sklearn 默认一致）。"""
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
    """无向网格边（右邻 + 下邻），整数 id。"""
    idx = np.arange(S * S, dtype=np.int64).reshape(S, S)
    e0 = np.concatenate([idx[:, :-1].ravel(), idx[:-1, :].ravel()])
    e1 = np.concatenate([idx[:, 1:].ravel(), idx[1:, :].ravel()])
    return e0, e1


def boundary_f1(pred, true, e0, e1):
    """严格边集 F1：边界边 = 两端点标签不同的相邻对。"""
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
    # 汇总表（按 S 跨实例聚合）
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
