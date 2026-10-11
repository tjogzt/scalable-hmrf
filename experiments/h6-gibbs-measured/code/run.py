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
"""H6: Gibbs 基线实测（多点每扫成本 + 250K/562.5K/1M 全量锚点）——替代 h2 的外推。

- 每扫成本模型：S ∈ {50,100,150,200,250,300} 实测 per-sweep（含在线计数），线性+幂律双拟合，报 R²；
- 全量锚点：与原外推口径一致（1500 sweeps、burn=500、oracle 发射参数、beta=0.7）；
- 在线计数版本与原 checkerboard_gibbs 语义一致（后验模态），内存 O(nK) 计数替代样本栈；
- 预算守卫：锚点超过 budget 即停，按已测每扫成本外推并如实标注。
用法: python run.py [smoke]
"""
import sys, os, time, csv, json
sys.path.insert(0, _os.path.join(_ROOT, "src"))
import numpy as np
from simulate import gaussian_rf_domains, gaussian_emission
from hmrf_solvers import gaussian_log_unaries, adjusted_rand_index

SMOKE = len(sys.argv) > 1 and sys.argv[1] == "smoke"
K, d, BETA, SIGMA, MUC = 8, 20, 0.7, 0.5, 2.0
FIT_SIZES = (50, 100, 150, 200, 250, 300)
FIT_SWEEPS = {50: 100, 100: 60, 150: 40, 200: 30, 250: 20, 300: 20}
if SMOKE:
    FIT_SIZES = (50, 100)
    FIT_SWEEPS = {50: 30, 100: 20}

OUT = _os.path.join(_ROOT, "experiments/h6-gibbs-measured/results")
os.makedirs(OUT, exist_ok=True)


def build_instance(S, seed_truth=11, seed_noise=12):
    truth = gaussian_rf_domains(S, K, seed=seed_truth)
    y, mu_true = gaussian_emission(truth, K, d, mu_scale=MUC, sigma=SIGMA, seed=seed_noise)
    log_u = gaussian_log_unaries(y.reshape(-1, d).astype(np.float64), mu_true, SIGMA).reshape(S, S, K)
    del y
    return truth, log_u


def run_gibbs_timed(log_u, beta, sweeps, burn, seed=5, budget_s=None, verbose_every=0):
    """与原 checkerboard_gibbs 相同的每步工作量；后验模态用在线计数（内存 O(nK)）。
    返回 (mode, total_s, achieved_sweeps, counts_sum)。"""
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
    # ---------- 每扫成本（多点） ----------
    cost_rows = []
    for S in FIT_SIZES:
        truth, log_u = build_instance(S, seed_truth=31, seed_noise=32)
        _ = run_gibbs_timed(log_u, BETA, sweeps=2, burn=0, seed=9)   # 热身
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

    # ---------- 全量锚点 ----------
    anchors = []
    plans = [(500, 600), (750, 900), (1000, 1800)]  # (S, budget_s)；1500 sweeps 目标
    if SMOKE:
        plans = [(500, 120)]
    for S, budget in plans:
        truth, log_u = build_instance(S)
        sweeps = 200 if SMOKE else 1500  # 统一 1500（与 h2 外推口径一致）
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
        # 增量落盘
        if anchors:
            with open(os.path.join(OUT, "h6_anchors.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(anchors[0].keys())); w.writeheader(); w.writerows(anchors)
        del log_u
    meta["anchors"] = anchors
    meta["old_extrapolation_reference"] = {"S500_est_s": 204, "S1000_est_s": 817,
                                           "note": "h2 analysis 的线性外推值，供对照"}
    with open(os.path.join(OUT, "h6_meta.json"), "w") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
    print("DONE h6", flush=True)


if __name__ == "__main__":
    main()
