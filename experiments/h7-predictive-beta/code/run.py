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
"""H7: 留出预测似然做 beta/模型选择（回应真实数据上 ELBO 失效）。

设计：随机遮罩 5% spot；用其余 spot 拟合（遮罩 spot 无一元项，仅靠邻域场）；
在遮罩 spot 上评估预测对数似然 ll_i = log Σ_k q_i(k) · N(y_i | mu_k, sigma²)；
β* = argmax 平均留出 ll；与「真值 ARI 最优 β」「ELBO 最优 β」对比（ARI/ELBO 仅用于评估/对照）。
用法: python run.py [smoke]
"""
import sys, os, time, csv, json
ROOT = _ROOT
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "experiments/h9-dlpfc12-benchmark/code"))
import numpy as np
from hmrf_solvers import gaussian_log_unaries, adjusted_rand_index
from run_ours import load_slice, kmeans_pp

OUT = os.path.join(ROOT, "experiments/h7-predictive-beta/results")
os.makedirs(OUT, exist_ok=True)
SMOKE = len(sys.argv) > 1 and sys.argv[1] == "smoke"
MASKS = (7,) if SMOKE else (7, 8, 9)
MASK_FRAC = 0.05
BETAS = (0.3, 0.5, 0.7, 1.0, 1.5)
N_INIT = 1 if SMOKE else 2
SLICES = ["151673"] if SMOKE else ["151673", "151509", "151671"]


def _nbr_sum(q, edges):
    out = np.zeros_like(q)
    np.add.at(out, edges[:, 0], q[edges[:, 1]])
    return out


def masked_em_mf_gaussian(y, k, beta, edges, mask, max_iter=300, tol=1e-4, seed=0, n_init=2):
    """遮罩感知的图版 MF-EM（高斯发射）。遮罩 spot 无一元项、不参与 M 步；
    返回 (q, mu, sigma, iters, elbo_unmasked, heldout_ll_per_spot)。"""
    n, d = y.shape
    best = None
    for r in range(n_init):
        rng = np.random.default_rng(seed + 100 * r)
        unm = np.where(~mask)[0]
        sub = y[unm]
        lab_sub = kmeans_pp(sub, k, rng)
        lab = np.empty(n, dtype=np.int64)
        # 遮罩 spot 用最近类均值初始化
        mu0 = np.array([sub[lab_sub == kk].mean(0) if (lab_sub == kk).any() else sub[rng.integers(0, len(sub))]
                        for kk in range(k)])
        lab[unm] = lab_sub
        lab[mask] = np.argmin(((y[mask][:, None, :] - mu0[None, :, :]) ** 2).sum(2), axis=1)
        mu = np.zeros((k, d))
        for kk in range(k):
            m = lab == kk
            mu[kk] = y[m].mean(0) if m.any() else y[rng.integers(0, n)]
        sigma = 1.0
        q = np.zeros((n, k))
        log_u = gaussian_log_unaries(y, mu, sigma)
        log_u[mask] = 0.0
        q = np.exp(log_u - log_u.max(axis=1, keepdims=True)); q /= q.sum(axis=1, keepdims=True)
        ysq = (y ** 2).sum(1)
        it = 0
        for it in range(max_iter):
            nbr = _nbr_sum(q, edges)
            logq = log_u + beta * nbr
            logq -= logq.max(axis=1, keepdims=True)
            new = np.exp(logq); new /= new.sum(axis=1, keepdims=True)
            dq = np.abs(new - q).max()
            q = new
            w = q.copy(); w[mask] = 0.0
            mu = (w.T @ y) / (w.sum(0)[:, None] + 1e-12)
            sig2 = (w.T @ ysq) / (w.sum(0) * d + 1e-12) - (mu ** 2).sum(1) / d
            sigma = float(np.sqrt(max(sig2.max(), 0.2 ** 2)))
            sigma = min(sigma, float(np.sqrt((y ** 2).mean())))
            log_u = gaussian_log_unaries(y, mu, sigma)
            log_u[mask] = 0.0
            if dq < tol and it > 10:
                break
        # ELBO（未遮罩部分，标准 MF 公式）
        w = q.copy(); w[mask] = 0.0
        nbr = _nbr_sum(q, edges)
        e_logp = float((w * log_u).sum())
        e_prior = 0.5 * beta * float((q * nbr).sum())
        ent = float(-(q * np.log(q + 1e-300)).sum())
        elbo = e_logp + e_prior + ent
        # 留出预测对数似然（遮罩 spot）
        ll = np.zeros(int(mask.sum()))
        s2 = sigma ** 2
        const = -0.5 * d * np.log(2 * np.pi * s2)
        ym = y[mask]
        ysqm = (ym ** 2).sum(1, keepdims=True)
        llk = const - (ysqm - 2 * ym @ mu.T + (mu ** 2).sum(1)[None, :]) / (2 * s2)
        qm = q[mask]
        mx = llk.max(1, keepdims=True)
        ll = (mx[:, 0] + np.log((qm * np.exp(llk - mx)).sum(1) + 1e-300)).mean()
        if best is None or ll > best[0]:
            best = (ll, q.copy(), mu.copy(), sigma, it + 1, elbo)
    return best[1], best[2], best[3], best[4], best[5], float(best[0])


def grid_edges(S):
    idx = np.arange(S * S, dtype=np.int64).reshape(S, S)
    e0 = np.concatenate([idx[:, :-1].ravel(), idx[:-1, :].ravel()])
    e1 = np.concatenate([idx[:, 1:].ravel(), idx[1:, :].ravel()])
    return np.stack([e0, e1], axis=1)


def hex_edges(rows, cols):
    pos2i = {(r, c): i for i, (r, c) in enumerate(zip(rows, cols))}
    e = []
    for i, (r, c) in enumerate(zip(rows, cols)):
        for dr, dc in ((0, 2), (0, -2), (1, -1), (1, 1), (-1, -1), (-1, 1)):
            j = pos2i.get((r + dr, c + dc))
            if j is not None:
                e.append((i, j))
    return np.array(e, dtype=np.int64)


def boundary_ratio(lab, edges):
    return float((lab[edges[:, 0]] != lab[edges[:, 1]]).mean())


def feature_pipeline(X):
    lx = np.log1p(X / (X.sum(1, keepdims=True) + 1e-9) * 1e4)
    m = lx.mean(0); v = lx.var(0); use = m > 1e-3
    lm = np.log10(m[use]); lv = np.log10(np.maximum(v[use], 1e-12))
    coef = np.polyfit(lm, lv, 2); resid = lv - np.polyval(coef, lm)
    gidx = np.where(use)[0][np.argsort(resid)[::-1][:3000]]
    Y = lx[:, gidx].astype(np.float64)
    Yz = (Y - Y.mean(0, keepdims=True)) / (Y.std(0, keepdims=True) + 1e-6)
    U, S, Vt = np.linalg.svd(Yz, full_matrices=False)
    PC = U[:, :50] * S[:50]
    PC = PC - PC.mean(0, keepdims=True); PC = PC / (PC.std(0, keepdims=True) + 1e-6)
    return PC


def run_case(name, y, k, edges, truth=None):
    """返回每个 beta 的汇总 + 三个准则的选择。"""
    rows = []
    for beta in BETAS:
        lls, elbos, brs, aris = [], [], [], []
        for mseed in MASKS:
            rng = np.random.default_rng(mseed)
            mask = rng.random(len(y)) < MASK_FRAC
            t0 = time.perf_counter()
            q, mu, sig, iters, elbo, ll = masked_em_mf_gaussian(y, k, beta, edges, mask,
                                                                seed=11, n_init=N_INIT)
            dt = time.perf_counter() - t0
            lab = np.argmax(q, axis=1)
            rec = dict(case=name, beta=beta, mask_seed=mseed, heldout_ll=round(ll, 2),
                       elbo=round(elbo, 1), boundary_ratio=round(boundary_ratio(lab, edges), 5),
                       iters=iters, t_s=round(dt, 2))
            if truth is not None:
                rec["ari"] = round(adjusted_rand_index(truth, lab), 4)
                aris.append(rec["ari"])
            rows.append(rec)
            lls.append(ll); elbos.append(elbo); brs.append(rec["boundary_ratio"])
        print(rec | {"ll_mean": round(float(np.mean(lls)), 2)}, flush=True)
    # 汇总（跨 mask 平均）
    by = {}
    for beta in BETAS:
        sub = [r for r in rows if r["beta"] == beta]
        by[beta] = dict(ll=float(np.mean([r["heldout_ll"] for r in sub])),
                        elbo=float(np.mean([r["elbo"] for r in sub])),
                        br=float(np.mean([r["boundary_ratio"] for r in sub])))
        if truth is not None:
            by[beta]["ari"] = float(np.mean([r["ari"] for r in sub]))
    sel_ll = max(by, key=lambda b: by[b]["ll"])
    sel_elbo = max(by, key=lambda b: by[b]["elbo"])
    brs = {b: by[b]["br"] for b in BETAS}
    rmin = min(brs.values())
    sel_elbow = min([b for b in BETAS if brs[b] <= 1.12 * rmin])
    out = dict(case=name, selected_by_heldout_ll=sel_ll, selected_by_elbo=sel_elbo,
               selected_by_elbow=sel_elbow, per_beta={str(b): {k: round(v, 4) for k, v in by[b].items()} for b in BETAS})
    if truth is not None:
        out["best_by_ari"] = max(by, key=lambda b: by[b]["ari"])
    print("SUMMARY", json.dumps(out, ensure_ascii=False), flush=True)
    return rows, out


def main():
    all_rows, all_sums = [], []
    # ---- DLPFC 切片 ----
    for sid in SLICES:
        X, rows_, cols_, truth = load_slice(sid)
        k = int(np.unique(truth).size)
        PC = feature_pipeline(X); del X
        edges = hex_edges(rows_, cols_)
        print(f"### {sid} n={len(truth)} k={k}", flush=True)
        r, s = run_case(f"dlpfc_{sid}", PC, k, edges, truth=truth)
        all_rows += r; all_sums.append(s)
    # ---- 高斯仿真（真值已知的对照） ----
    sys.path.insert(0, _os.path.join(_ROOT, "src"))
    from simulate import gaussian_rf_domains, gaussian_emission
    S = 250
    truth_s = gaussian_rf_domains(S, 8, seed=11)
    ys, _ = gaussian_emission(truth_s, 8, 20, mu_scale=2.0, sigma=0.5, seed=12)
    y2 = ys.reshape(-1, 20).astype(np.float64)
    eg = grid_edges(S)
    print(f"### sim S={S}", flush=True)
    r, s = run_case("gauss_sim_S250", y2, 8, eg, truth=truth_s.ravel())
    all_rows += r; all_sums.append(s)
    with open(os.path.join(OUT, "h7_runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader(); w.writerows(all_rows)
    with open(os.path.join(OUT, "h7_summary.json"), "w") as f:
        json.dump(all_sums, f, indent=1, ensure_ascii=False)
    print("DONE h7", flush=True)


if __name__ == "__main__":
    main()
