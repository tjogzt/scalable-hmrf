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
"""H9: 12 切片 DLPFC 基准——本方法（VI-HMRF）全部切片运行。
加载器与 h4d 特征管线逐行一致（VST-HVG3000 + z-score + PCA50 + kmeans++ 5 起点, beta 网格）。
用法: python run_ours.py [slice_id | all | smoke]
"""
import sys, os, time, csv, json, gzip
sys.path.insert(0, _os.path.join(_ROOT, "src"))
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from hmrf_solvers import em_mf_gaussian_graph, mean_field_graph, gaussian_log_unaries, adjusted_rand_index

ROOT = _ROOT
DATA = os.path.join(ROOT, "data/dlpfc12")
OUT = os.path.join(ROOT, "experiments/h9-dlpfc12-benchmark/results")
LAB = os.path.join(OUT, "labels")
os.makedirs(LAB, exist_ok=True)
K, N_PC = 7, 50
BETAS = (0.3, 0.5, 0.7, 1.0)
N_INIT = 5
ALL_SLICES = ["151507", "151508", "151509", "151510", "151669", "151670",
              "151671", "151672", "151673", "151674", "151675", "151676"]


def nmi(a, b, k):
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


def boundary_f1_edges(pred, true, edges):
    bp = pred[edges[:, 0]] != pred[edges[:, 1]]
    bt = true[edges[:, 0]] != true[edges[:, 1]]
    den = int(bp.sum()) + int(bt.sum())
    if den == 0:
        return 1.0
    return float(2.0 * int((bp & bt).sum()) / den)


def load_slice(sid):
    d = os.path.join(DATA, sid)
    with gzip.open(os.path.join(d, "counts.mtx.gz"), "rb") as fh:
        m = sio.mmread(fh).tocsr()          # genes x spots
    X = np.asarray(m.T.todense(), dtype=np.float32)  # spots x genes
    spots = np.genfromtxt(os.path.join(d, "spots.csv"), delimiter=",", dtype=None,
                          names=True, encoding="utf-8")
    rows = spots["row"].astype(np.int64); cols = spots["col"].astype(np.int64)
    labels_str = spots["label"].astype(str)
    classes = sorted(set(labels_str)); truth = np.array([classes.index(t) for t in labels_str])
    return X, rows, cols, truth


def kmeans_pp(Xm, k, rng):
    c = [int(rng.integers(0, len(Xm)))]
    for _ in range(k - 1):
        d2 = ((Xm[:, None, :] - Xm[c][None, :, :]) ** 2).sum(2).min(1)
        c.append(int(rng.choice(len(Xm), p=(d2 + 1e-12) / (d2.sum() + 1e-12))))
    mu = Xm[c].copy()
    for _ in range(30):
        lb = np.argmin(((Xm[:, None, :] - mu[None, :, :]) ** 2).sum(2), axis=1)
        for kk in range(k):
            mm = lb == kk
            if mm.any(): mu[kk] = Xm[mm].mean(0)
    return lb


def run_slice(sid, quick=False):
    t0 = time.perf_counter()
    X, rows, cols, truth = load_slice(sid)
    n = X.shape[0]
    k = int(np.unique(truth).size)  # 按切片自适应类数（151669-672 仅 5 层）
    lx = np.log1p(X / (X.sum(1, keepdims=True) + 1e-9) * 1e4)
    del X
    m = lx.mean(0); v = lx.var(0); use = m > 1e-3
    lm = np.log10(m[use]); lv = np.log10(np.maximum(v[use], 1e-12))
    coef = np.polyfit(lm, lv, 2); resid = lv - np.polyval(coef, lm)
    gidx = np.where(use)[0][np.argsort(resid)[::-1][:3000]]
    Y = lx[:, gidx].astype(np.float64)
    Yz = (Y - Y.mean(0, keepdims=True)) / (Y.std(0, keepdims=True) + 1e-6)
    del Y
    U, S, Vt = np.linalg.svd(Yz, full_matrices=False)
    PC = U[:, :N_PC] * S[:N_PC]
    PC = PC - PC.mean(0, keepdims=True); PC = PC / (PC.std(0, keepdims=True) + 1e-6)
    pos2i = {(r, c): i for i, (r, c) in enumerate(zip(rows, cols))}
    edges = []
    for i, (r, c) in enumerate(zip(rows, cols)):
        for dr, dc in ((0, 2), (0, -2), (1, -1), (1, 1), (-1, -1), (-1, 1)):
            j = pos2i.get((r + dr, c + dc))
            if j is not None: edges.append((i, j))
    edges = np.array(edges, dtype=np.int64)
    t_pipe = time.perf_counter() - t0

    # oracle 参照
    mu_or = np.array([PC[truth == kk].mean(0) for kk in range(k)])
    log_u = gaussian_log_unaries(PC, mu_or, 0.8)
    best_o, best_o_lab, best_o_beta = 0.0, None, None
    for b in BETAS:
        qo, _ = mean_field_graph(log_u, b, edges, max_iter=500, tol=1e-3)
        lb_o = np.argmax(qo, axis=1)
        a_o = adjusted_rand_index(truth, lb_o)
        if a_o > best_o:
            best_o, best_o_lab, best_o_beta = a_o, lb_o.copy(), b

    # VI-EM: beta 网格 x 5 起点
    runs = []
    betas = (0.7,) if quick else BETAS
    n_init = 1 if quick else N_INIT
    for b in betas:
        for t in range(n_init):
            rng = np.random.default_rng(100 + t)
            init = kmeans_pp(PC, k, rng)
            t1 = time.perf_counter()
            q, mu, sig, iters, elbo = em_mf_gaussian_graph(PC, k, b, edges, restarts=1,
                                                           max_iter=300, tol=1e-4, seed=t, init_labels=init)
            dt = time.perf_counter() - t1
            lab = np.argmax(q, axis=1)
            runs.append(dict(slice=sid, beta=b, init=t,
                             ari=round(adjusted_rand_index(truth, lab), 4),
                             nmi=round(nmi(truth, lab, k), 4),
                             bf1=round(boundary_f1_edges(lab, truth, edges), 4),
                             elbo=round(float(elbo), 1), iters=iters, t_s=round(dt, 1)))
    aris = [r["ari"] for r in runs]
    eb = int(np.argmax([r["elbo"] for r in runs]))
    mx = int(np.argmax(aris))
    best_lab = None  # 重取最优标签（保存用）
    rng = np.random.default_rng(100 + runs[mx]["init"])
    init = kmeans_pp(PC, k, rng)
    q, mu, sig, iters, elbo = em_mf_gaussian_graph(PC, k, runs[mx]["beta"], edges, restarts=1,
                                                   max_iter=300, tol=1e-4, seed=runs[mx]["init"], init_labels=init)
    best_lab = np.argmax(q, axis=1)
    np.savez(os.path.join(LAB, f"{sid}.npz"), truth=truth, rows=rows, cols=cols,
             oracle=best_o_lab, best=best_lab)
    summary = dict(slice=sid, n=int(n), n_edges=int(len(edges)),
                   oracle=round(float(best_o), 4), oracle_beta=best_o_beta,
                   ari_max=round(float(max(aris)), 4), ari_median=round(float(np.median(aris)), 4),
                   ari_elbo=round(runs[eb]["ari"], 4),
                   nmi_max=round(runs[mx]["nmi"], 4), bf1_max=round(runs[mx]["bf1"], 4),
                   t_pipe_s=round(t_pipe, 1), t_runs_s=round(sum(r["t_s"] for r in runs), 1))
    return summary, runs


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    if arg == "smoke":
        s, runs = run_slice("151673", quick=True)
        print("SMOKE:", s, flush=True)
        print(json.dumps(s, indent=1))
        return
    slices = ALL_SLICES if arg == "all" else [arg]
    f_long = os.path.join(OUT, "h9_ours_runs.csv")
    f_sum = os.path.join(OUT, "h9_ours_summary.csv")
    new_long = not os.path.exists(f_long) or arg == "all"
    with open(f_long, "w" if new_long else "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["slice", "beta", "init", "ari", "nmi", "bf1", "elbo", "iters", "t_s"])
        if new_long: w.writeheader()
        sums = []
        for sid in slices:
            t0 = time.perf_counter()
            s, runs = run_slice(sid)
            for r in runs: w.writerow(r)
            f.flush()
            sums.append(s)
            print(s, flush=True)
    # summary 全量重写（支持续跑合并）
    all_sums = {}
    if os.path.exists(f_sum):
        for r in csv.DictReader(open(f_sum)):
            all_sums[r["slice"]] = r
    for s in sums:
        all_sums[s["slice"]] = {k: str(v) for k, v in s.items()}
    with open(f_sum, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sums[0].keys()))
        w.writeheader()
        for sid in ALL_SLICES:
            if sid in all_sums:
                w.writerow(all_sums[sid])
    print("DONE h9 ours", flush=True)


if __name__ == "__main__":
    main()
