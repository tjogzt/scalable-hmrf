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
"""H4d: DLPFC 151673 指标补全（NMI + 边界 F1）与标签留档（供图 4 出图）。

复跑 run_final.py 的最终配置（VST-HVG3000 + zscore + PCA50 + kmeans++ 5 起点，beta=0.7），
补算 NMI/边界F1，并把 truth/各起点/标签存 npz（供论文图件）。
用法: python run.py [smoke]
"""
import sys, os, time, csv, json
sys.path.insert(0, _os.path.join(_ROOT, "src"))
import numpy as np
import pandas as pd
import h5py
from hmrf_solvers import em_mf_gaussian_graph, mean_field_graph, gaussian_log_unaries, adjusted_rand_index

SMOKE = len(sys.argv) > 1 and sys.argv[1] == "smoke"
DATA = _os.path.join(_ROOT, "data/dlpfc")
OUT = _os.path.join(_ROOT, "experiments/h4d-dlpfc-metrics/results")
os.makedirs(OUT, exist_ok=True)
K, N_PC = 7, 50
BETA = 0.7
N_INIT = 1 if SMOKE else 5
BETAS_ORACLE = (0.3, 0.5, 0.7, 1.0)


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


def main():
    t_start = time.perf_counter()
    f = h5py.File(f"{DATA}/151673_raw_feature_bc_matrix.h5", "r")
    M = f["matrix"]
    n_genes, n_spots = int(M["shape"][0]), int(M["shape"][1])
    data = np.array(M["data"], dtype=np.float32); indices = np.array(M["indices"], dtype=np.int64); indptr = np.array(M["indptr"], dtype=np.int64)
    barcodes = [b.decode() for b in f["matrix"]["barcodes"]]; f.close()
    X = np.zeros((n_spots, n_genes), dtype=np.float32)
    for s in range(n_spots):
        a, b = indptr[s], indptr[s + 1]; X[s, indices[a:b]] = data[a:b]
    pos = pd.read_csv(f"{DATA}/tissue_positions_list.txt", header=None, index_col=0); pos.index = pos.index.astype(str)
    pos.columns = ["in_tissue", "row", "col", "px_row", "px_col"]
    lab = pd.read_csv(f"{DATA}/cluster_labels.csv"); lab.columns = [c.strip() for c in lab.columns]
    lab["barcode"] = lab[lab.columns[0]].astype(str).str.replace("^151673_", "", regex=True)
    lab = lab.set_index("barcode")["ground_truth"]; lab = lab[~lab.isna()]
    in_t = pos[pos["in_tissue"] == 1]
    common = in_t.index.intersection(lab.index)
    bc_idx = {b: i for i, b in enumerate(barcodes)}
    X = X[[bc_idx[b] for b in common]]
    rows = in_t.loc[common, "row"].values.astype(np.int64); cols = in_t.loc[common, "col"].values.astype(np.int64)
    truth_str = lab.loc[common].values
    classes = sorted(set(truth_str)); truth = np.array([classes.index(t) for t in truth_str])
    n = X.shape[0]
    print("spots:", n, "classes:", len(classes), flush=True)
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
    print("edges:", len(edges), "pipeline t=%.0fs" % (time.perf_counter() - t_start), flush=True)

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

    # ---------- oracle 参照（含最优标签留档） ----------
    mu_or = np.array([PC[truth == k].mean(0) for k in range(K)])
    log_u = gaussian_log_unaries(PC, mu_or, 0.8)
    best_o, best_o_lab, best_o_beta = 0.0, None, None
    for b in BETAS_ORACLE:
        qo, _ = mean_field_graph(log_u, b, edges, max_iter=500, tol=1e-3)
        lb_o = np.argmax(qo, axis=1)
        a_o = adjusted_rand_index(truth, lb_o)
        if a_o > best_o:
            best_o, best_o_lab, best_o_beta = a_o, lb_o.copy(), b
    print(f"ORACLE best ARI={best_o:.4f} @beta={best_o_beta}", flush=True)

    # ---------- 5 起点 VI-EM @ beta=0.7 ----------
    rows_out, labs = [], []
    for t in range(N_INIT):
        rng = np.random.default_rng(100 + t)
        init = kmeans_pp(PC, K, rng)
        t0 = time.perf_counter()
        q, mu, sig, iters, elbo = em_mf_gaussian_graph(PC, K, BETA, edges, restarts=1, max_iter=300, tol=1e-4, seed=t, init_labels=init)
        lab_pred = np.argmax(q, axis=1)
        rec = dict(init=t, ari=round(adjusted_rand_index(truth, lab_pred), 4),
                   nmi=round(nmi(truth, lab_pred, K), 4),
                   bf1=round(boundary_f1_edges(lab_pred, truth, edges), 4),
                   elbo=round(float(elbo), 1), iters=iters, t_s=round(time.perf_counter() - t0, 1),
                   sigma_hat=round(float(sig), 3))
        rows_out.append(rec); labs.append(lab_pred)
        print("INIT", rec, flush=True)
    aris = [r["ari"] for r in rows_out]
    eb = int(np.argmax([r["elbo"] for r in rows_out]))
    meta = dict(oracle_best_ari=round(float(best_o), 4), oracle_best_beta=best_o_beta,
                oracle_nmi=round(nmi(truth, best_o_lab, K), 4),
                oracle_bf1=round(boundary_f1_edges(best_o_lab, truth, edges), 4),
                ari_max=round(float(max(aris)), 4), ari_median=round(float(np.median(aris)), 4),
                ari_elbo_selected=round(rows_out[eb]["ari"], 4), elbo_selected_init=eb,
                n=len(truth), n_edges=int(len(edges)))
    with open(os.path.join(OUT, "h4d_metrics.csv"), "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows_out[0].keys())); w.writeheader(); w.writerows(rows_out)
    with open(os.path.join(OUT, "h4d_meta.json"), "w") as fo:
        json.dump(meta, fo, indent=1)
    np.savez(os.path.join(OUT, "h4d_labels.npz"), truth=truth, rows=rows, cols=cols,
             oracle=best_o_lab, classes=np.array(classes), edges=edges,
             **{f"init{t}": labs[t] for t in range(N_INIT)})
    print("META", meta, flush=True)
    print("DONE h4d", flush=True)


if __name__ == "__main__":
    main()
