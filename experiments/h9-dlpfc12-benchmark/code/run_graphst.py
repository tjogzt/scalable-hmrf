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
"""H9: GraphST 基线（12 切片 DLPFC）。基于 GraphST 1.1.1 标准流程（对 151673 复现其教程设定）。
用法: python run_graphst.py [slice_id | all | smoke]
"""
import sys, os, time, csv, gzip
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import pandas as pd

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def mclust_labels(X, n_clusters, model_names="EEE", seed=2020):
    """Mclust(EEE) 聚类；经 Rscript 子进程调用（进程内 rpy2 嵌入会触发
    R/numpy 双 OpenBLAS 死锁——dger_thread 挂起，已实测踩坑，勿改回 rpy2）。"""
    import tempfile, subprocess
    with tempfile.TemporaryDirectory() as td:
        xf = os.path.join(td, "X.csv"); of = os.path.join(td, "labels.csv")
        np.savetxt(xf, np.asarray(X, dtype=float), delimiter=",", fmt="%.6f")
        rcode = ("suppressPackageStartupMessages(library(mclust))\n"
                 f'X <- as.matrix(read.csv("{xf}", header=FALSE))\n'
                 f"set.seed({seed})\n"
                 f'res <- Mclust(X, G={n_clusters}, modelNames="{model_names}")\n'
                 f'write.table(res$classification, "{of}", row.names=FALSE, col.names=FALSE)')
        rf = os.path.join(td, "fit.R")
        with open(rf, "w") as fh:
            fh.write(rcode)
        subprocess.run(["/usr/local/bin/Rscript", rf], check=True,
                       capture_output=True, timeout=900)
        return np.loadtxt(of, dtype=int)

ROOT = _ROOT
DATA = os.path.join(ROOT, "data/dlpfc12")
OUT = os.path.join(ROOT, "experiments/h9-dlpfc12-benchmark/results/graphst")
os.makedirs(OUT, exist_ok=True)
ALL_SLICES = ["151507", "151508", "151509", "151510", "151669", "151670",
              "151671", "151672", "151673", "151674", "151675", "151676"]


def nmi(a, b):
    # 真值标签可能是 'Layer3' 字符串：先因子化再编号（勿用 astype(int)）
    a = np.unique(np.asarray(a), return_inverse=True)[1]
    b = np.unique(np.asarray(b), return_inverse=True)[1]
    ka = a.max() + 1; kb = b.max() + 1
    ct = np.bincount(a * kb + b, minlength=ka * kb).reshape(ka, kb).astype(np.float64)
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


def run_slice(sid):
    import torch
    import scanpy as sc
    from GraphST import GraphST
    from sklearn.metrics import adjusted_rand_score
    t0 = time.time()
    with gzip.open(os.path.join(DATA, sid, "counts.mtx.gz"), "rb") as fh:
        m = sio.mmread(fh).tocsr()          # genes x spots
    spots = pd.read_csv(os.path.join(DATA, sid, "spots.csv"))
    counts = np.asarray(m.T.todense(), dtype=np.float32)   # spots x genes
    k = spots["label"].nunique()
    adata = sc.AnnData(counts)
    adata.obs_names = spots["barcode"].astype(str)
    adata.obsm["spatial"] = spots[["col", "row"]].values.astype(float)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=3000)
    adata = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata, max_value=10)
    model = GraphST.GraphST(adata, device="cpu", epochs=600, random_seed=49)
    adata = model.train()
    # GraphST 教程聚类设定：emb -> PCA(20) -> Mclust(EEE) -> refine_label(50)
    # （GraphST.utils.clustering 内置 rpy2 弃用 API 在 rpy2>=3.6 崩溃，此处以现代 API 等价实现）
    from sklearn.decomposition import PCA
    from GraphST.utils import refine_label
    emb_pca = PCA(n_components=20, random_state=42).fit_transform(adata.obsm["emb"])
    labels = mclust_labels(emb_pca, k, model_names="EEE", seed=49)
    adata.obs["domain"] = labels
    refined = refine_label(adata, radius=50, key="domain")
    pred = np.asarray(refined).astype(int)
    dt = time.time() - t0
    ari = adjusted_rand_score(spots["label"], pred)
    nm = nmi(spots["label"], pred)
    pd.DataFrame({"barcode": spots["barcode"], "pred": pred}).to_csv(
        os.path.join(OUT, f"{sid}_labels.csv"), index=False)
    rec = dict(slice=sid, method="GraphST", k=int(k), ari=round(float(ari), 4),
               nmi=round(float(nm), 4), t_s=round(dt, 1))
    with open(os.path.join(OUT, f"{sid}_result.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rec.keys())); w.writeheader(); w.writerow(rec)
    print(rec, flush=True)
    return rec


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    slices = ALL_SLICES if arg == "all" else [arg]
    rows = []
    for sid in slices:
        try:
            rows.append(run_slice(sid))
        except Exception as e:
            print(f"{sid} FAILED: {type(e).__name__} {e}", flush=True)
    if rows:
        f = os.path.join(OUT, "graphst_all.csv")
        prev = pd.read_csv(f, dtype={"slice": str}) if os.path.exists(f) else None
        new = pd.DataFrame(rows)
        if prev is not None:
            new = pd.concat([prev[~prev.slice.isin(new.slice)], new])
        new["slice"] = new["slice"].astype(str)
        new.sort_values("slice").to_csv(f, index=False)
    print("GRAPHST DONE", flush=True)


if __name__ == "__main__":
    main()
