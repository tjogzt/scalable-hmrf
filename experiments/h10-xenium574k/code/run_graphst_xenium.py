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
"""H10: GraphST baseline @ Xenium Breast IDC (smoke=50K subsample / full=574K full data)
- Official pipeline: log1p(CP10K) -> scale -> GraphST(datatype='10x', n=6 kNN graph, 600 epochs) ->
  PCA20(emb) -> Mclust(EEE) -> 50-NN majority-vote refinement
- Note: the native GraphST.utils.refine_label builds a 574K x 574K all-pairs distance matrix (infeasible);
  here it is implemented equivalently with a cKDTree 50-NN majority vote (semantics: mode of the labels of the 50 nearest neighbors).
Usage: python run_graphst_xenium.py smoke|full
"""
import sys, os, time, csv, json
import numpy as np
import scipy.sparse as sp
import pandas as pd

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = _ROOT
XEN = _os.environ.get("HMRF_XENIUM_DIR", "")
if not XEN or not _os.path.isdir(XEN):
    raise SystemExit("env var HMRF_XENIUM_DIR is unset or invalid: point it at the Xenium outs directory (10x Genomics public data, see README)")
OUT = os.path.join(ROOT, "experiments/h10-xenium574k/results")
os.makedirs(OUT, exist_ok=True)
MODE = sys.argv[1] if len(sys.argv) > 1 else "smoke"
SMOKE_N = 50000
K = 8


def mclust_labels(X, n_clusters, model_names="EEE", seed=2020):
    """Calls Mclust via an Rscript subprocess (in-process rpy2 triggers a double-OpenBLAS deadlock)."""
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
                       capture_output=True, timeout=7200)
        return np.loadtxt(of, dtype=int)


def refine_knn(labels, coords, k=50):
    """50-NN majority vote (equivalent to GraphST refine_label semantics, avoids the all-pairs distance matrix)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(coords)
    _, idx = tree.query(coords, k=k + 1)   # includes self
    neigh = labels[idx[:, 1:]]             # (n, 50)
    out = np.empty_like(labels)
    for i in range(len(labels)):
        vals, counts = np.unique(neigh[i], return_counts=True)
        out[i] = vals[counts.argmax()]
    return out


t0 = time.time()
import scanpy as sc
import torch
from sklearn.metrics import adjusted_rand_score

print(f"[{MODE}] loading h5 ...", flush=True)
ad = sc.read_10x_h5(os.path.join(XEN, "cell_feature_matrix.h5"))
cells = pd.read_csv(os.path.join(XEN, "cells.csv.gz"))
coords = cells[["x_centroid", "y_centroid"]].values.astype(np.float64)
ref_k8 = pd.read_csv(os.path.join(XEN, "analysis/clustering/gene_expression_kmeans_8_clusters/clusters.csv"))
lab_k8 = np.full(len(cells), -1, dtype=np.int64); lab_k8[ref_k8["Barcode"].values - 1] = ref_k8["Cluster"].values
ref_gc = pd.read_csv(os.path.join(XEN, "analysis/clustering/gene_expression_graphclust/clusters.csv"))
lab_gc = np.full(len(cells), -1, dtype=np.int64); lab_gc[ref_gc["Barcode"].values - 1] = ref_gc["Cluster"].values

if MODE == "smoke":
    rng = np.random.default_rng(49)
    idx = np.sort(rng.choice(len(cells), SMOKE_N, replace=False))
    ad = ad[idx].copy(); coords = coords[idx]; lab_k8 = lab_k8[idx]; lab_gc = lab_gc[idx]
    print(f"[smoke] subsample n={ad.shape[0]}", flush=True)

ad.obs_names = [str(i) for i in ad.obs_names]
ad.obsm["spatial"] = coords
sc.pp.normalize_total(ad, target_sum=1e4)
sc.pp.log1p(ad)
sc.pp.scale(ad, max_value=10)
ad.var["highly_variable"] = True   # skip GraphST's internal seurat_v3 HVG (280 genes < 3000 triggers a loess extrapolation error)
print(f"preprocessed {time.time()-t0:.1f}s", flush=True)

from GraphST import GraphST
from sklearn.decomposition import PCA
model = GraphST.GraphST(ad, device="cpu", epochs=600, random_seed=49, datatype="Stereo")
t_train = time.time()
ad = model.train()
print(f"trained 600 epochs in {time.time()-t_train:.1f}s (total {time.time()-t0:.1f}s)", flush=True)

emb_pca = PCA(n_components=20, random_state=42).fit_transform(ad.obsm["emb"])
print("mclust ...", flush=True)
labels = mclust_labels(emb_pca, K, model_names="EEE", seed=49)
ari_raw = adjusted_rand_score(lab_k8[lab_k8 >= 0], labels[lab_k8 >= 0])
print(f"mclust done | ARI(raw, kmeans8) = {ari_raw:.4f}", flush=True)

t_ref = time.time()
labels_ref = refine_knn(labels, coords, k=50)
ari_ref = adjusted_rand_score(lab_k8[lab_k8 >= 0], labels_ref[lab_k8 >= 0])
mask_gc = lab_gc >= 0
ari_gc_raw = adjusted_rand_score(lab_gc[mask_gc], labels[mask_gc])
ari_gc_ref = adjusted_rand_score(lab_gc[mask_gc], labels_ref[mask_gc])
print(f"refine done {time.time()-t_ref:.1f}s | ARI(refined, kmeans8) = {ari_ref:.4f} | "
      f"graphclust: raw {ari_gc_raw:.4f} / refined {ari_gc_ref:.4f}", flush=True)

rec = dict(mode=MODE, n=int(ad.shape[0]), K=K, ari_raw_kmeans8=round(float(ari_raw), 4),
           ari_ref_kmeans8=round(float(ari_ref), 4),
           ari_raw_graphclust=round(float(ari_gc_raw), 4),
           ari_ref_graphclust=round(float(ari_gc_ref), 4),
           t_total=round(time.time() - t0, 1))
json.dump(rec, open(os.path.join(OUT, f"graphst_xenium_{MODE}.json"), "w"), indent=1)
if MODE == "full":
    pd.DataFrame({"cell_id": cells["cell_id"], "pred": labels_ref}).to_csv(
        os.path.join(OUT, "graphst_xenium_labels.csv"), index=False)
    pd.DataFrame({"cell_id": cells["cell_id"], "pred_raw": labels}).to_csv(
        os.path.join(OUT, "graphst_xenium_labels_raw.csv"), index=False)
print("RESULT:", json.dumps(rec), flush=True)
print("DONE", flush=True)
