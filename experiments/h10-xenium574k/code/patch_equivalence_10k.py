# --- portable project root (no hardcoded user paths) ---
import os as _os
import pathlib as _pl


def _find_root(_start):
    _p = _pl.Path(_start).resolve()
    for _anc in [_p] + list(_p.parents):
        if (_anc / "src" / "hmrf_solvers.py").exists():
            return str(_anc)
    raise RuntimeError("project root not found; run inside the repository")


_ROOT = _os.environ.get("HMRF_ROOT") or _find_root(__file__)
# ------------------------------------------------------
"""GraphST sparse-patch equivalence check at 10K cells (review R9 / R2 W4).

The Xenium GraphST run required a 3-part sparse-storage patch because the stock
dense construction is physically infeasible at 574K cells (2.6 TB). This script
verifies the patch is semantics-preserving at a scale where BOTH legs are feasible:
  sparse leg = patched GraphST (current state of site-packages)
  dense leg  = original GraphST (restored from .bak-hmrfs by the driver)
Identical inputs/seed -> compare the constructed graph tensors and the emitted labels.

Usage (driven by e8_driver.sh):
  python patch_equivalence_10k.py sparse|dense|compare
"""
import json
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = _ROOT
XEN = _os.environ.get("HMRF_XENIUM_DIR", "")
if not XEN or not _os.path.isdir(XEN):
    raise SystemExit("set HMRF_XENIUM_DIR to the Xenium outs directory")
RES = _os.path.join(ROOT, "experiments/h10-xenium574k/results")
OUTS = _os.path.join(RES, "patch_equiv_10k")
_os.makedirs(OUTS, exist_ok=True)
N_SUB = 10000
K = 8
BASE = "/opt/anaconda3/lib/python3.12/site-packages/GraphST"
MARK = "SPARSE-PATCH HMRF-bench"


def patched_state():
    st = {}
    for f in ("GraphST.py", "model.py", "preprocess.py"):
        st[f] = MARK in open(f"{BASE}/{f}", encoding="utf-8").read()
    return st


def mclust_labels(X, n_clusters, seed=49):
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        xf = _os.path.join(td, "X.csv"); of = _os.path.join(td, "labels.csv")
        np.savetxt(xf, np.asarray(X, dtype=float), delimiter=",", fmt="%.6f")
        rcode = ("suppressPackageStartupMessages(library(mclust))\n"
                 f'X <- as.matrix(read.csv("{xf}", header=FALSE))\n'
                 f"set.seed({seed})\n"
                 f'res <- Mclust(X, G={n_clusters}, modelNames="EEE")\n'
                 f'write.table(res$classification, "{of}", row.names=FALSE, col.names=FALSE)')
        rf = _os.path.join(td, "fit.R")
        with open(rf, "w") as fh:
            fh.write(rcode)
        subprocess.run(["/usr/local/bin/Rscript", rf], check=True, capture_output=True, timeout=3600)
        return np.loadtxt(of, dtype=int)


def run_leg(leg):
    import scanpy as sc
    import torch
    from sklearn.decomposition import PCA
    from GraphST import GraphST
    t0 = time.time()
    ad = sc.read_10x_h5(_os.path.join(XEN, "cell_feature_matrix.h5"))
    cells = pd.read_csv(_os.path.join(XEN, "cells.csv.gz"))
    coords = cells[["x_centroid", "y_centroid"]].values.astype(np.float64)
    rng = np.random.default_rng(49)
    idx = np.sort(rng.choice(len(cells), N_SUB, replace=False))
    ad = ad[idx].copy(); coords = coords[idx]
    print(f"[{leg}] n={ad.shape[0]}", flush=True)
    ad.obs_names = [str(i) for i in ad.obs_names]
    ad.obsm["spatial"] = coords
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    sc.pp.scale(ad, max_value=10)
    ad.var["highly_variable"] = True
    model = GraphST.GraphST(ad, device="cpu", epochs=600, random_seed=49, datatype="Stereo")
    _src = getattr(model, "adata", ad)  # GraphST sets obsm on its internal copy
    gn = _src.obsm["graph_neigh"]
    if hasattr(gn, "tocsr"):
        G = gn.tocsr().astype(np.float32)
        gn_kind = "sparse"
    else:
        G = sp.csr_matrix(np.asarray(gn, dtype=np.float32))
        gn_kind = "dense"
    print(f"[{leg}] graph_neigh {gn_kind} nnz={G.nnz}", flush=True)
    ad = model.train()
    emb = PCA(n_components=20, random_state=42).fit_transform(ad.obsm["emb"])
    labels = mclust_labels(emb, K, seed=49)
    np.savez_compressed(_os.path.join(OUTS, f"leg_{leg}.npz"),
                        labels=labels.astype(np.int16),
                        indptr=G.indptr.astype(np.int64), indices=G.indices.astype(np.int32),
                        data=G.data.astype(np.float32))
    print(f"[{leg}] saved ({time.time()-t0:.0f}s)", flush=True)


def compare():
    a = np.load(_os.path.join(OUTS, "leg_sparse.npz"))
    b = np.load(_os.path.join(OUTS, "leg_dense.npz"))
    same_struct = bool(np.array_equal(a["indptr"], b["indptr"]) and np.array_equal(a["indices"], b["indices"]))
    same_data = bool(a["data"].shape == b["data"].shape and np.allclose(a["data"], b["data"]))
    sys.path.insert(0, _os.path.join(ROOT, "src"))
    from hmrf_solvers import adjusted_rand_index
    la, lb = a["labels"].astype(np.int64), b["labels"].astype(np.int64)
    ari = float(adjusted_rand_index(la, lb))
    agree = float((la == lb).mean())
    rep = {"same_graph_indices": same_struct, "same_graph_data": same_data,
           "labels_ari_sparse_vs_dense": round(ari, 6), "labels_agreement": round(agree, 6),
           "n": int(len(la))}
    json.dump(rep, open(_os.path.join(OUTS, "equivalence_report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode in ("sparse", "dense"):
        st = patched_state()
        want = {f: (mode == "sparse") for f in st}
        assert st == want, f"wrong patch state for leg {mode}: {st}"
        run_leg(mode)
    elif mode == "compare":
        compare()
    else:
        raise SystemExit("mode must be sparse|dense|compare")
