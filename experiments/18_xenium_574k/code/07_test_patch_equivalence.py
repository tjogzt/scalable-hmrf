"""Verify that the sparse patch is numerically equivalent to dense GraphST.

Author : Tao Zhu <zhutao@tjh.tjmu.edu.cn>
Created: 2026-10-02
Updated: 2026-10-20

Inputs
------
Xenium subset and patched GraphST.
Configuration: config/defaults.yaml, loaded through hmrf.config.

Outputs
-------
results/patch_equiv_10k/equivalence_report.json.

Usage
-----
    python 07_test_patch_equivalence.py [options]
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

import json
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = _ROOT
XEN = os.environ.get("HMRF_XENIUM_DIR", "")
if not XEN or not os.path.isdir(XEN):
    raise SystemExit("set HMRF_XENIUM_DIR to the Xenium outs directory")
RES = str(results_dir("18_xenium_574k"))
OUTS = os.path.join(RES, "patch_equiv_10k")
os.makedirs(OUTS, exist_ok=True)
N_SUB = CFG["experiments"]["h10"]["patch_subset"]
K = CFG["datasets"]["xenium"]["n_clusters"]
BASE = CFG["experiments"]["h10"]["graphst_dir"]
if not BASE:
    raise SystemExit("set HMRF_GRAPHST_DIR to the installed GraphST package")
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
        xf = os.path.join(td, "X.csv"); of = os.path.join(td, "labels.csv")
        np.savetxt(xf, np.asarray(X, dtype=float), delimiter=",", fmt="%.6f")
        rcode = ("suppressPackageStartupMessages(library(mclust))\n"
                 f'X <- as.matrix(read.csv("{xf}", header=FALSE))\n'
                 f"set.seed({seed})\n"
                 f'res <- Mclust(X, G={n_clusters}, modelNames="EEE")\n'
                 f'write.table(res$classification, "{of}", row.names=FALSE, col.names=FALSE)')
        rf = os.path.join(td, "fit.R")
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
    ad = sc.read_10x_h5(os.path.join(XEN, "cell_feature_matrix.h5"))
    cells = pd.read_csv(os.path.join(XEN, "cells.csv.gz"))
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
    np.savez_compressed(os.path.join(OUTS, f"leg_{leg}.npz"),
                        labels=labels.astype(np.int16),
                        indptr=G.indptr.astype(np.int64), indices=G.indices.astype(np.int32),
                        data=G.data.astype(np.float32))
    print(f"[{leg}] saved ({time.time()-t0:.0f}s)", flush=True)


def compare():
    a = np.load(os.path.join(OUTS, "leg_sparse.npz"))
    b = np.load(os.path.join(OUTS, "leg_dense.npz"))
    same_struct = bool(np.array_equal(a["indptr"], b["indptr"]) and np.array_equal(a["indices"], b["indices"]))
    same_data = bool(a["data"].shape == b["data"].shape and np.allclose(a["data"], b["data"]))
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from hmrf.solvers import adjusted_rand_index
    la, lb = a["labels"].astype(np.int64), b["labels"].astype(np.int64)
    ari = float(adjusted_rand_index(la, lb))
    agree = float((la == lb).mean())
    rep = {"same_graph_indices": same_struct, "same_graph_data": same_data,
           "labels_ari_sparse_vs_dense": round(ari, 6), "labels_agreement": round(agree, 6),
           "n": int(len(la))}
    json.dump(rep, open(os.path.join(OUTS, "equivalence_report.json"), "w"), indent=1)
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
