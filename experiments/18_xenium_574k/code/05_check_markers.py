"""Marker-expression sanity check of the Xenium domains.

Author : Tao Zhu <zhutao@tjh.tjmu.edu.cn>
Created: 2026-10-02
Updated: 2026-10-20

Inputs
------
Xenium inputs, label NPZ and marker panel from configuration.
Configuration: config/defaults.yaml, loaded through hmrf.config.

Outputs
-------
results/marker_check.csv.

Usage
-----
    python 05_check_markers.py [options]
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

import sys

import numpy as np
import pandas as pd

XEN = os.environ.get("HMRF_XENIUM_DIR")
if not XEN:
    raise SystemExit("set HMRF_XENIUM_DIR to the Xenium outs directory")
RES = str(results_dir("18_xenium_574k"))
NPZ = os.path.join(RES, "h10_labels_K8_b0.7_r0.npz")

CANDIDATES = CFG["datasets"]["xenium"]["marker_panel"]

d = np.load(NPZ)
pred = d["pred"].astype(np.int64)
lab_k8 = d["lab_k8"].astype(np.int64)
K = int(d["K"])
print(f"npz: pred {pred.shape}, K={K}, beta={float(d['beta'])}")

import scanpy as sc  # noqa: E402

ad = sc.read_10x_h5(os.path.join(XEN, "cell_feature_matrix.h5"))
print("h5:", ad.shape)
assert ad.n_obs == pred.shape[0], f"cell count mismatch: {ad.n_obs} vs {pred.shape[0]}"
genes = list(ad.var_names)
markers = [g for g in CANDIDATES if g in genes]
missing = [g for g in CANDIDATES if g not in genes]
print(f"markers available: {len(markers)}; missing: {missing}")
with open(os.path.join(RES, "xenium_panel_genes.txt"), "w") as f:
    f.write("\n".join(genes))

# sanity: pred vs the official k-means-8 reference
sys.path.insert(0, os.path.join(_ROOT, "src"))
from hmrf.solvers import adjusted_rand_index  # noqa: E402

print(f"sanity ARI(pred, kmeans_8 ref) = {adjusted_rand_index(lab_k8, pred):.4f}")

# CP10K -> log1p
X = ad[:, markers].X.tocsr().astype(np.float64)
lib = np.asarray(X.sum(1)).ravel()
scale = np.divide(1e4, np.maximum(lib, 1.0))
X = X.multiply(scale[:, None]).tocsr()
X.data = np.log1p(X.data)

# per-domain means
dom = np.zeros((K, len(markers)))
sizes = np.zeros(K, dtype=int)
for kk in range(K):
    m = pred == kk
    sizes[kk] = int(m.sum())
    dom[kk] = np.asarray(X[m].mean(0)).ravel()

mean_df = pd.DataFrame(dom, index=[f"D{k}" for k in range(K)], columns=markers)
mean_df.insert(0, "n_cells", sizes)
mean_df.round(4).to_csv(os.path.join(RES, "xenium_marker_means_K8.csv"))

# z-score across domains per marker
z = (dom - dom.mean(0)) / (dom.std(0) + 1e-9)
z_df = pd.DataFrame(z, index=[f"D{k}" for k in range(K)], columns=markers)
z_df.round(2).to_csv(os.path.join(RES, "xenium_marker_zscores_K8.csv"))

pd.set_option("display.width", 240, "display.max_columns", 100)
print("\n== per-domain mean expression (CP10K log1p) ==")
print(mean_df.round(3).to_string())
print("\n== z-scores across domains (per marker) ==")
print(z_df.round(2).to_string())

# top markers per domain (by z)
print("\n== top-4 z markers per domain ==")
for kk in range(K):
    top = z_df.iloc[kk].sort_values(ascending=False).head(4)
    print(f"D{kk} (n={sizes[kk]}): " + ", ".join(f"{g}({v:+.1f})" for g, v in top.items()))

# ---- heatmap figure ----
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.linewidth": 0.6})
# order markers by peak domain for readability
order = np.argsort(-z.max(0)) if False else np.argsort([int(np.argmax(z[:, j])) * 100 - z[:, j].max() for j in range(z.shape[1])])
markers_o = [markers[j] for j in order]
zo = z[:, order]
fig, ax = plt.subplots(figsize=(7.4, 4.6), dpi=200)
im = ax.imshow(zo, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
ax.set_xticks(range(len(markers_o)), markers_o, rotation=90, fontsize=7)
ax.set_yticks(range(K), [f"D{k}" for k in range(K)])
for kk in range(K):
    for jj in range(len(markers_o)):
        v = zo[kk, jj]
        if abs(v) >= 1.5:
            ax.text(jj, kk, f"{v:+.1f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(v) > 2.2 else "black")
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
cb.set_label("z-score across domains", fontsize=8)
ax.set_title("Xenium breast IDC (574,852 cells): marker profile of K=8 HMRF domains", fontsize=9)
fig.tight_layout()
FIGD = os.path.join(_ROOT, "paper", "figures")
fig.savefig(os.path.join(FIGD, "figS1_xenium_markers.pdf"))
fig.savefig(os.path.join(FIGD, "figS1_xenium_markers.png"))
fig.savefig(os.path.join(RES, "figS1_xenium_markers.png"))
print("\nfigure saved: paper/figures/figS1_xenium_markers.{pdf,png}")
print("DONE")
