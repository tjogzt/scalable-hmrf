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
"""Xenium marker plausibility check (review R9, R2 W3 / R3 W6):

Per-domain mean expression of canonical breast-IDC markers across the K=8 HMRF domains
of the 574,852-cell Xenium section. Produces a marker table (CSV) and a heatmap figure
(figS1), addressing "does the map correspond to tumour/immune/stroma architecture".

Requires HMRF_XENIUM_DIR pointing at the 10x Xenium breast-IDC outs directory.
Usage: python marker_check.py
"""
import sys

import numpy as np
import pandas as pd

XEN = _os.environ.get("HMRF_XENIUM_DIR",
                      "/Volumes/tjogzt4T/data/Xenium_V1_FFPE_Human_Breast_IDC_outs")
RES = _os.path.join(_ROOT, "experiments/h10-xenium574k/results")
NPZ = _os.path.join(RES, "h10_labels_K8_b0.7_r0.npz")

CANDIDATES = [
    # tumour epithelium / epithelial markers
    "EPCAM", "KRT8", "KRT19", "KRT7", "KRT17", "KRT14", "KRT5", "ELF3", "CDH1", "MUC1", "TACSTD2",
    # proliferation
    "MKI67", "TOP2A", "CCNB1", "CCND1", "BIRC5",
    # immune general / T cells
    "PTPRC", "CD2", "CD3E", "IL7R", "TRBC1", "TRBC2", "CD8A",
    # myeloid
    "CD68", "CD14", "AIF1", "ITGAX", "LYZ", "FCGR3A", "CD163", "C1QA", "APOC1", "CD74",
    # B / plasma
    "MS4A1", "CD79A", "IGHG1",
    # stroma / ECM
    "COL1A2", "COL3A1", "COL5A1", "DCN", "LUM", "FN1", "POSTN", "SPARC", "VIM", "COL6A1",
    # myoepithelial / vascular / smooth muscle
    "ACTA2", "TAGLN", "MYH11", "ACTG2", "MYL9", "VWF", "PECAM1", "ENG",
    # receptors / clinical
    "ESR1", "PGR", "ERBB2", "GRB7",
    # adipose
    "ADIPOQ", "FABP4", "PLIN1",
    # other TME
    "CXCL12", "CXCL14", "SPP1", "MMP11", "MMP2", "TIMP1", "TGFB1", "IGFBP5", "CCL19", "CCL21",
]

d = np.load(NPZ)
pred = d["pred"].astype(np.int64)
lab_k8 = d["lab_k8"].astype(np.int64)
K = int(d["K"])
print(f"npz: pred {pred.shape}, K={K}, beta={float(d['beta'])}")

import scanpy as sc  # noqa: E402

ad = sc.read_10x_h5(_os.path.join(XEN, "cell_feature_matrix.h5"))
print("h5:", ad.shape)
assert ad.n_obs == pred.shape[0], f"cell count mismatch: {ad.n_obs} vs {pred.shape[0]}"
genes = list(ad.var_names)
markers = [g for g in CANDIDATES if g in genes]
missing = [g for g in CANDIDATES if g not in genes]
print(f"markers available: {len(markers)}; missing: {missing}")
with open(_os.path.join(RES, "xenium_panel_genes.txt"), "w") as f:
    f.write("\n".join(genes))

# sanity: pred vs the official k-means-8 reference
sys.path.insert(0, _os.path.join(_ROOT, "src"))
from hmrf_solvers import adjusted_rand_index  # noqa: E402

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
mean_df.round(4).to_csv(_os.path.join(RES, "xenium_marker_means_K8.csv"))

# z-score across domains per marker
z = (dom - dom.mean(0)) / (dom.std(0) + 1e-9)
z_df = pd.DataFrame(z, index=[f"D{k}" for k in range(K)], columns=markers)
z_df.round(2).to_csv(_os.path.join(RES, "xenium_marker_zscores_K8.csv"))

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
FIGD = _os.path.join(_ROOT, "paper", "figures")
fig.savefig(_os.path.join(FIGD, "figS1_xenium_markers.pdf"))
fig.savefig(_os.path.join(FIGD, "figS1_xenium_markers.png"))
fig.savefig(_os.path.join(RES, "figS1_xenium_markers.png"))
print("\nfigure saved: paper/figures/figS1_xenium_markers.{pdf,png}")
print("DONE")
