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
"""fig8 v3: Xenium Breast IDC 574K 细胞三面板域图（参照 | 本方法 | GraphST，匈牙利匹配对齐颜色）。
数据: h10_labels_K8_b0.7_r2.npz（本方法） + graphst_xenium_labels_raw.csv（GraphST raw）
输出: paper/figures/fig8_xenium_domains.{pdf,png}
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

plt.rcParams.update({
    "font.family": "Arial", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42,
})
PROJ = _ROOT
RES = os.path.join(PROJ, "experiments/h10-xenium574k/results")
FIG = next(p for p in (os.path.join(PROJ, "paper", "figures"), os.path.join(PROJ, "figures"))
           if os.path.isdir(p))  # 兼容工作区与发布两种布局

d = np.load(os.path.join(RES, "h10_labels_K8_b0.7_r2.npz"))
ours = d["pred"].astype(int); ref = d["lab_k8"].astype(int); xy = d["coords"]
gs = pd.read_csv(os.path.join(RES, "graphst_xenium_labels_raw.csv"))
gs_lab = np.full(len(ref), -1, dtype=int)
gs_lab[gs["cell_id"].values - 1] = gs["pred_raw"].values
assert len(gs_lab) == len(ref)

cmap = plt.get_cmap("tab10")

def match(pred, ref):
    mask = (ref >= 0) & (pred >= 0)
    pu = np.unique(pred[mask]); ru = np.unique(ref[mask])
    C = np.zeros((len(pu), len(ru)))
    for i, p in enumerate(pu):
        pm = pred == p
        for j, rr in enumerate(ru):
            C[i, j] = np.sum(pm & (ref == rr))
    ri, ci = linear_sum_assignment(-C)
    return {int(pu[i]): int(ru[j]) for i, j in zip(ri, ci)}, mask

mp_ours, m1 = match(ours, ref)
mp_gs, m2 = match(gs_lab, ref)
ari_ours = adjusted_rand_score(ref[m1], ours[m1])
ari_gs = adjusted_rand_score(ref[m2], gs_lab[m2])

fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.3))
panels = [
    ("10x k-means (k=8) reference", ref, None),
    (f"Ours: HMRF-MF ($\\beta$=0.7) — ARI {ari_ours:.3f}", ours, mp_ours),
    (f"GraphST (600 epochs) — ARI {ari_gs:.3f}", gs_lab, mp_gs),
]
for ax, (ttl, lab, mp) in zip(axes, panels):
    for v in np.unique(lab[lab >= 0]):
        m = lab == v
        cl = v if mp is None else mp.get(int(v), int(v))
        ax.scatter(xy[m, 0], xy[m, 1], s=0.015, color=cmap(cl % 10), linewidths=0, rasterized=True)
    ax.set_title(ttl, fontsize=8.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")
    ax.invert_yaxis()
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.6); sp.set_color("0.6")

fig.text(0.5, 0.012,
         "Xenium human breast IDC — 574,852 cells; spatial kNN graphs; colours aligned by Hungarian matching; "
         "GraphST via sparse-storage patch (official dense implementation infeasible at this scale)",
         ha="center", fontsize=7, color="0.35")
fig.tight_layout(rect=[0, 0.035, 1, 1])
fig.savefig(os.path.join(FIG, "fig8_xenium_domains.pdf"))
fig.savefig(os.path.join(FIG, "fig8_xenium_domains.png"))
print(f"saved | ARI ours={ari_ours:.4f} graphst={ari_gs:.4f}")
