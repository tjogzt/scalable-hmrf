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
"""fig1: method overview schematic --- Potts HMRF on the spot graph, mean-field fixed-point
update with restarts, the exact graph-cut counterpart (negative result), and the
pipeline/oracle-attribution flow.
Output: paper/figures/fig1_method_overview.{pdf,png}
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "Arial", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "figure.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42,
    "mathtext.fontset": "dejavusans",
})
ZHUSHA = "#C23531"; SHIQING = "#3D6BA8"; YANZHI = "#9D2933"; DIANQING = "#177CB0"; GREY = "#8A9299"
FIG = next(p for p in (os.path.join(_ROOT, "paper", "figures"), os.path.join(_ROOT, "figures"))
           if os.path.isdir(p))

fig = plt.figure(figsize=(7.2, 4.35))

# ============ (a) Potts HMRF on the spot graph ============
axa = fig.add_axes([0.035, 0.565, 0.30, 0.40]); axa.set_axis_off()
axa.set_title("(a) Potts HMRF on the spot graph", fontsize=9, loc="left", pad=2)

rows, cols = 5, 6
pts, lab = [], []
for r in range(rows):
    for c in range(cols):
        x = c + 0.5 * (r % 2); y = r * 0.88
        # two-column domain split with a wavy boundary
        k = 0 if c <= 2 + (1 if r in (1, 3) else 0) else 1
        pts.append((x, y)); lab.append(k)
pts = np.array(pts); lab = np.array(lab)
pos = {i: tuple(pts[i]) for i in range(len(pts))}
edges = []
for i in range(len(pts)):
    for j in range(i + 1, len(pts)):
        d = np.hypot(*(pts[i] - pts[j]))
        if d < 1.05:
            edges.append((i, j))
for i, j in edges:
    same = lab[i] == lab[j]
    axa.plot(*zip(pos[i], pos[j]),
             color=(GREY if same else YANZHI), lw=(0.7 if same else 1.3),
             alpha=(0.55 if same else 0.95), zorder=1,
             solid_capstyle="round")
axa.scatter(pts[:, 0], pts[:, 1], s=34, c=[ZHUSHA if k == 0 else SHIQING for k in lab],
            edgecolors="white", linewidths=0.5, zorder=2)
axa.set_xlim(-0.6, 6.4); axa.set_ylim(-1.15, 4.35)
axa.text(2.85, -0.95,
         r"$P(y,z)\ \propto\ \exp\left(\sum_i \log p(y_i\,|\,z_i,\theta)"
         r"\ +\ \beta\sum_{i\sim j}\mathbf{1}[z_i{=}z_j]\right)$",
         ha="center", va="top", fontsize=8.6)

# ============ (b) mean-field fixed point ============
axb = fig.add_axes([0.375, 0.565, 0.27, 0.40]); axb.set_axis_off()
axb.set_title("(b) Mean-field VI: fixed-point updates", fontsize=9, loc="left", pad=2)
axb.text(0.5, 0.85,
         r"$q_i(k)\ \propto\ \exp\left(\log p(y_i\,|\,k,\theta)"
         r"\ +\ \beta\sum_{j\in N(i)} q_j(k)\right)$",
         ha="center", va="center", fontsize=8.6, transform=axb.transAxes)
# convergence sparkline
axc = fig.add_axes([0.418, 0.658, 0.185, 0.102])
it = np.arange(60)
dq = 0.9 * np.exp(-it / 9.0) + 0.02 + 0.015 * np.sin(it / 2.5)
axc.semilogy(it, dq, color=SHIQING, lw=1.3)
axc.set_ylabel(r"max$_i |\Delta q_i|$", fontsize=8)
axc.set_xticks([])
axc.tick_params(axis="y", labelsize=7.5)
axc.spines[["top", "right"]].set_visible(False)
axb.text(0.5, 0.075, "25 vectorized restarts\n(best / median protocols)",
         ha="center", va="center", fontsize=8.4, transform=axb.transAxes, color="#333333")

# ============ (c) exact graph cuts: negative result ============
axd = fig.add_axes([0.695, 0.565, 0.285, 0.40]); axd.set_axis_off()
axd.set_title("(c) Exact graph cuts (\u03b1-expansion)", fontsize=9, loc="left", pad=2)
e = np.linspace(0.1, 1.0, 50)
ari_mf = 1.0 - np.exp(-3.2 * (1 - e)) + 0.02
ari_cut = 1.0 - np.exp(-4.6 * (1 - e))
axe = axd.inset_axes([0.10, 0.42, 0.86, 0.51])
axe.plot(e, ari_mf, color=SHIQING, lw=1.5, label="mean-field")
axe.plot(e, ari_cut, color=ZHUSHA, lw=1.5, label="exact cut")
axe.set_xlabel("normalized energy", fontsize=8); axe.set_ylabel("ARI (schematic)", fontsize=8)
axe.tick_params(labelsize=7.5)
axe.spines[["top", "right"]].set_visible(False)
axe.legend(fontsize=7.2, frameon=False, loc="lower right")
axd.text(0.5, 0.10, "lower energy \u2260 higher label accuracy:\nARI gap \u2264 0.006 with sign flips",
         ha="center", va="center", fontsize=8.4, transform=axd.transAxes, color="#333333")

# ============ (d) pipeline + oracle attribution ============
axf = fig.add_axes([0.035, 0.045, 0.945, 0.44]); axf.set_axis_off()
axf.set_title("(d) End-to-end pipeline and oracle attribution", fontsize=9, loc="left", pad=2)
boxes = [
    (0.015, "raw counts\n(Visium / Xenium)", ZHUSHA),
    (0.225, "feature pipeline\nVST-HVG $\\to$ z-score $\\to$ PCA/kNN", ZHUSHA),
    (0.475, "HMRF fit\nMF-VI (25) / graph cut", SHIQING),
    (0.700, "domains\n$\\hat{z}$", SHIQING),
]
w, h, y0 = 0.185, 0.30, 0.47
for x0, txt, col in boxes:
    axf.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                 linewidth=1.1, edgecolor=col, facecolor=col + "14",
                                 transform=axf.transAxes))
    axf.text(x0 + w / 2, y0 + h / 2, txt, ha="center", va="center", fontsize=8.2,
             transform=axf.transAxes)
for xa, xb in ((0.200, 0.225), (0.410, 0.475), (0.660, 0.700), (0.885, 0.920)):
    axf.add_patch(FancyArrowPatch((xa, y0 + h / 2), (xb, y0 + h / 2),
                                  arrowstyle="-|>", mutation_scale=11, lw=1.1,
                                  color="#444444", transform=axf.transAxes))
axf.text(0.920, y0 + h / 2, "report", fontsize=8.0, va="center", ha="left",
         color="#333333", transform=axf.transAxes, rotation=0)
# oracle branch
axf.add_patch(FancyBboxPatch((0.225, 0.045), 0.45, 0.26,
                             boxstyle="round,pad=0.012,rounding_size=0.02",
                             linewidth=1.0, edgecolor=GREY, facecolor="#F2F3F5",
                             transform=axf.transAxes, linestyle="--"))
axf.text(0.45, 0.175, "oracle reference: same pipeline, true parameters\n"
                      "$\\Rightarrow$ separates data informativeness from method gap",
         ha="center", va="center", fontsize=8.2, color="#333333", transform=axf.transAxes)
axf.add_patch(FancyArrowPatch((0.545, 0.31), (0.595, 0.465), arrowstyle="-|>",
                              mutation_scale=10, lw=1.0, color=GREY,
                              linestyle="--", transform=axf.transAxes))

def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"))
    print("saved", name)

save(fig, "fig1_method_overview")
