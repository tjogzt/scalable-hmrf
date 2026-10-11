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
"""Batch-1 figures (second version): fig2 scalability curve + measured Gibbs / fig4 DLPFC spatial-domain maps.
Output: paper/figures/fig2_scalability.{pdf,png}, fig4_dlpfc_maps.{pdf,png}
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "Arial", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
})

ZHUSHA = "#C23531"; SHIQING = "#3D6BA8"; YANZHI = "#9D2933"; DIANQING = "#177CB0"; GREY = "#8A9299"
PROJ = _ROOT
FIG = next(p for p in (os.path.join(PROJ, "paper", "figures"), os.path.join(PROJ, "figures"))
           if os.path.isdir(p))  # compatible with both the workspace and release layouts

long = pd.read_csv(f"{PROJ}/experiments/h2d-multiseed-metrics/results/h2d_results.csv")
summ = pd.read_csv(f"{PROJ}/experiments/h2d-multiseed-metrics/results/h2d_summary.csv")
anch = pd.read_csv(f"{PROJ}/experiments/h6-gibbs-measured/results/h6_anchors.csv")

# ---------------- fig2 ----------------
t1_med = long.groupby("S").t_s.median()          # median time of a single run
t3 = long.groupby(["S", "inst_seed"]).t_s.sum().groupby("S")  # total time of 3 restarts per instance
t3_mean = t3.mean(); t3_sd = t3.std()

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
ax = axes[0]
ns = summ.n.values
ax.errorbar(ns, summ.ari_mean3_mean, yerr=summ.ari_mean3_sd, fmt="-o", ms=4, lw=1.2,
            color=SHIQING, capsize=2.5, label="mean (3 starts)")
ax.errorbar(ns, summ.ari_max_mean, yerr=summ.ari_max_sd, fmt="-s", ms=4, lw=1.2,
            color=ZHUSHA, capsize=2.5, label="best (3 starts)")
ax.set_xscale("log"); ax.set_xlabel("spots $n$"); ax.set_ylabel("ARI")
ax.set_title("(a) Accuracy (5 instances, mean ± sd)")
ax.set_ylim(0.90, 1.005)
ax.legend(frameon=False, loc="lower left", fontsize=7)
ax.annotate("slow-converging\ninstances", xy=(6.9e4, 0.951), xytext=(6.0e5, 0.9025),
            fontsize=6.8, color=GREY, ha="center",
            arrowprops=dict(arrowstyle="-", color=GREY, lw=0.6, shrinkA=1, shrinkB=2))

ax = axes[1]
ax.plot(summ.n, t1_med.values, "-o", ms=4, lw=1.2, color=SHIQING, label="VI-EM, single run (median)")
ax.plot(summ.n, t3_mean.values, "-^", ms=4, lw=1.2, color=DIANQING, label="VI-EM, 3 restarts (mean)")
ax.plot(anch.n, anch.total_s, "-D", ms=4.5, lw=1.2, color=ZHUSHA, label="Gibbs, 1500 sweeps (measured)")
ax.plot([2.5e5, 1e6], [204, 817], ":", color=GREY, lw=1.0, label="Gibbs (old extrapolation)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("spots $n$"); ax.set_ylabel("wall time (s)")
ax.set_title("(b) Runtime")
ax.legend(frameon=False, loc="upper left", fontsize=7)

ax = axes[2]
gib = anch.set_index("S").total_s
scales = [500, 750, 1000]
sp_single = [gib[S] / t1_med[S] for S in scales]
sp_robust = [gib[S] / t3_mean[S] for S in scales]
ax.plot(scales, sp_single, "-o", ms=4, lw=1.2, color=SHIQING, label="vs single run")
ax.plot(scales, sp_robust, "-^", ms=4, lw=1.2, color=DIANQING, label="vs 3-restart policy")
ax.axhline(10, color=ZHUSHA, ls="--", lw=0.9)
ax.text(510, 10.8, "kt1 gate: 10$\\times$", fontsize=7, color=ZHUSHA)
ax.plot([500, 1000], [145, 76], "s", ms=4, mfc="none", mec=GREY, label="old claim")
ax.set_yscale("log"); ax.set_xticks(scales)
ax.set_xlabel("spots $n$"); ax.set_ylabel("speedup vs Gibbs")
ax.set_title("(c) Speedup (measured)")
ax.legend(frameon=False, loc="upper right", fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig2_scalability.pdf"))
fig.savefig(os.path.join(FIG, "fig2_scalability.png"))
print("saved fig2")
for S in scales:
    print(f"  S={S}: single {sp_single[scales.index(S)]:.0f}x, robust {sp_robust[scales.index(S)]:.0f}x")

# ---------------- fig4 ----------------
d = np.load(f"{PROJ}/experiments/h4d-dlpfc-metrics/results/h4d_labels.npz", allow_pickle=True)
truth = d["truth"]; rows = d["rows"]; cols = d["cols"]; oracle = d["oracle"]
inits = [d[f"init{i}"] for i in range(5)]
ari = [0.4557, 0.2503, 0.2121, 0.4213, 0.1302]
best = inits[0]; med = inits[1]
palette = ["#C23531", "#3D6BA8", "#E9B44C", "#6B8E5A", "#8E7CC3", "#D97B66", "#4FA3A5"]

fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.15))
panels = [("(a) Ground truth", truth, None),
          ("(b) Oracle ref.", oracle, 0.622),
          ("(c) VI-HMRF best", best, 0.456),
          ("(d) VI-HMRF median", med, 0.250)]
for ax, (title, lab, a) in zip(axes, panels):
    c = [palette[int(v)] for v in lab]
    ax.scatter(cols, rows, c=c, s=7, marker="h", linewidths=0)
    ax.invert_yaxis(); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color("#CCCCCC")
    t = title if a is None else f"{title} (ARI {a:.3f})"
    ax.set_title(t, fontsize=8.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig4_dlpfc_maps.pdf"))
fig.savefig(os.path.join(FIG, "fig4_dlpfc_maps.png"))
print("saved fig4")
print("FIGURES2 DONE")
