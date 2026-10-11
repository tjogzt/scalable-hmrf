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
"""fig7: 12 切片基准对比（本方法 best/median vs BayesSpace/GraphST/SC.MEB + Oracle 参照）。
数据源: experiments/h9-dlpfc12-benchmark/results/h9_benchmark_slices.csv（由 aggregate.py 生成）。
输出: paper/figures/fig7_dlpfc12_benchmark.{pdf,png}
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "Arial", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.direction": "out", "ytick.direction": "out",
})
ZHUSHA = "#C23531"; SHIQING = "#3D6BA8"; YANZHI = "#9D2933"; DIANQING = "#177CB0"; GREY = "#8A9299"
PROJ = _ROOT
RES = os.path.join(PROJ, "experiments/h9-dlpfc12-benchmark/results")
FIG = next(p for p in (os.path.join(PROJ, "paper", "figures"), os.path.join(PROJ, "figures"))
           if os.path.isdir(p))  # 兼容工作区与发布两种布局

def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"))
    print("saved", name)

tab = pd.read_csv(os.path.join(RES, "h9_benchmark_slices.csv"), dtype={"slice": str})
SEVEN = {"151507", "151508", "151509", "151510", "151673", "151674", "151675", "151676"}
_all = sorted(tab["slice"].unique())
slices = [s for s in _all if s in SEVEN] + [s for s in _all if s not in SEVEN]  # 7 层块在前、5 层块在后
x = np.arange(len(slices))

methods = ["Ours(best)", "Ours(median)", "BayesSpace", "GraphST", "SC.MEB"]
style = {
    "Ours(best)":   dict(color=ZHUSHA, marker="o", ms=4.2, filled=True),
    "Ours(median)": dict(color=ZHUSHA, marker="o", ms=4.2, filled=False),
    "BayesSpace":   dict(color=SHIQING, marker="s", ms=4.0, filled=True),
    "GraphST":      dict(color=DIANQING, marker="^", ms=4.6, filled=True),
    "SC.MEB":       dict(color=YANZHI, marker="D", ms=3.8, filled=True),
}
offsets = np.linspace(-0.30, 0.30, len(methods))

fig, ax = plt.subplots(figsize=(7.3, 3.0))

# Oracle 参照（背景层）
handles = {}
oracle = tab[tab.method == "Oracle"].set_index("slice").loc[slices, "ari"]
ln_or, = ax.plot(x, oracle.values, ls="none", marker="*", ms=6, mfc="white", mec=GREY,
                 mew=0.9, zorder=1, label="Oracle (ref.)")
handles["Oracle"] = ln_or

tmed = {}
for mi, m in enumerate(methods):
    sub = tab[tab.method == m].set_index("slice")
    if sub.empty:
        print("skip missing method:", m); continue
    ys = [sub.loc[s, "ari"] if s in sub.index else np.nan for s in slices]
    tser = sub["t_s"].dropna()
    tmed[m] = float(np.median(tser)) if len(tser) else np.nan
    st = style[m]
    ln, = ax.plot(x + offsets[mi], ys, ls="none", marker=st["marker"], ms=st["ms"],
                  mfc=(st["color"] if st["filled"] else "white"), mec=st["color"],
                  mew=1.0, zorder=3 if st["filled"] else 4,
                  label=f"{m}  (~{tmed[m]:.0f} s)" if np.isfinite(tmed[m]) else m)
    handles[m] = ln

# 组间分隔（7 层 | 5 层）
cut = sum(1 for s in slices if s in SEVEN)
ax.axvline(cut - 0.5, color="0.75", lw=0.7, ls=(0, (4, 3)), zorder=0)
ax.text((cut - 1) / 2, 0.80, "7 layers (k=7)", ha="center", fontsize=7.5, color="0.35")
ax.text(cut + (len(slices) - cut - 1) / 2, 0.80, "5 layers (k=5)", ha="center", fontsize=7.5, color="0.35")

ax.set_xticks(x); ax.set_xticklabels(slices, rotation=45, ha="right")
ax.set_ylabel("ARI (vs. ground truth)")
ax.set_ylim(-0.03, 0.85); ax.set_xlim(-0.6, len(slices) - 0.4)
order = ["Ours(best)", "BayesSpace", "Ours(median)", "GraphST", "Oracle", "SC.MEB"]
hs = [handles[m] for m in order if m in handles]
ax.legend(hs, [h.get_label() for h in hs], loc="upper center", ncol=3, frameon=False,
          columnspacing=1.2, handletextpad=0.4, bbox_to_anchor=(0.5, 1.18))
save(fig, "fig7_dlpfc12_benchmark")
