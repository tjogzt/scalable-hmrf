"""Generate the predictive-beta figure from result tables.

Author : Tao Zhu <zhutao@tjh.tjmu.edu.cn>
Created: 2026-10-02
Updated: 2026-10-20

Inputs
------
experiments/16_predictive_beta/results/.
Configuration: config/defaults.yaml, loaded through hmrf.config.

Outputs
-------
figures/output/fig6.pdf.

Usage
-----
    python 04_make_fig6.py [options]
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
from hmrf.paths import data_dir, figures_dir, results_dir  # noqa: E402

CFG = load_config()
_ROOT = str(REPO_ROOT)  # backwards-compatible alias used by legacy code paths

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
PAL = CFG["figures"]["palette"]
ZHUSHA = PAL["zhusha"]; SHIQING = PAL["shiqing"]; YANZHI = PAL["yanzhi"]
DIANQING = PAL["dianqing"]; GREY = PAL["grey"]
PROJ = _ROOT
RES = str(results_dir("17_dlpfc12_benchmark"))
FIG = str(figures_dir())

def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"))
    print("saved", name)

tab = pd.read_csv(os.path.join(RES, "h9_benchmark_slices.csv"), dtype={"slice": str})
SEVEN = set(CFG["experiments"]["h9"]["seven_slices"])
_all = sorted(tab["slice"].unique())
slices = [s for s in _all if s in SEVEN] + [s for s in _all if s not in SEVEN]  # 7-layer block first, 5-layer block after
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

# Oracle reference (background layer)
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

# group separator (7-layer | 5-layer)
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
save(fig, "fig6_dlpfc12_benchmark")
