"""Generate Figures 2-5 (benchmark, runtime, stress, topology).

Author : Tao Zhu <zhutao@tjh.tjmu.edu.cn>
Created: 2026-10-02
Updated: 2026-10-20

Inputs
------
Result tables under experiments/*/results/.
Configuration: config/defaults.yaml, loaded through hmrf.config.

Outputs
-------
figures/output/fig2..fig5.pdf.

Usage
-----
    python 02_make_figs_2_5.py [options]
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
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.direction": "out", "ytick.direction": "out",
})

PAL = CFG["figures"]["palette"]
ZHUSHA = PAL["zhusha"]; SHIQING = PAL["shiqing"]; YANZHI = PAL["yanzhi"]
DIANQING = PAL["dianqing"]; GREY = PAL["grey"]
PROJ = _ROOT
FIG = str(figures_dir())
os.makedirs(FIG, exist_ok=True)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"))
    print("saved", name)


# ---------------- fig3: graph cut vs mean field (negative result) ----------------
h3b = pd.read_csv(os.path.join(PROJ, "experiments/09_stress_tests/results/h3b_seeds_results.csv"))
h3b["dE_pct"] = (h3b.e_mf - h3b.e_gc) / h3b.e_mf * 100.0   # >0: graph cut attains lower energy
g = h3b.groupby(["sigma", "muc", "beta"]).agg(
    dE=("dE_pct", "mean"), dE_sd=("dE_pct", "std"),
    dA=("d_ari", "mean"), dA_sd=("d_ari", "std")).reset_index()
labels = [f"$\\sigma$={r.sigma:g}, $\\mu$={r.muc:g}, $\\beta$={r.beta:g}" for r in g.itertuples()]
dE = g.dE.values; dA = g.dA.values
print("fig3 dE range:", dE.min(), dE.max(), "| dA(mean) range:", dA.min(), dA.max())

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
x = np.arange(len(g))
axes[0].bar(x, dE, yerr=g.dE_sd.values, color=ZHUSHA, width=0.62, error_kw=dict(lw=0.8, capsize=2))
axes[0].axhline(0, color="k", lw=0.7)
axes[0].set_title("(a) Energy advantage of graph cut")
axes[0].set_ylabel("$\\Delta$Energy$(\\mathrm{MF}-\\mathrm{GC})/E_{\\mathrm{MF}}$  (%)")
axes[1].bar(x, dA, yerr=g.dA_sd.values, color=[SHIQING if v >= 0 else YANZHI for v in dA], width=0.62,
            error_kw=dict(lw=0.8, capsize=2))
axes[1].axhline(0, color="k", lw=0.7)
axes[1].set_title("(b) Label accuracy (5 seeds, mean $\\pm$ sd)")
axes[1].set_ylabel("$\\Delta$ARI (GC $-$ MF)")
for ax in axes:
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=32, ha="right", fontsize=7)
maxE = dE.max(); maxA = np.abs(dA).max()
axes[0].annotate(f"always lower energy\n(max {maxE:.3f}%)", xy=(1.2, maxE * 0.94), fontsize=7.5, color=ZHUSHA)
axes[1].annotate(f"sign inconsistent\n(mean |$\\Delta$ARI| $\\leq$ {np.abs(dA).max():.3f})", xy=(0.6, maxA * 1.35),
                 fontsize=7.5, color=GREY)
fig.tight_layout()
save(fig, "fig3_graphcut_negative"); plt.close(fig)

# ---------------- fig5: beta selection ----------------
h5b = pd.read_csv(os.path.join(PROJ, "experiments/14_boundary_criterion/results/h5b_results.csv"))
h5b["beta_num"] = pd.to_numeric(h5b["beta"], errors="coerce")
nb = h5b[(h5b.scene == "nb_sim") & h5b.beta_num.notna()]
dl = h5b[(h5b.scene == "dlpfc") & h5b.beta_num.notna()]
h4 = pd.read_csv(os.path.join(PROJ, "experiments/11_dlpfc_single/results/h4_final.csv"))
h4["elbo_rel"] = (h4.elbo_best - h4.elbo_best.min()) / 1e3   # relative to beta=0.3, units of 10^3

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
ax = axes[0]
ax.plot(nb.beta_num, nb.ari, "-o", ms=3.5, color=SHIQING, label="NB sim (250K)")
ax.plot(dl.beta_num, dl.ari, "-s", ms=3.5, color=ZHUSHA, label="DLPFC 151673")
ax.axvline(1.3, color=SHIQING, lw=0.7, ls=":")
ax.axvline(0.7, color=ZHUSHA, lw=0.7, ls=":")
ax.text(1.34, 0.665, "best 1.3", fontsize=7, color=SHIQING, va="center")
ax.text(0.66, 0.60, "best 0.7", fontsize=7, color=ZHUSHA, va="center", ha="right")
ax.set_xlabel("$\\beta$"); ax.set_ylabel("ARI"); ax.set_title("(a) Accuracy vs $\\beta$")
ax.legend(frameon=False, loc="lower right")
ax = axes[1]
ax.plot(nb.beta_num, nb.br, "-o", ms=3.5, color=SHIQING)
ax.plot(dl.beta_num, dl.br, "-s", ms=3.5, color=ZHUSHA)
ax.set_xlabel("$\\beta$"); ax.set_ylabel("Boundary ratio"); ax.set_title("(b) Boundary ratio")
ax = axes[2]
ax.plot(nb.beta_num, (nb.elbo - nb.elbo.min()) / 1e3, "-o", ms=3.5, color=SHIQING, label="NB sim")
ax.plot(h4.beta, h4.elbo_rel, "-s", ms=3.5, color=ZHUSHA, label="DLPFC")
ax.set_xlabel("$\\beta$"); ax.set_ylabel("$\\Delta$ELBO ($\\times 10^3$)")
ax.set_title("(c) ELBO is monotone in $\\beta$")
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
save(fig, "fig7_beta_selection"); plt.close(fig)

# ---------------- fig6: oracle attribution ----------------
meta = pd.read_json(os.path.join(PROJ, "experiments/11_dlpfc_single/results/h4_final_meta.json"), typ="series")
h4b = pd.read_csv(os.path.join(PROJ, "experiments/11_dlpfc_single/results/h4b_results.csv"))
# Oracle ceilings of the pipeline ablation: v1=0.308 / v2=0.666 from the h4 analysis (diagnostic process), v3=0.6215 from meta.json
v1_ceiling = 0.308; v2_ceiling = 0.666
v3_ceiling = float(meta["oracle_best"])
h2b = pd.read_csv(os.path.join(PROJ, "experiments/04_nb_redesign/results/h2b_results.csv"))
h2c = pd.read_csv(os.path.join(PROJ, "experiments/05_nb_convergence/results/h2c_results.csv"))

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))

ax = axes[0]
vals = [float(meta["oracle_best"]), h4.ari_max.max(), float(h4.loc[h4.beta == 0.7, "ari_median"].iloc[0])]
names = ["Oracle\nreference", "VI-HMRF\nbest-of-5", "VI-HMRF\nmedian"]
cols = [GREY, ZHUSHA, "#E0A3A0"]
bars = ax.bar(range(3), vals, color=cols, width=0.62)
for i, v in enumerate(vals):
    ax.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=8)
ax.set_xticks(range(3)); ax.set_xticklabels(names, fontsize=7.5)
ax.set_ylim(0, 0.72); ax.set_ylabel("ARI")
ax.set_title("(a) DLPFC 151673")
ax.annotate(f"{vals[1]/vals[0]*100:.0f}% of oracle", xy=(1, vals[1] + 0.05), ha="center", fontsize=7.5, color=ZHUSHA)

ax = axes[1]
nb_oracle = float(h2c.loc[(h2c.S == 500) & (h2c.beta == 0.7), "ari_oracle"].iloc[0])   # 0.868 @250K fixed beta=0.7
vi_tuned = float(h2c.loc[(h2c.S == 500) & (h2c.beta == 1.0), "ari_viem"].iloc[0])   # 0.890
vi_at_oracle_beta = float(h2c.loc[(h2c.S == 500) & (h2c.beta == 0.7), "ari_viem"].iloc[0])  # 0.713
vals = [nb_oracle, vi_at_oracle_beta, vi_tuned]
names = ["Oracle ref.\n(fixed $\\beta$=0.7)", "VI-EM\n($\\beta$=0.7)", "VI-EM\ntuned $\\beta$=1.0"]
cols = [GREY, "#A8C4E0", SHIQING]
ax.bar(range(3), vals, color=cols, width=0.62)
for i, v in enumerate(vals):
    ax.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=8)
ax.set_xticks(range(3)); ax.set_xticklabels(names, fontsize=7.5)
ax.set_ylim(0, 1.0); ax.set_ylabel("ARI")
ax.set_title("(b) NB simulation (250K)")

ax = axes[2]
vals = [v1_ceiling, v2_ceiling, v3_ceiling]
names = ["v1: basic HVG", "v2: VST+z-score", "v3: +PCA+kmeans++"]
cols = ["#BCD0E6", "#7FA6CD", DIANQING]
ax.bar(range(3), vals, color=cols, width=0.62)
for i, v in enumerate(vals):
    ax.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=8)
ax.set_xticks(range(3)); ax.set_xticklabels(names, fontsize=7, rotation=30, ha="right")
ax.set_ylim(0, 0.80); ax.set_ylabel("Oracle-reference ARI")
ax.set_title("(c) Feature pipeline ceiling")
fig.tight_layout()
save(fig, "fig5_oracle_attribution"); plt.close(fig)
print("FIGURES DONE")
