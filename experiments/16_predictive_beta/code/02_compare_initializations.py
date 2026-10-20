"""Compare initialization strategies for VI-EM.

Author : Tao Zhu <zhutao@tjh.tjmu.edu.cn>
Created: 2026-10-02
Updated: 2026-10-20

Inputs
------
DLPFC slices under data/dlpfc12/.
Configuration: config/defaults.yaml, loaded through hmrf.config.

Outputs
-------
results/h7_init_selection.csv.

Usage
-----
    python 02_compare_initializations.py [options]
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

sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "experiments/16_predictive_beta/code"))
sys.path.insert(0, os.path.join(ROOT, "experiments/17_dlpfc12_benchmark/code"))
import numpy as np
from hmrf.solvers import adjusted_rand_index
from run_ours import load_slice
from run import masked_em_mf_gaussian, feature_pipeline, hex_edges
from scipy.stats import spearmanr

OUT = str(results_dir("16_predictive_beta"))
BETA = CFG["experiments"]["h7"]["betas"][2]
MASK_FRAC = CFG["experiments"]["h7"]["mask_frac"]
MASKS = tuple(CFG["experiments"]["h7"]["masks"])
SLICES = list(CFG["experiments"]["h7"]["slices"])

rows = []
for sid in SLICES:
    X, rows_, cols_, truth = load_slice(sid)
    k = int(np.unique(truth).size)
    PC = feature_pipeline(X); del X
    edges = hex_edges(rows_, cols_)
    print(f"### {sid} k={k}", flush=True)
    for mseed in MASKS:
        rng = np.random.default_rng(mseed)
        mask = rng.random(len(truth)) < MASK_FRAC
        recs = []
        for t in range(5):
            q, mu, sig, iters, elbo, ll = masked_em_mf_gaussian(
                PC, k, BETA, edges, mask, seed=100 + t, n_init=1, max_iter=300, tol=1e-4)
            lab = np.argmax(q, axis=1)
            ari = adjusted_rand_index(truth, lab)
            rec = dict(slice=sid, mask_seed=mseed, init=t, heldout_ll=round(ll, 3),
                       elbo=round(elbo, 1), ari=round(ari, 4), iters=iters)
            rows.append(rec); recs.append(rec)
            print(rec, flush=True)
        ari_r = [r["ari"] for r in recs]
        ll_r = [r["heldout_ll"] for r in recs]
        elbo_r = [r["elbo"] for r in recs]
        rho_ll = spearmanr(ll_r, ari_r).correlation
        rho_elbo = spearmanr(elbo_r, ari_r).correlation
        best_by = {"ari": int(np.argmax(ari_r)), "ll": int(np.argmax(ll_r)), "elbo": int(np.argmax(elbo_r))}
        print(f"  mask {mseed}: spearman(LL,ARI)={rho_ll:.2f} spearman(ELBO,ARI)={rho_elbo:.2f} | "
              f"best init by ari/ll/elbo = {best_by['ari']}/{best_by['ll']}/{best_by['elbo']} "
              f"(ARI@{best_by['ari']}={ari_r[best_by['ari']]:.3f}, "
              f"ARI@{best_by['ll']}={ari_r[best_by['ll']]:.3f}, ARI@{best_by['elbo']}={ari_r[best_by['elbo']]:.3f})",
              flush=True)
        for r in recs:
            r["spearman_ll_ari"] = round(float(rho_ll), 3)
            r["spearman_elbo_ari"] = round(float(rho_elbo), 3)

FIELDS = ["slice", "mask_seed", "init", "heldout_ll", "elbo", "ari", "iters",
          "spearman_ll_ari", "spearman_elbo_ari"]
with open(os.path.join(OUT, "h7b_init_selection.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
    w.writeheader(); w.writerows(rows)
print("DONE h7b", flush=True)
