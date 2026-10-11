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
"""H7b: at fixed beta, use the held-out likelihood to choose "which start solution is trustworthy" -- compared against ELBO selection.
h4d already showed: among 5 starts, ELBO selected one of the worst solutions (ARI 0.1302 vs best 0.4557).
This probe: same 5 starts (rng 100+t), beta=0.7, 5% mask, 3 masks; compares
the rank correlation (Spearman) of held-out LL / ELBO with the true ARI.
"""
import sys, os, csv, json
ROOT = _ROOT
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "experiments/h7-predictive-beta/code"))
sys.path.insert(0, os.path.join(ROOT, "experiments/h9-dlpfc12-benchmark/code"))
import numpy as np
from hmrf_solvers import adjusted_rand_index
from run_ours import load_slice
from run import masked_em_mf_gaussian, feature_pipeline, hex_edges
from scipy.stats import spearmanr

OUT = os.path.join(ROOT, "experiments/h7-predictive-beta/results")
BETA = 0.7
MASK_FRAC = 0.05
MASKS = (7, 8, 9)
SLICES = ["151673", "151509"]

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
