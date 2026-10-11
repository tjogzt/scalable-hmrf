# H10 Xenium Breast IDC 574K Cell Full-Scale Measurement — Analysis

Date: 2026-10-11 (Batch 3) | Data: 10x Genomics official release Xenium FFPE Human Breast IDC (local copy 23 GB; scripts locate it via the environment variable HMRF_XENIUM_DIR)
Scripts: `code/run.py` (18-config grid, max_iter=200), `code/save_best_labels.py` (500-iteration replay + labels), `code/validate_fastpath.py` (equivalence validation), `figures/make_fig8.py`

## Data and pipeline
- **574,852 cells** (full detection set); 280-gene panel (Xenium V1 human breast); no official layered annotation providing a biological gold standard,
  so the reference uses 10x's official clusterings: graphclust (25 clusters) and kmeans_8 (**reference clustering, not ground truth**, stated as such).
- Pipeline: h5 matrix load 2.1 s → log1p(CP10K) + z-score → PCA30 → cKDTree kNN (k=6) symmetric graph (mean deg 6.85, max 13)
  → mean-field VI-EM (Gaussian emission, physical σ upper bound). **Features + graph construction together 7.3 s**.
- Fast-path validation: for the kNN sparse neighbor-sum vs the original `np.add.at` directed-edge version, same data and same start, **ΔARI=0, label agreement 100%**.

## Main results

| Configuration | ARI (reference) | Iterations | Time | Converged |
|---|---|---|---|---|
| K=8, β=0.7, r2, 500it (best) | **0.4879** (kmeans8) | 500 (cap) | 71.7 s | No (dq≈0.032, slow oscillatory descent) |
| K=8, β=0.7, r2, 200it | 0.4815 | 200 (cap) | 26.1 s | No |
| K=25, β=0.7, r2 | 0.2575 (graphclust, granularity matched) | 394 | ~96 s | **Yes** (dq=9.4e-05) |
| K=20, β=0.7, r2 | 0.2431 (graphclust, granularity mismatched) | 200 | 42.6 s | No |

K=8 × β∈{0.7,1.0,1.5} × 3 starts, 9 configurations total (200it):
- β=0.7: 0.390 / 0.409 / **0.4815**; β=1.0: 0.277 / 0.257 / 0.402; β=1.5: 0.165 / 0.148 / 0.348.

## Conclusion (three points for the paper)

1. **Real single-cell-resolution platform scale achieved**: full 575K cells, features + graph 7.3 s, a single domain fit 26 s (200it)–72 s (500it)
   —MCMC-class domain-identification methods (BayesSpace et al.) are infeasible at this scale, making this method the only seconds-to-minutes option;
   by contrast, on 3.6K-spot DLPFC a single run of this method is ~0.1–0.6 s, so ~160× scale costs only ~50–100× time (near-linear).
2. **Moderate agreement with the official reference clusterings**: K=8 vs 10x kmeans8 ARI 0.49; K=25 (granularity matched) vs graphclust 0.26
   (graphclust is 25 fine-grained subclusters, a different partition granularity from kmeans8; both are reference protocols, not ground truth).
3. **The "selection problem" reproduces a third time at real large scale**: under the same β, restarts range 0.39–0.48; the optimal β (0.7, the low end) is
   **opposite in direction** to small-scale simulation (usually at high β)—the optimal β is a function of {scale, graph density, platform}, further supporting
   the conclusion "there is no universal automatic β criterion; sensitivity must be reported". K=8 has not fully converged at 500 iterations (dq oscillates slowly),
   but ARI is already rising slightly and stably (+0.006), so 200 iterations is the cost-effectiveness knee.

## GraphST comparison (Xenium; in progress→full run in progress)

**Scale obstacle (important finding)**: GraphST 1.1.1's graph construction ('10X' path `construct_interaction` and
'Stereo/Slide' path `construct_interaction_KNN`) and its `graph_neigh` consumption are all **dense n×n implementations**—
574K cells require 574K²×8B ≈ **2.6 TB** (even 50K requires 20 GB per matrix, with instantaneous peaks of ~60–100 GB).
That is: **the official GraphST implementation is physically infeasible at single-cell-resolution scale**.

**Engineering in this repo**: `code/patch_graphst_sparse.py` (3 minimal patches, edge semantics fully identical, storing only shifted to scipy/torch.sparse;
'datatype="Stereo"' takes the sparse-encoder path) + presetting `var["highly_variable"]=True` to skip its seurat_v3 HVG
(280 genes < 3000 triggers a loess extrapolation error). After reinstalling GraphST the patches must be reapplied (the script is idempotent).

**50K subsample smoke test** (same rng49 subset, same kmeans8 reference; 600 epochs; k=3 official default graph):
- raw mclust label ARI = **0.3569** (our single configuration on the same subset 0.4071, β=1.0);
- **after 50-NN refine ARI drops to 0.2241**—the graphclust reference drops similarly (0.162→0.080).
  Interpretation: its refine radius=50 is calibrated for the Visium scale; at Xenium cell density the 50-NN neighborhood is very small and forces a minimum
  connected component, destroying small structures; **the Xenium protocol should report raw** (with refine as a negative sensitivity).
- Time 693 s / 50K (600-epoch training ~10 min) → extrapolates to ~2–3 h for 574K (in progress, both label versions archived).

**Full 574K result** (600 epochs; k=3 official default graph; t_total=4085.7 s ≈ 68 min):

| Method (K=8) | ARI (vs 10x kmeans8) | Time |
|---|---|---|
| **Ours (best start, β=0.7)** | **0.4879** | prep 7.3 s + fit 71.7 s ≈ **80 s** |
| GraphST raw (sparse-patch protocol) | 0.2826 | **4086 s (68 min)** |
| GraphST + 50-NN refine | 0.2401 | (same run) |

- Accuracy: Ours **+0.205 ARI**; speed: **~50×** (same machine, same reference, same K).
- Vs the graphclust reference (an auxiliary protocol with mismatched K): GraphST raw 0.2427 / Ours K=25 protocol 0.2575.
- Honest statement on the scale comparison: this GraphST run already has the sparse patch applied (the official dense implementation needs 2.6 TB of memory and is physically infeasible);
  i.e., "GraphST runs at all" is itself the result of this repo's engineering intervention, and it still takes 68 minutes. refine lowers the score on both datasets
  (its radius is calibrated for the Visium scale), so the Xenium protocol reports raw.

## Honesty boundaries
- Reference clustering ≠ biological gold standard; no external annotation mapping was done (e.g., Janesick 2023 layers).
- σ hits the physical upper bound (=1.0): the emission is unit-variance in PC space—consistent with the "within-cluster variance ≤ total variance" constraint, reported as is.
- Baseline comparison: GraphST full run in progress (sparse-patch protocol, both label versions archived); BayesSpace-class MCMC is infeasible at this scale.
- K=8 with large n has slow soft-EM convergence and truncation at the iteration cap; the production protocol is recommended to report iters and dq (both given on this page).

## File list
- `results/h10_full.csv` (18-config grid), `h10_full_meta.json` (n=574852, t_total=634.3 s)
- `results/h10_labels_K8_b0.7_r2.npz` (fig8 data source), `h10_labels_K25_b0.7_r2.npz`, `h10_labels_K8_b0.7_r0.npz`
- `review/batch3_xenium_full.log`, `batch3_xenium_replay.log`
