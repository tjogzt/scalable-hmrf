# Scalable HMRF

**Mean-field variational hidden Markov random field (HMRF) for spatial domain identification at
single-cell / million-spot scale.**

Code, results and figure sources for the manuscript
*"Mean-Field Variational HMRF for Million-Spot Spatial Domain Identification: Exact Inference Is
Not the Bottleneck"*.

## Headline results (all reproducible from this repository)

| Claim | Evidence | Where |
|---|---|---|
| Mean-field VI-EM scales without accuracy loss | ARI ≥ 0.993 from 62.5K to 1M simulated spots, 5 instance seeds × 3 restarts; **154–201× measured speedup** vs a Gibbs baseline granted oracle parameters (16–65× under a robust 3-restart policy) | `experiments/h2d-multiseed-metrics`, `h6-gibbs-measured` |
| Exact inference is not the bottleneck | α-expansion graph cuts always lower the energy but never raise ARI (Δ ≤ 0.006, sign inconsistent) | `experiments/h3-graphcut`, `h3b-stress` |
| Competitive accuracy on real data at ~10× lower cost | 12 DLPFC slices: best-of-restart **0.463** mean ARI vs BayesSpace 0.429 / SC-MEB 0.396 / GraphST 0.309; 8.9 s vs 76–88 s per slice. Paired Wilcoxon (n=12): **vs GraphST p=0.0015** (11/12 slices), vs BayesSpace/SC-MEB not significant (p=0.62/0.18) — competitiveness, not dominance; unified NMI / boundary-F1 reported alongside ARI | `experiments/h9-dlpfc12-benchmark` |
| Single-cell-resolution scale | Xenium breast IDC, **574,852 cells**: full pipeline ~10 s preparation + 26–72 s per fit; ARI 0.488 vs the 10x k-means-8 reference; **head-to-head with GraphST: 0.488 in ~80 s vs 0.283 in 68 min** | `experiments/h10-xenium574k` |
| Selection, not capacity, is the binding constraint | ELBO, held-out predictive likelihood (5 sections × 3 masks) and boundary-elbow criteria all fail as automatic selectors; restart spread reproduced on real data at every scale | `experiments/h7-predictive-beta`, `h4d-dlpfc-metrics` |

## Repository layout

```
src/                     core method (numpy only)
  hmrf_solvers.py        MF-VI / VI-EM (grid & graph), Gibbs reference
  graphcut_hmrf.py       alpha-expansion graph-cut solver (PyMaxflow)
  simulate.py            simulation base (smoothed Gaussian fields, Gaussian/NB emissions)
experiments/<name>/      one directory per experiment: code/ + results/ + analysis.md
figures/                 manuscript figures (PDF + PNG) and the scripts that generate them
```

## Data

Raw data are **not** included; both datasets are public.

- **DLPFC 12 slices** (Maynard et al. 2021): obtained via `spatialLIBD`. The 12-slice export used
  by this work is regenerated with `experiments/h9-dlpfc12-benchmark/code/export_sce.R`.
  Set `HMRF_DLPFC_CACHE` to a `dlpfc_sce_cache.rds` created with
  `spatialLIBD::fetch_data("sce")` (see the script header).
- **Xenium human breast IDC** (10x Genomics public release, panel `xenium_human_breast_gene_expression_panel`):
  download the `outs` bundle from 10x Genomics and point `HMRF_XENIUM_DIR` at it.

All scripts auto-detect the repository root (or honour `HMRF_ROOT`); no absolute user paths.

## Environment

- Python 3.12: `numpy`, `scipy`, `pandas`, `matplotlib`, `scikit-learn`, `anndata`, `scanpy`, `torch`.
- R 4.5: `BayesSpace`, `SC.MEB` (GitHub `feiyoung/SC-MEB`), `mclust`, `scater`/`scran`/`scuttle`.
- GraphST 1.1.1 (pip) — **requires the sparse-storage patch** in
  `experiments/h10-xenium574k/code/patch_graphst_sparse.py` to run at single-cell scale; the
  official implementation allocates dense n×n graphs (≈2.6 TB at 574K cells). The patch keeps
  edge semantics identical; re-run the script after any GraphST reinstall.

## Reproducing the paper

| Experiment | Command (from repo root) |
|---|---|
| Scale curve (h2d) | `python experiments/h2d-multiseed-metrics/code/run.py` |
| Measured Gibbs (h6) | `python experiments/h6-gibbs-measured/code/run.py` |
| Graph-cut negative (h3b) | `python experiments/h3b-stress/code/run.py` |
| DLPFC 12-slice benchmark (h9) | `python experiments/h9-dlpfc12-benchmark/code/run_ours.py all` → baselines via `run_bayesspace.R all`, `run_scmeb.R all`, `run_graphst.py all` → `python experiments/h9-dlpfc12-benchmark/code/aggregate.py`; unified metrics + paired tests via `code/metrics_all_methods.py` |
| Selection criteria (h7/h7b) | `python experiments/h7-predictive-beta/code/run.py`, `run_init_selection.py` |
| Xenium 574K (h10) | `python experiments/h10-xenium574k/code/run.py full` (+ `save_best_labels.py`, `run_graphst_xenium.py full`; marker validation via `code/marker_check.py`; sparse-patch equivalence via `code/e8_driver.sh`) |
| Matched restart budgets (BayesSpace/SC-MEB 5 seeds) | `Rscript experiments/h9-dlpfc12-benchmark/code/run_bayesspace_multi.R`, `run_scmeb_multi.R` → `python experiments/h9-dlpfc12-benchmark/code/run_matched_budget.py` → `analyze_matched_budget.py` |
| Reference robustness / donor LMM / K sensitivity | `code/ref_robustness.py`, `code/donor_lmm.py`, `code/k_sensitivity.py` |
| NB ceiling grid (h2e) | `python experiments/h2e-nb-ceiling-grid/code/run.py` |
| kNN topology check (h3c) | `python experiments/h3c-knn-topology/code/run.py` |
| Timing protocol (iterations/per-iteration) | `python experiments/h2d-multiseed-metrics/code/timing_analysis.py` |
| Stress dispersion (5 seeds) | `python experiments/h3b-stress/code/run_seeds.py` |
| Figures | `python figures/make_fig1.py`, `make_fig6.py`, `make_fig8.py`, `make_figures_batch1*.py` |

## Notes and honest caveats

- On real data, the 10x-provided clusterings are used as **reference partitions, not biological
  ground truth**; the "pipeline-conditioned reference" in the DLPFC benchmark is class-mean
  unaries from the truth labels + pure mean-field, maximised over the beta grid, and is itself
  construction-sensitive (0.21-0.63 across initialisation variants on 151673).
- Baseline-vs-ours is reported under two protocols: official single runs for baselines and the
  full multi-start protocol for ours, plus a matched-restart-budget control (equal restart counts;
  Table: `results/matched_budget/matched_budget_summary.csv`).
- Held-out likelihood is a weak (non-catastrophic) restart selector; ELBO-based selection is an
  anti-pattern on real data.

## License

Code released under the MIT License (see `LICENSE`). Public datasets remain under their original
terms; the 10x Xenium bundle and publisher supplementary files are not redistributed here.
