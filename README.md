# Scalable-HMRF

Scalable mean-field variational inference for hidden Markov random fields (HMRFs) on spatial
transcriptomics, with exact inference (graph cuts, blocked Gibbs) as audited references.

Code, results and figure sources for the manuscript *"Mean-Field Variational HMRF for
Million-Spot Spatial Domain Identification: Exact Inference Is Not the Bottleneck"*.

## Headline results

| Claim | Evidence |
|---|---|
| Mean-field VI-EM scales without accuracy loss | ARI ≥ 0.99 from 62.5K to 1M simulated spots (5 instance seeds × restarts); 154–201× measured speed-up vs a Gibbs baseline given oracle parameters (16–65× under a robust 3-restart policy) |
| Exact inference is not the bottleneck | α-expansion graph cuts always lower the Potts energy but never raise ARI (Δ ≤ 0.006, sign inconsistent across settings) |
| Competitive accuracy on real data at ~10× lower cost | 12 DLPFC sections: best-of-restart **0.463** mean ARI vs BayesSpace 0.429 / SC-MEB 0.396 / GraphST 0.309; 8.9 s vs 76–88 s per section. Paired Wilcoxon (n=12): vs GraphST p=0.0015 (11/12), vs BayesSpace/SC-MEB not significant (p=0.62/0.18) — competitiveness, not dominance. Under a matched restart budget (best-of-5 seeds), BayesSpace reaches 0.478 > ours 0.427 ≈ SC-MEB 0.441 (p=0.68); reported transparently in the manuscript |
| Single-cell-resolution scale | Xenium breast IDC, **574,852 cells**: ~10 s preparation + 26–72 s per fit; ARI 0.488 vs the 10x k-means-8 reference; head-to-head vs GraphST 0.488 in ~80 s vs 0.283 in 68 min |
| Selection, not capacity, is the binding constraint | ELBO, held-out predictive likelihood (5 sections × 3 masks) and boundary-elbow criteria all fail as automatic selectors; restart spread is reproduced at every scale |

## Repository layout

```
config/
  defaults.yaml            all datasets, seeds, grids and palettes (single source of truth)
  slices_dlpfc12.txt       section list shared with the R baselines
src/hmrf/
  config.py                YAML loader (env overrides, no constants in analysis code)
  paths.py                 repository/data/results/figure path resolution
  solvers.py               MF-VI, VI-EM (Gaussian / NB), blocked Gibbs, ARI
  graphcut.py              α-expansion graph cuts (grid and arbitrary graph)
  simulate.py              Potts / Gaussian / NB synthetic generators
experiments/
  01_vi_vs_gibbs/          VI vs Gibbs reference (synthetic)
  02_vi_em/                variational EM sanity checks
  03_scaling/              runtime scaling
  04_nb_redesign/          NB regime redesign
  05_nb_convergence/       NB convergence
  06_multiseed_metrics/    multi-seed unified metrics + timing
  07_nb_ceiling_grid/      informativeness ceiling over β for both NB parameterizations
  08_graphcut/             graph-cut benchmark + construction unit tests
  09_stress_tests/         stress regimes + seed sweep
  10_knn_topology/         irregular kNN topologies
  11_dlpfc_single/         single-section DLPFC pipeline
  12_dlpfc_metrics/        unified metrics on one section
  13_beta_adaptive/        adaptive β rules
  14_boundary_criterion/   boundary-based selection criteria
  15_gibbs_timing/         measured Gibbs cost
  16_predictive_beta/      held-out predictive β, initialization comparison
  17_dlpfc12_benchmark/    12-section benchmark vs BayesSpace / SC-MEB / GraphST
  18_xenium_574k/          574K-cell Xenium application, GraphST head-to-head, patch tests
figures/                   01–05 scripts + output/ (final PDF/PNG)
docs/                      workflow, data manifest, revision log
```

Every experiment directory contains `code/` (numbered entry points), `results/` (CSV/JSON/NPZ
outputs quoted in the manuscript) and, where applicable, `analysis.md` / `protocol.md`.

## Requirements

* Python ≥ 3.10 with `numpy`, `scipy`, `pandas`, `matplotlib`, `PyYAML`, `PyMaxflow`.
* R ≥ 4.2 with `BayesSpace`, `SC.MEB`, `SingleCellExperiment`, `SpatialExperiment`, `Matrix`
  (only for the baseline scripts in experiments 17 and the SCE export).
* `GraphST` (optional) for the Xenium baseline; point `HMRF_GRAPHST_DIR` at its install directory.
* Raw data: see `docs/data_manifest.md`. `data/` is not versioned.

```bash
python -m pip install -r requirements.txt
export HMRF_XENIUM_DIR=/path/to/Xenium_V1_FFPE_Human_Breast_IDC_outs
export HMRF_GRAPHST_DIR=/path/to/site-packages/GraphST   # optional
```

## Quick start

All entry points discover the repository root from `config/defaults.yaml`, so they run from any
working directory:

```bash
python experiments/07_nb_ceiling_grid/code/01_run_beta_ceiling.py
python experiments/17_dlpfc12_benchmark/code/01_run_ours.py all
python figures/04_make_fig6.py
```

`HMRF_ROOT` overrides the root, `HMRF_SEED` overrides the project seed (49), `HMRF_CONFIG`
points to an alternative YAML file.

## Reproducing the manuscript

| Manuscript element | Command |
|---|---|
| Fig. 1 (method overview) | `python figures/01_make_fig1.py` |
| Figs. 2–5 (benchmark, runtime, graph-cut, stress) | `python figures/02_make_figs_2_5.py` |
| Figs. 6–7 (predictive β, Gibbs cost) | `python figures/03_make_figs_6_7.py` |
| Fig. 8 (Xenium) | `python figures/05_make_fig8.py` |
| 12-section benchmark table | `01_run_ours.py all` → `02_run_graphst.py all` → R baselines → `08_compute_metrics.py` → `09_aggregate_metrics.py` |
| Matched-budget control | `07_run_matched_budget.py all` → `10_analyze_matched_budget.py` |
| Xenium 574K result | `01_run_xenium.py full` (requires `HMRF_XENIUM_DIR`) |

`docs/workflow.md` lists the full order, expected runtimes and outputs.

## Reproducibility notes

* The project seed is 49 (`config/defaults.yaml`); per-experiment grids and seeds are stored in the
  same file and are identical to the values used for the published runs.
* Scripts write only into their own `results/` directory and never mutate raw data.
* Kernels are deterministic given the seed; GraphST uses its own seeding and is pinned by version.
* Negative results (graph cuts, selection criteria) are shipped with their scripts and outputs.

## Citation

Manuscript under review. If you use this code, please cite the repository (DOI to be added) and
the manuscript.

## License

MIT — see `LICENSE`.

## Contact

Tao Zhu — zhutao@tjh.tjmu.edu.cn
