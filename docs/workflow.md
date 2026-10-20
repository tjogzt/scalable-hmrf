# Workflow and experiment map

Every entry point is self-contained: it loads `config/defaults.yaml`, resolves paths through
`hmrf.paths`, and writes into its own `experiments/<NN_name>/results/` directory. Run the
commands from the repository root (the scripts also work from any directory).

## 1. Synthetic studies (no external data)

| # | Directory | Entry point | Purpose | Output | Typical runtime |
|---|---|---|---|---|---|
| 1 | `01_vi_vs_gibbs` | `01_run_vi_vs_gibbs.py` | VI vs blocked Gibbs on a Potts field | `results/h1_results.csv` | < 1 min |
| 2 | `02_vi_em` | `01_run_vi_em.py` | VI-EM sanity check (Gaussian) | `results/h1b_results.csv` | < 1 min |
| 3 | `03_scaling` | `01_run_scaling.py` | Runtime vs grid size | `results/h2_results.csv` | ~2 min |
| 4 | `04_nb_redesign` | `01_run_nb_redesign.py` | Redesigned NB regime | `results/h2b_results.csv` | ~2 min |
| 5 | `05_nb_convergence` | `01_run_nb_convergence.py` | NB convergence | `results/h2c_results.csv` | ~2 min |
| 6 | `06_multiseed_metrics` | `01_run_multiseed.py`, `02_probe_convergence.py`, `03_analyze_timing.py` | Multi-seed metrics + timing | `results/h2d_*.csv` | 10–20 min |
| 7 | `07_nb_ceiling_grid` | `01_run_beta_ceiling.py` | β-grid informativeness ceiling (both NB parameterizations) | `results/h2e_results.csv` | 5–10 min |
| 8 | `08_graphcut` | `01_run_graphcut.py`, `02_test_graphcut.py` | Graph-cut benchmark + construction tests | `results/h3_results.csv` | < 2 min |
| 9 | `09_stress_tests` | `01_run_stress.py`, `02_run_seed_sweep.py` | Stress regimes and seed sweep | `results/h3b_*.csv` | 5–10 min |
| 10 | `10_knn_topology` | `01_run_knn_topology.py` | Irregular kNN topologies | `results/h3c_results.csv` | ~5 min |

## 2. Real-data studies (require `data/`)

| # | Directory | Entry point | Purpose | Output |
|---|---|---|---|---|
| 11 | `11_dlpfc_single` | `01_run_single_slice.py`, `02_diagnose_slice.py`, `03_run_final_config.py` | Single-section DLPFC pipeline | `results/h4_*.csv` |
| 12 | `12_dlpfc_metrics` | `01_run_metrics.py` | Unified metrics on one section | `results/h4d_metrics.csv` |
| 13 | `13_beta_adaptive` | `01_run_beta_adaptive.py` | Adaptive β rules | `results/h5_results.csv` |
| 14 | `14_boundary_criterion` | `01_run_boundary_criterion.py` | Boundary-based selection | `results/h5b_results.csv` |
| 15 | `15_gibbs_timing` | `01_measure_gibbs_timing.py` | Measured Gibbs cost | `results/h6_*.csv` |
| 16 | `16_predictive_beta` | `01_run_predictive_beta.py`, `02_compare_initializations.py` | Held-out predictive β; initialization comparison | `results/h7_*.csv` |

## 3. Twelve-section benchmark (BayesSpace / SC-MEB / GraphST)

Order matters; each step writes labels consumed by the metrics step.

```bash
cd experiments/17_dlpfc12_benchmark/code
python 01_run_ours.py all          # VI-HMRF, best-of-restart per section
python 02_run_graphst.py all       # GraphST baseline
Rscript 03_run_bayesspace.R all    # BayesSpace
Rscript 04_run_scmeb.R all         # SC-MEB
Rscript 05_run_bayesspace_multi.R  # multi-restart controls
Rscript 06_run_scmeb_multi.R
python 07_run_matched_budget.py all   # matched restart budget
python 08_compute_metrics.py          # ARI/NMI/boundary-F1 + paired Wilcoxon
python 09_aggregate_metrics.py        # manuscript table
python 10_analyze_matched_budget.py
python 11_k_sensitivity.py
python 12_donor_lmm.py
python 13_reference_robustness.py
python 14_export_features.py
Rscript 15_export_sce.R
```

Expected: ~10–20 min for VI-HMRF, 60–90 s per section for the R baselines, 68 min for GraphST
(per section on CPU).

## 4. Xenium 574K application

```bash
export HMRF_XENIUM_DIR=/path/to/Xenium_V1_FFPE_Human_Breast_IDC_outs
cd experiments/18_xenium_574k/code
python 01_run_xenium.py smoke      # 50K-cell smoke test
python 01_run_xenium.py full       # full 574,852 cells
python 02_run_graphst.py full      # GraphST head-to-head (needs HMRF_GRAPHST_DIR)
python 03_save_best_labels.py
python 04_validate_fast_path.py
python 05_check_markers.py
```

Sparse-patch equivalence (optional, requires GraphST):

```bash
export GRAPHST_DIR=/path/to/site-packages/GraphST
bash 08_run_patch_equivalence.sh
```

## 5. Figures

```bash
python figures/01_make_fig1.py        # Fig. 1
python figures/02_make_figs_2_5.py    # Figs. 2–5
python figures/03_make_figs_6_7.py    # Figs. 6–7
python figures/04_make_fig6.py        # Fig. 6 variant from h9 tables
python figures/05_make_fig8.py        # Fig. 8
```

All figures land in `figures/output/`.

## 6. Configuration and overrides

| Variable | Effect |
|---|---|
| `HMRF_ROOT` | Repository root (otherwise discovered from `config/defaults.yaml`) |
| `HMRF_CONFIG` | Alternative YAML configuration file |
| `HMRF_SEED` | Overrides `project.seed` (default 49) |
| `HMRF_XENIUM_DIR` | Xenium `outs` directory (required by experiment 18) |
| `HMRF_GRAPHST_DIR` | Installed GraphST package (required by 02/06/07 in experiment 18) |

Business constants live only in `config/defaults.yaml`; scripts read them through `hmrf.config`.
If you change a grid or seed there, outputs remain reproducible but will not match the shipped
`results/` files exactly.
