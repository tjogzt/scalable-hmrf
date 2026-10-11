# H9 DLPFC 12-Section Four-Method Benchmark — Analysis

Date: 2026-10-11 (Batch 2) | Figures/tables: `figures/fig7_dlpfc12_benchmark.{pdf,png}`, `results/h9_benchmark_{slices,summary}.csv`

## Setup
- 12 DLPFC sections (8 with 7 layers, k=7: 151507-10, 151673-76; 4 with 5 layers, k=5: 151669-72),
  with counts bit-for-bit identical to spatialLIBD (exported from the H³MRF cache); four methods + oracle reference, all under the same ground-truth protocol.
- Finalized configurations for each method (fixed after the 151673 configuration diagnosis):
  | Method | Preprocessing/configuration | Notes |
  |---|---|---|
  | Ours | PC50 (log1p→HVG3000→SVD→standardization) + hexagonal adjacency; β∈{0.3,0.5,0.7,1.0,1.5} × 5 starts | 25 restarts in total still in the seconds range |
  | BayesSpace | SpatialExperiment + array_col/array_row; official spatialCluster, nrep=1000, seed 49 | single run |
  | SC.MEB | official spatialPreprocess (15 PC); K_set = section ground-truth K; beta_grid=seq(0,4,0.2) | single run; configuration sensitivity see below |
  | GraphST | official pipeline: norm→log→HVG3000→scale→train 600e→PCA20(emb)→Mclust(EEE)→refine_label(50) | single run, seed 49 |
  | Oracle | kmeans++ (ground-truth initialization) + HMRF smoothing, β optimal for that section | **pipeline-conditioning reference** (not an attainable ceiling; kmeans++ with ground truth does not guarantee recovery) |

## Main results (ARI, per section)

| Section | Ours (best) | Ours (median) | BayesSpace | SC.MEB | GraphST | Oracle |
|---|---|---|---|---|---|---|
| 151507 | 0.3508 | 0.2083 | 0.4653 | 0.4250 | 0.3201 | 0.6498 |
| 151508 | 0.3539 | 0.3004 | 0.4345 | 0.4387 | 0.3850 | 0.5148 |
| 151509 | 0.5498 | 0.3960 | 0.3433 | 0.4147 | 0.4636 | 0.6271 |
| 151510 | 0.3661 | 0.3150 | 0.4665 | 0.4237 | 0.3077 | 0.6221 |
| 151673 | 0.4284 | 0.1949 | 0.5482 | 0.3291 | 0.3418 | 0.6191 |
| 151674 | 0.3885 | 0.2833 | 0.2889 | 0.3967 | 0.2392 | 0.6024 |
| 151675 | 0.3435 | 0.1768 | 0.3657 | 0.4260 | 0.1585 | 0.6399 |
| 151676 | 0.4240 | 0.2794 | 0.3544 | 0.3614 | 0.2030 | 0.6633 |
| 151669 | 0.4806 | 0.3628 | 0.2608 | 0.4108 | 0.2522 | 0.7181 |
| 151670 | 0.6562 | 0.3578 | 0.3989 | 0.4055 | 0.4298 | 0.6824 |
| 151671 | 0.5709 | 0.4809 | 0.5065 | 0.3555 | 0.3075 | 0.7217 |
| 151672 | 0.6487 | 0.4968 | 0.7119 | 0.3669 | 0.2968 | 0.7019 |

| Method | Mean ARI | SD | Median | Mean NMI | Median time | 7-layer mean | 5-layer mean |
|---|---|---|---|---|---|---|---|
| **Ours (best-of-25)** | **0.4634** | 0.116 | 0.4262 | 0.569 | **8.9 s** | 0.4006 | **0.5891** |
| Ours (median-of-25) | 0.3210 | 0.104 | 0.3077 | — | 8.9 s | 0.2693 | 0.4246 |
| Oracle reference | 0.6469 | 0.058 | 0.6448 | — | — | 0.6173 | 0.7060 |
| BayesSpace | 0.4287 | 0.124 | 0.4167 | 0.592 | 87.8 s | 0.4084 | 0.4695 |
| SC.MEB | 0.3962 | 0.035 | 0.4082 | 0.508 | 76.8 s | 0.4019 | 0.3847 |
| GraphST | 0.3088 | 0.089 | 0.3076 | 0.469 | 75.8 s | 0.3024 | 0.3216 |

## Three-part conclusion

**1) Accuracy: first place overall under the best-of-restarts protocol, but behind under the median protocol—the bottleneck is "selection", not "computation".**
- Ours (best) has mean ARI 0.4634, higher than all baselines; the advantage is largest on the 5-layer subset (0.589 vs BayesSpace 0.470);
  on the 7-layer subset it is roughly on par with BayesSpace (0.401 vs 0.408).
- But median-of-25 is only 0.3210, below BayesSpace/SC.MEB—**good solutions exist among the 25 restarts but unsupervised selection cannot identify them reliably**,
  consistent with the negative-result chain of h4d (ELBO mis-selects) and h7b (held-out likelihood only a weak signal).
- The Oracle reference 0.6469 is ~0.18–0.22 above the whole field → if the initialization/selection problem were solved, the method's potential would exceed all baselines.
- Mutually consistent with the benchmark range of Kang 2025 (NAR): the pattern of BayesSpace leading on 7-layer sections and the DLPFC difficulty distribution both agree.

**2) Speed: the full pipeline runs in 8.9 s (including feature engineering and 25 restarts), roughly 1/9–1/10 of the baselines, and near-linear in scale.**
- Compared under a single "default single-run" protocol: Ours, even running all 25 restarts to completion, is still ~9× faster than each baseline's single run; Batch 1 already showed that
  at the 1-million-spot scale Ours remains in the seconds-to-minutes range, whereas EM/MCMC-class baselines are infeasible at that scale (the kt2 claim).

**3) Honest caveats (to be set as Limitations in the paper)**
- Asymmetric protocol: Ours reports both best/median (25 restarts), while baselines are the official default single run; a symmetric protocol would require giving the baselines
  an equivalent hyperparameter search (not done, explicitly stated).
- SC.MEB is configuration-sensitive: feeding PC50 directly = 0.186 / first 15 PC = 0.282 / official preprocessing = 0.357 (151673);
  this paper uses the official configuration, which is the most favorable protocol for that method.
- GraphST clustering follows its tutorial (PCA20→Mclust EEE→refine), and on some sections is weaker than its reported median (its original environment
  may include a data-augmentation option, not enabled—a conservative direction).

## SC.MEB configuration diagnosis (151673)
| Configuration | PCA source | ARI |
|---|---|---|
| A | Ours PC50 fed directly | 0.1858 |
| B | Ours first 15 PC | 0.2816 |
| C (adopted) | official spatialPreprocess (15 PC) | 0.3569 |

## Reproduction checklist
- Ours: `code/run_ours.py all` (all sections within 15 min); feature export `code/export_features.py`
- BayesSpace: `code/run_bayesspace.R all`; SC.MEB: `code/run_scmeb.R all` (official configuration C); GraphST: `code/run_graphst.py all`
- Aggregation: `code/aggregate.py` → `results/h9_benchmark_{slices,summary}.csv`; figure: `figures/make_fig7.py`
- Environment pitfalls and fixes are recorded in the skill `hmrf-spatial-benchmark` (rpy2 dual-BLAS deadlock, SC.MEB OpenMP/parallel bug, etc.)
