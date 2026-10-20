# Revision log (R1–R10)

Summary of the revision round that produced this release. Numbers below are the ones shipped in
`experiments/*/results/` and quoted in the manuscript.

| Item | Issue | Action | Evidence |
|---|---|---|---|
| R1 | Emission models mixed in the abstract; boundary-F1 missing | Abstract split into Gaussian vs NB regimes; boundary-F1 reported as metric-dependent (0.94 → 0.66) with an explicit no-tolerance definition | `main.tex` abstract; `17_dlpfc12_benchmark/results/h9_benchmark_metrics.csv` |
| R2 | Headline claim overstated | "Highest ARI" replaced by a matched-budget statement: ours best-of-5 0.427 ≈ SC-MEB 0.441 (p=0.68) < BayesSpace 0.478 (p=0.077); 20-run envelope 0.463 | `h9_benchmark_summary.csv`, `matched_budget_summary.csv` |
| R3 | Symmetric restart budget missing | BayesSpace 5 seeds + SC-MEB 5 seeds (103/104 added) across 12 sections; donor-blocked LMM | `07_run_matched_budget.py`, `12_donor_lmm.py` |
| R4 | Novelty framing | Related work rewritten (Zhang 2001, Krähenbühl 2011, EM/VI lineage) + method positioning table | `refs.bib` (+6 Crossref-verified entries) |
| R5 | Exactness scope | "Never raises ARI" removed; guarantee restricted to binary submodular / strong local optimum for multi-label; kNN topology experiment added | `10_knn_topology/results/h3c_results.csv` |
| R6 | Auditable timing | Per-iteration linear fit (2.8e-4 ms/spot, R²=0.95), iteration counts 12–300, 562.5K right-tail, hardware documented; median timing | `06_multiseed_metrics/results/h2d_timing_table.csv` |
| R7 | "Oracle" naming and single-β ceiling | Renamed to pipeline-conditioned reference; ceiling taken over β ∈ {0.3–3.0}: redesigned 0.995/0.997/0.998, original 0.891/0.895; construction robustness 0.21–0.63 | `07_nb_ceiling_grid/results/h2e_results.csv`, `13_reference_robustness.py` |
| R8 | Graph-cut dispersion / equivalence / donor effects | h3b 5 seeds × 6 settings (mean Δ +0.003, 90% CI [+0.0003, +0.005], within ±0.01 equivalence bound); donor random-intercept LMM (vs GraphST −0.155, p<0.001; vs SC-MEB −0.067, p=0.071); single-thread NumPy Gibbs disclosed | `09_stress_tests/results/h3b_seeds_*.csv`, `donor_lmm` output |
| R9 | Xenium truth anchoring | Eight-domain marker sanity check (epithelial, proliferative, myoepithelial, vascular, T cell, myeloid, adipose); σ physical cap and 500-iteration truncation disclosed; dense/sparse patch equivalence on 10K cells (identical graph tensor, ΔARI=0, 100% agreement) | `18_xenium_574k/results/marker_check.csv`, `patch_equiv_10k/equivalence_report.json` |
| R10 | Guidelines and K sensitivity | Discussion rewritten with eight limitations (K assumption, exactness ≤10K, matched budget = count not compute); K sensitivity K=5–10 (best 0.30–0.44 / median 0.18–0.30) | `17_dlpfc12_benchmark/results/k_sensitivity.csv` |

## Metric-definition changes propagated to the manuscript

* best-of-25 / median-of-25 → **best-of-20 / median-of-20** (4 β values × 5 starts).
* Robust speed-up 16–65× → **40–66×** (median over 3-restart policies); single-run 154–201× unchanged.
* NB "ceiling 0.47" → ceiling taken over the β grid (**≥0.89** original / **≥0.98** redesigned).
* Reference "0.647" → **0.65** (construction-sensitive 0.21–0.63, protocol stated).
* Figures 2, 3, 4, 5, 7 regenerated; Fig. S1 added; internal experiment codes (h1, h2d, h6, h9,
  h10, kt1, …) removed from the manuscript text and figure labels.
