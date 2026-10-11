# H4d Analysis: DLPFC 151673 Metric Completion (NMI + Boundary F1) and Label Archival (Post-review Batch 1)

## Purpose
1. Complete the NMI / boundary F1 for DLPFC (the original h4 reported only ARI);
2. Quantify the full distribution over 5 starts (best/median/ELBO selection)—responding to the "best-of-5 selection bias" criticism;
3. Archive truth/oracle/per-start labels (for fig4 plotting and subsequent analyses).

## Method
Re-run the h4 final configuration (VST-HVG3000 + z-score + PCA50 + k-means++ 5 starts, β=0.7, max_iter=300, tol=1e-4;
oracle reference: true-label means + pure MF, best over the β grid {0.3,0.5,0.7,1.0}). Metrics: ARI / NMI (arithmetic-mean normalization) / boundary F1 (strict edge set, 21038 hexagonal adjacency edges). The numbers reproduce h4_final exactly (init0=0.4557; the ELBO-optimal solution elbo=-242867.4 also matches).

## Results (h4d_metrics.csv + h4d_meta.json)

| Start | ARI | NMI | Boundary F1 | ELBO | Iterations |
|---|---|---|---|---|---|
| init0 | **0.4557** | 0.6029 | 0.1598 | −244009.8 | 220 |
| init1 | 0.2503 | 0.3889 | 0.0889 | −243148.8 | 273 |
| init2 | 0.2121 | 0.3309 | 0.0580 | −243431.2 | 162 |
| init3 | 0.4213 | 0.5972 | 0.0849 | −243529.9 | 81 |
| init4 | 0.1302 | 0.3211 | 0.0206 | **−242867.4 (ELBO-optimal)** | 55 |

- **ARI**: max 0.4557 / **median 0.2503** / ELBO-selected **0.1302**;
- **Oracle reference**: ARI 0.6215 / NMI 0.7185 / boundary F1 0.1846.

## Conclusion
1. **Quantified selection bias**: best-of-5 = 0.456, median = 0.250, ELBO selection = 0.130. When the paper reports 0.456 it must report median alongside; when comparing against the literature range (0.40–0.58) it should state that the max is used.
2. **ELBO failure confirmed again**: the optimal-ELBO solution (init4) is precisely the second-worst solution—on real data the negative result of ELBO selection extends from "two task families" to three decision points (h4/h5b/h4d).
3. **Low boundary F1 is a property of the metric**: the strict edge set has no tolerance, so a ±1 spot boundary shift yields near-zero overlap (the oracle is only 0.185 too). h4d retains this strict definition (reproducible, parameter-free); a "radius-1 tolerance" variant is recommended for subsequent work and the paper's main text; the current numbers are suitable only for **relative comparison between methods**.
4. NMI and ARI orderings agree (0.60 corresponds to 0.456; 0.32 to 0.13), with no inversions.

## Files
- results/h4d_metrics.csv, h4d_meta.json, **h4d_labels.npz** (truth/oracle/5-start labels + spatial coordinates + edge set, for fig4)
- Figures: figures/fig4_dlpfc_maps.* (four panels: truth/oracle/best/median)
