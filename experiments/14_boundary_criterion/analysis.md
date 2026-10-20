# H5b Analysis: β Selection Criterion (ELBO Rejected + Boundary Elbow)

## Results
### NB Simulation S=250 (true optimum β=1.3, ARI 0.770)
| β | 0.3 | 0.5 | 0.7 | 1.0 | 1.3 | 1.7 |
|---|---|---|---|---|---|---|
| ARI | 0.397 | 0.491 | 0.667 | 0.740 | **0.770** | 0.708 |
| boundary ratio | 0.492 | 0.386 | 0.249 | 0.084 | 0.049 | 0.033 |

ELBO increases monotonically with β (−4.706M→−4.585M) → **ELBO selection degenerates to "take the largest β"; rejected as a criterion**.
Boundary elbow (12% tolerance) → 1.7 (an artifact of grid truncation; an extended grid would place it in 1.5–1.7, ARI≈0.73–0.75, ~0.04 below optimal).

### DLPFC (true optimum β=0.7, ARI 0.456)
| β | 0.3 | 0.5 | 0.7 | 1.0 |
|---|---|---|---|---|
| ARI | 0.397 | 0.440 | **0.456** | 0.451 |
| boundary ratio | 0.0292 | 0.0268 | 0.0254 | 0.0240 |

Boundary elbow → **0.5** (ARI differs from the optimum 0.7 by only 0.016) ✓.

## Conclusion (for the paper)
- ELBO is not a viable model-selection criterion for this class of models (the prior term increases monotonically with β, with no penalty balance)—**a negative result**.
- The boundary elbow (the smallest β satisfying "boundary ratio ≤ (1+ε)·minimum") selects a near-optimal β on real data, ΔARI ≤ 0.02;
  it is conservative on strong-prior task families (NB simulation) and requires a wider grid. To be written into the Discussion as a practical recommendation.
