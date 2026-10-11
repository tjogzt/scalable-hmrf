# H2c Analysis: NB VI-EM Convergence Improvements

## Changes
- log1p + per-gene standardized initialization (used only for the hard-EM mode determination; mu is then re-estimated from the raw counts)
- 3 starts selected by ELBO; tol 1e-3, max_iter 300.

## Results (oracle ceiling 0.857/0.868, oracle with fixed beta=0.7)
| S | beta=0.5 | beta=0.7 | beta=1.0 | oracle@0.7 |
|---|---|---|---|---|
| 250 | 0.491 | 0.667 | 0.740 | 0.857 |
| 500 | 0.514 | 0.713 | **0.890** | 0.868 |

## Findings
1. The optimal beta in the NB scenario is clearly higher than in the Gaussian scenario (0.7→1.0+): weak unary terms need a stronger spatial prior — on real data Direction Two should perform beta adaptation/gridding (connecting to the review's pain point that "the smoothing parameter is globally fixed").
2. Multiple starts + log1p initialization pushed S=500 VI-EM to 0.890, surpassing the fixed-beta oracle (the oracle did not tune beta, so this is not a ceiling contradiction; but it shows that the "oracle ceiling" must be defined over a beta grid).
3. S=250 still has a 0.12 gap: small grids + high zero fractions make local optima more stubborn; one can try beta=1.2-1.5 and more starts (low cost).
4. Conclusion: the NB path is usable; the evidence chain for the paper's "count emission + VI-EM" section is complete.
