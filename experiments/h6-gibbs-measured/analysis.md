# H6 Analysis: Gibbs Baseline Measured (Replacing Extrapolation, Post-review Batch 1)

## Purpose
Eliminate h2's "exact Gibbs baseline is extrapolated" soft spot: replace the 76–145× speedup obtained by linear extrapolation from 2 points (2.5K/10K) with **measured** numbers.

## Method
- **Per-sweep cost**: S ∈ {50,100,150,200,250,300} (2.5K–90K), measured per-sweep point by point (with online counting), linear fit;
- **Full-scale anchors**: exactly the same protocol as the original extrapolation—1500 sweeps (burn=500), oracle emission parameters, β=0.7, instance seeds (11,12);
- The online-counting version has the same per-step workload as the original checkerboard_gibbs (it accumulates O(nK) counts instead of a sample stack, semantically still the posterior mode);
- Budget guard enabled, not triggered (all three tiers ran the full 1500 sweeps).

## Results

### Per-sweep cost (h6_sweepcost.csv)
| S | n | per-sweep (s) |
|---|---|---|
| 50 | 2.5K | 0.00108 |
| 100 | 10K | 0.00432 |
| 150 | 22.5K | 0.01013 |
| 200 | 40K | 0.01975 |
| 250 | 62.5K | 0.03225 |
| 300 | 90K | 0.04617 |

Linear fit: t = 5.23e-7·n − 8.7e-4 (**R² = 0.9992**); the power-law fit does not hold (it becomes linear + constant for n≥2.5K).

### Full-scale anchors (h6_anchors.csv, 1500 sweeps measured)
| S | n | total time (s) | per-sweep (ms) | mode ARI |
|---|---|---|---|---|
| 500 | 250K | **138.66** | 92.4 | 0.9936 |
| 750 | 562.5K | **360.99** | 240.7 | 0.9937 |
| 1000 | 1M | **674.21** | 449.5 | 0.9934 |

### Old extrapolation vs measured
| Scale | Old extrapolation (h2) | Measured (h6) | Deviation |
|---|---|---|---|
| 250K | 204 s | 138.7 s | old value **32% too high** |
| 1M | 817 s | 674.2 s | old value **17% too high** |

Cause: the old fit used only the two small points 2.5K/10K, and the fixed overhead of the small grid inflated the slope, so the extrapolation systematically overestimated the per-sweep cost.

### Corrected speedups (vs VI-EM time from h2d)
| Scale | Gibbs measured | VI-EM single run (median) | speedup | VI-EM 3-start (mean) | speedup |
|---|---|---|---|---|---|
| 250K | 138.66 s | 0.69 s | **201×** | 2.12 s | **65×** |
| 562.5K | 360.99 s | 2.34 s | **154×** | 22.49 s | **16×** |
| 1M | 674.21 s | 3.98 s | **169×** | 13.74 s | **49×** |

(The 3-start mean at 562.5K is inflated by two runs for instance 11 that hit the cap; using the median 8.99 s it is 40×.)

## Conclusion
1. **kt1 (≥10× speedup gate) passes in full under the new protocol**: 16–65× under the robust policy, 154–201× for single runs.
2. All speedup numbers in the paper are changed to the **measured protocol**, with the policy explicitly labeled (single run vs 3-start); "76–145× (extrapolated)" is void.
3. **The Gibbs mode ARI (0.9934–0.9937) is fully on par with VI-EM** → further supporting the core narrative: "the gap is not in inference quality but in cost; the infeasibility of exact/sampling inference at scale is an economic problem".
4. Fairness note (to be written into the paper): Gibbs is given oracle emission parameters (free) whereas VI-EM estimates parameters in full—the comparison favors Gibbs; this implementation is single-threaded numpy, not BayesSpace's C++ MCMC, so cross-implementation comparison requires caution.
5. The per-sweep cost model (R²=0.9992) can be extrapolated to the expected cost at larger scales, as a scalability argument for the paper.

## Files
- code/run.py (includes smoke mode); results/h6_sweepcost.csv, h6_anchors.csv, h6_meta.json, h6_speedup_summary.csv
