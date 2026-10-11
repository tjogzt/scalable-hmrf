# H1b Analysis: Joint VI-EM Estimation = Oracle ✅

## Results (40x40, K=5, d=20, σ=0.5, mu_scale=2.0, β=0.8, 3 seeds)
| seed | ARI(VI-EM) | ARI(VI-oracle) | ARI(Gibbs-oracle) | ARI(VI-EM, Gibbs) | σ̂ |
|---|---|---|---|---|---|
| 0 | 0.9812 | 0.9812 | 0.9812 | 0.9967 | 0.503 |
| 1 | 0.9860 | 0.9843 | 0.9843 | 0.9952 | 0.499 |
| 2 | 0.9783 | 0.9785 | 0.9785 | 0.9966 | 0.506 |

- VI-EM agrees with the two oracle variants (difference < 0.003); σ recovery error < 1.5%; converges in 12 iterations (0.01s).
- Conclusion: after removing the oracle assumption, VI-EM remains faithful — the second gate for Direction Two's "VI can replace Gibbs" passes.
