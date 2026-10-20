# H3b Analysis: Stress-Region Graph Cut vs Mean Field (S=100, K=8)

| sigma | mu_scale | beta | ARI(MF) | ARI(GC) | Energy MF | Energy GC | t_MF | t_GC |
|---|---|---|---|---|---|---|---|---|
| 0.7 | 1.3 | 1.5 | 0.9751 | 0.9783 | 213832 | 213823 | 59ms | 0.25s |
| 0.7 | 1.3 | 2.5 | 0.9026 | 0.8998 | 216810 | 216758 | 280ms | 0.24s |
| 1.0 | 1.3 | 1.5 | 0.9536 | 0.9555 | 285140 | 285115 | 102ms | 0.33s |
| 1.0 | 1.3 | 2.5 | 0.7957 | 0.7964 | 287632 | 287357 | 279ms | 0.34s |
| 0.7 | 2.0 | 1.5 | 0.9924 | 0.9927 | 213889 | 213888 | 13ms | 0.25s |
| 0.7 | 2.0 | 2.5 | 0.9080 | 0.9133 | 219262 | 219044 | 279ms | 0.24s |

## Conclusion (closing of the h3 series)
1. Graph cut achieves lower energy in all stress settings (exactness verified ✓), but the ARI difference is |Δ|≤0.006, with inconsistent direction
   (at β=2.5, σ=0.7 the graph-cut energy is lower but the ARI is instead lower by 0.003).
2. Mechanism: when the model and the truth are mismatched, "more exact inference" ≠ "more accurate labels" — the error comes mainly from model specification (isotropic Potts, emission assumptions) and data noise.
3. Final judgment for Direction Two: **at the million scale, MF-VI is the only reasonable engine**; the exact graph-cut solution yields no ARI gain and cannot be scaled in pure Python.
   The graph-cut module is retained as a diagnostic tool (energy lower-bound reference).
4. Paper evidence chain: h3/h3b = "exact inference does not buy domain-detection accuracy at scale" (an honest negative result).
