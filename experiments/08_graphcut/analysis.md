# H3 Analysis: α-expansion Graph Cut vs Mean-Field VI

## Correctness (exhaustive verification)
- 2×2 single-step 300 cases, 3×3 single-step 30 cases: cut result energy = brute-force optimum (all PASS).
- 2×2 multi-label full enumeration: 49/50 reach the global optimum (1 case is the intrinsic locality of α-expansion, consistent with theory).
- Two bugs fixed during development: ① pymaxflow get_segment semantics inverted (0=source side); ② potts_energy double-counting undirected edges.

## Same-instance comparison (VI-EM unary terms, K=8, β=0.7)
| S | n | ARI(MF) | ARI(GC) | Energy MF | Energy GC | t_MF | t_GC |
|---|---|---|---|---|---|---|---|
| 100 | 10K | 0.9942 | 0.9942 | 145758.0 | 145758.0 | 5ms | 0.25s |
| 250 | 62.5K | 0.9933 | 0.9934 | 908688.9 | 908689.0 | 38ms | 1.49s |

## Findings
1. On easily separable data the two inference kernels give nearly identical solutions (identical ARI, energy difference <1e-3 relative) → exact graph cut yields no benefit, and MF is 40-50x faster.
2. Graph-cut energy is slightly lower (closer to the global optimum), consistent with theory; but a difference can only emerge in the "high β / weak unary terms" stress region → h3b.
3. Engineering reality: pure-Python graph construction is the bottleneck (S=250 needs 1.49s; at the million scale it is projected to be >30s with doubled memory); C++ graph construction (pygco) is the prerequisite for scaling the graph-cut path — directly relevant to Direction Two's "engineering" proposition.
4. Conclusion (current evidence): MF-VI remains the main path; graph cut serves as a backup engine for "exact solutions in difficult regions", pending the h3b decision.
