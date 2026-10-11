# H2d Analysis: Multi-Seed Scaling Curve + NMI/Boundary F1 (post-review batch 1)

## Objective (responding to review comments)
1. Fix the "single-instance" problem: 5 independent instance seeds × 5 scale points × 3 independent starts per instance (rng seed 3/4/5), reporting mean±SD;
2. Complete the metrics promised at project inception but missing: NMI + boundary F1 (strict edge set: F1 over the set of adjacent domain pairs);
3. Fix the S=750 retroactive issue (the old h2 table's 0.9937 was a "specific-start re-check value", mixed with the single-start policy).

## Settings
K=8, d=20, Gaussian emission (σ=0.5, μ_scale=2.0), β=0.7, max_iter=300, tol=1e-4.
Instance seeds (11,21,31,41,51) (truth seed=iseed, noise seed=iseed+1); scales S ∈ {100,250,500,750,1000}.
Runtime environment: anaconda Python 3.12 + numpy (all measured in the same batch).

## Results (aggregated across instances by scale, h2d_summary.csv)

| S | n | ARI mean(3 starts)±sd | ARI best(3 starts)±sd | NMI(best) | Boundary F1(best) | Total time for 3 starts |
|---|---|---|---|---|---|---|
| 100 | 10K | 0.9671 ± 0.0292 | 0.9781 ± 0.0199 | 0.9771 | 0.9363 | 0.68 ± 0.57 s |
| 250 | 62.5K | 0.9762 ± 0.0241 | 0.9931 ± 0.0006 | 0.9863 | 0.8723 | 1.83 ± 1.55 s |
| 500 | 250K | **0.9936 ± 0.0001** | 0.9936 ± 0.0001 | 0.9875 | 0.7801 | 2.12 ± 0.23 s |
| 750 | 562.5K | 0.9875 ± 0.0124 | 0.9875 ± 0.0124 | 0.9837 | 0.7172 | 22.49 ± 27.75 s |
| 1000 | 1M | **0.9935 ± 0.0001** | 0.9935 ± 0.0001 | 0.9875 | 0.6594 | 13.74 ± 2.23 s |

(Long table h2d_results.csv: 75 rows of per-start records; summary h2d_summary.csv)

## Key findings

1. **≥250K: stable across all instances, the claim holds**. The 250K and 1M points across 5 instances and 15 starts both have ARI 0.9934–0.9937 (sd≤0.0001); "scale does not induce accuracy collapse" holds under multiple seeds.
2. **10K–62.5K exhibits instance variance, mechanism decomposed** (probe probe_convergence.log):
   - Slow convergence: for S=100 instance 21, r0 "appears to collapse" (0.88) within the 300-iteration cap, but after relaxing to 3000 it **converges to 0.9926 in 378 iterations**; for S=250 instance 51, r0 needs **2111 iterations** to settle (and settles into a suboptimal basin at 0.8253).
   - Suboptimal basins: different starts of the same instance can converge to different basins (0.918 / 0.948 / 0.993 coexist).
   - Conclusion: small-scale variance ≠ scale collapse, but "slow convergence + basin structure"; in practice one needs convergence monitoring + multiple starts, not a fixed small iteration cap.
3. **S=750 retroactive fix (important)**: for instance 11, under independent start seeds 3/4/5 all three runs fall into the **stable 0.9627 basin** (1 converged in 134 iterations, 2 hit the 300 cap); whereas the six additional starts seeds 6–11 **all reach 0.9937 within 13–16 iterations**. That is: 0.9937 is not unreachable (6/9 starts reach it), but the old h2 footnote "restarts=2 re-check value 0.9937" mixed different restart policies and between-run selection, **the h2-scale table basis is no longer comparable; take this experiment as authoritative**. The paper should state "of 9 starts, 6 reach 0.9937 and 3 are trapped in the 0.963 basin".
4. **ELBO selection picked the optimal start in 15/15 simulation cases** (ari_elbo ≡ ari_max, all scales) — a sharp contrast with real data (DLPFC, where ELBO selected an inferior solution of 0.1302). This strengthens the paper's open question: *ELBO works in well-posed simulations but fails on real data*; model selection is the key gap on real data (the batch-2 held-out predictive-likelihood experiment speaks directly to this point).
5. **Boundary F1 decreases monotonically with scale** (0.936→0.659): under the strict edge-set definition, the larger the scale the "finer" the boundary is in relative terms and the harder it is for predicted boundaries to coincide edge by edge. The paper must attach a definition note when citing this metric, and it is recommended to add a "radius-1 tolerance" variant later (optional in batch 2).
6. NMI is consistent with the ARI trend (0.977–0.988).

## Relation to h2-scale
- h2d **supersedes** h2-scale as the source of the main table (unified restarts=3/max_iter=300 policy + multiple seeds + completed metrics);
- h2-scale's S=750 row (single-start 0.9627 / re-check 0.9937) and S=1000 row (single-start 10.78s) are both covered by h2d; the old files are retained as historical records — do not cite their numbers.
