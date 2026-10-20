# H1 Analysis: Mean-Field Variational Inference Is Faithful to the Gibbs Posterior ✅

## Results (30x30 grid, K=5, d=20 Gaussian emission, oracle parameters)
- 9 main settings: |ARI(VI, truth) − ARI(Gibbs, truth)| ≤ 0.008; ARI(VI mode, Gibbs mode) ∈ [0.9625, 0.9830].
- beta-mismatch stress test (true 0.5 → model 1.5): both methods degrade in tandem (0.373 vs 0.385), and VI still tracks Gibbs (ARI 0.923).
- Speed: VI 0–3 ms (6–41 fixed-point iterations); Gibbs 0.42 s (1500 sweeps × 2 half-sweeps) → ~200x at the current scale.
- Decision criterion (≥2/3 settings with |Δ|≤0.02 and ARI(vi,gibbs)≥0.90): **all pass**.

## Limitations and next steps
1. Absolute ARI is only ~0.55 (the task is intrinsically hard under σ=0.8 noise) → switch the scale experiments to a more identifiable setting (σ=0.5 / mu_scale=2.0) so that an "accuracy collapse" would be detectable.
2. Emission parameters are currently oracle; real scenarios require joint estimation of mu via EM → next round h1b: VI-EM joint vs Gibbs.
3. Gibbs is fast on small grids, but O(iterations × sites × K) in pure numpy is infeasible on large grids → the scaling curve is carried by VI, with Gibbs serving only as a small-sample reference.
4. Count emissions (NB/ZINB) and sparse dropout are the real challenges of million-scale platforms → the h2 main line.
