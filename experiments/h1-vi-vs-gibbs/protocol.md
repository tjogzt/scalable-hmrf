# Experiment H1: Mean-Field Variational Inference ≈ Gibbs Posterior (Correctness Gate)

## Objective
Before Direction Two replaces Gibbs with VI/graph cuts, first prove that the mean-field variational approximation to the posterior mode of the Potts HMRF is faithful.
If the VI surrogate itself is distorted, every subsequent speedup claim loses its foundation.

## Design
- Simulation: 30x30 grid, K=5 domains (true labels sampled from a low-temperature Potts model via Checkerboard-Gibbs, with natural boundaries);
  Gaussian emission with d=20 genes, mu_k separation 1.5, noise σ=0.8.
- Comparison: ① full Bayesian Gibbs (oracle emission parameters, posterior-mode labels) ② mean-field VI (same unary terms, fixed-point iteration).
- Sweep: model beta ∈ {0.2, 0.5, 1.0} (= true beta), plus a beta-mismatch test {0.5 true vs 1.5 model}.
- Metrics: ARI(method, truth), ARI(VI mode, Gibbs mode), wall-clock time.
- Seeds: 3 (variation from noise/truth sampling).

## Prediction (mechanism judgment)
- The two methods have similar ARI(·, truth); ARI(VI, Gibbs) >= 0.95 (at beta=1.0); VI is 100x+ faster.

## Decision (success criteria)
- In >=2 of the 3 beta settings: |ARI(VI,truth) - ARI(Gibbs,truth)| <= 0.02, and ARI(VI,Gibbs) >= 0.90.
- If not satisfied → VI cannot directly replace Gibbs (need to investigate MF degradation at high beta); recording a negative result still counts as progress.

## Output
- results/h1_results.csv (each row: seed, beta, method, ari_truth, ari_vi_vs_gibbs, wall_s, iters)
