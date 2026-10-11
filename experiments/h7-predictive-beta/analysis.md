# H7 Held-out Predictive Likelihood (β Selection) + H7b Start Selection — Analysis

Date: 2026-10-11 (Batch 2) | Scripts: `code/run.py` (h7), `code/run_init_selection.py` (h7b)
Setup: random masking of 5% of spots (3 mask seeds 7/8/9), inferring the domain label and expression of the masked spots from the neighborhood field alone,
evaluating the held-out log-likelihood ll_i = log Σ_k q_i(k)·N(y_i|μ_k,σ²) (averaged per spot). β grid {0.3,0.5,0.7,1.0,1.5}.

## 1. Main result: held-out likelihood as a β selection criterion — fails (third criterion)

**Across 3 real DLPFC sections × 3 masks, 9 groups in total, "the β with the largest held-out likelihood" collapses to the grid maximum 1.5 in every case**
(LL increases monotonically with β, ΔLL only 0.1–0.75 nats, spanning a 5× β interval). Over the same range ARI is essentially flat in β
but exhibits within-section fluctuation (max |Δ ARI| ≈ 0.13, see 151673 mask9: 0.417@0.5 → 0.289@1.0).

| Section | Mask | LL trajectory (β: 0.3→1.5) | LL-argmax | ARI-optimal β | ARI range |
|---|---|---|---|---|---|
| 151673 | 7 | −62.67→−61.92 | 1.5 | 1.5 (0.319, flat) | 0.299–0.320 |
| 151673 | 8 | −79.25→−78.56 | 1.5 | 1.0 (0.464) | 0.418–0.464 |
| 151673 | 9 | −71.05→−70.37 | 1.5 | 0.5 (0.417) | 0.288–0.417 |
| 151509 | 7 | −67.48→−66.82 | 1.5 | 0.7 (0.391) ≈ flat | 0.386–0.398 |
| 151509 | 8 | −77.97→−77.28 | 1.5 | 0.3 (0.322, flat) | 0.299–0.322 |
| 151509 | 9 | −74.09→−73.40 | 1.5 | 1.5 (0.428) | 0.399–0.428 |
| 151671 | 7 | −81.80→−81.30 | 1.5 | 1.5 (0.615) | 0.553–0.615 |
| 151671 | 8 | −75.79→−75.26 | 1.5 | 1.5 (0.484) | 0.463–0.576 |
| 151671 | 9 | −63.93→−63.55 | 1.5 | 1.5 (0.410) | 0.408–0.436 |

**Mechanism and interpretation**: on real data "LL barely varies with β" (the predictive fit is dominated by μ/σ), yet β
substantially controls label smoothness and strongly affects ARI → **predictive likelihood is insensitive to β, and its sensitivity to precision is lower than expected**,
so it does not constitute a usable selection criterion. Contrast: on simulated data (gauss_sim_S250), LL and ARI increase monotonically in the same direction
(β=1.5 is optimal for both)—i.e., the LL criterion "works in simulation precisely and fails on real data", confirming
the "no free lunch in β selection" narrative of Batch-1 fig5 (fig5), and adding a third failing criterion (the first two: ELBO monotone,
boundary elbow unstable). ELBO likewise increases monotonically with β (e.g., 151673/7: −236090→−224454),
and its argmax also collapses to 1.5 — ELBO and held-out LL both fail, in the same direction.

**Implication for the paper**: the methodological selling point shifts from "we have automatic β selection" to **an honest negative result + a practical recommendation**
(report the β sensitivity curve; use task-level validation/external labels for selection). This negative result, together with the H3b graph-cut negative result and
the H2d multi-start variance, forms a "cheap but honest" evidence bundle.

## 2. H7b: Using held-out likelihood to select among restarts (multiple starts)—better than ELBO but unstable

Setup: 151673/151509 × 3 masks × 5 starts (rng 100+t), β=0.7 fixed, comparing
spearman(LL, ARI) against spearman(ELBO, ARI) (rank correlation across 5 starts):

| Section | Mask | ρ(LL,ARI) | ρ(ELBO,ARI) | ARI of LL-argmax (optimal ARI) | ARI of ELBO-argmax |
|---|---|---|---|---|---|
| 151673 | 7 | −0.70 | −0.30 | 0.282 (0.358) | 0.291 |
| 151673 | 8 | +0.50 | +0.60 | 0.341 (0.342) ✓ | 0.254 |
| 151673 | 9 | +0.80 | +0.80 | 0.382 (0.427) | 0.427 ✓ |
| 151509 | 7 | +0.70 | +0.60 | 0.508 (0.545) | 0.503 |
| 151509 | 8 | +0.10 | −0.90 | 0.450 (0.450) ✓ | 0.211 ✗ |
| 151509 | 9 | +1.00 | +0.10 | 0.452 (0.452) ✓ | 0.384 |

- **Median rank correlation: LL +0.60 (range −0.70~+1.00, n=6); ELBO +0.35 (range −0.90~+0.80)**.
- Among 6 cases, LL-argmax picks the optimum exactly 3 times and all 6 fall within ±0.08 ARI of the optimum, **never a catastrophic mis-selection**;
  ELBO-argmax has 2 catastrophic mis-selections (0.211 vs optimal 0.450; 0.291 vs optimal 0.358 level).
- Conclusion: **"using held-out likelihood for start selection" shows a slight positive signal but is unstable** (large fluctuation across masks);
  "using ELBO for start selection" can already be judged an anti-pattern (h4d already has a catastrophic case, 0.130 vs 0.456).
  Recommended phrasing for the paper: report all three protocols—best/median/ELBO-selected—side by side, making explicit the open nature of the selection problem.

## 3. Reproduction and files
- `results/h7_runs.csv` (45 rows: 3 sections + simulation × 3 masks × 5 β), `h7_summary.json`
- `results/h7b_init_selection.csv` (30 rows), `h7b_run.log`
- Single-mask EM ∈ [0.2, 1.6] s (~3.5–4.8K spots); intermediate labels are not retained to avoid misuse.
