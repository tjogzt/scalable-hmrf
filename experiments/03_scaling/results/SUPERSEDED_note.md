# ⚠ The numbers in this directory (h2-scale) have been superseded by h2d — do not cite them any longer

Retroactive issues found during post-review batch 1 (2026-10-11) and their handling:

1. **S=750 row (core issue)**: In this directory's h2_results.csv, S=750 is a single-start run value (ARI 0.9627, hitting the cap at 150 iterations);
   at the time the paper footnote claimed "restarts=2 re-check value = 0.9937 / 39.8s", but that value mixed restart policies and was never written back to the CSV.
   The batch-1 h2d experiment (5 instances × 3 independent starts) established that this instance, under independent start seeds 3/4/5, **fell into the stable 0.9627 basin on all three runs**,
   whereas the six additional starts (seeds 6–11) all reached 0.9937 within 13–16 iterations. The old "re-check value 0.9937" = reachable from a specific start, insufficient to support "the number in the table",
   and under the 3-start policy this instance should honestly be reported as 0.963 (of 9 starts, 6 reach 0.9937 and 3 are trapped).

2. **Time-column basis**: the 10.78s for S=1000 is an old dsh runtime measurement; h2d re-measured it in a unified environment (anaconda) as
   13.74s total for 3 starts (median single run 3.98s).

3. **Handling**: this directory is retained as a historical record; **the paper's main table, speedup numbers, and Figure 2 all switch to**:
   - experiments/h2d-multiseed-metrics/ (multi-seed ARI/NMI/boundary F1/time)
   - experiments/h6-gibbs-measured/ (measured Gibbs anchors and corrected speedups)
   See the two analysis.md files.
