# H2b Analysis: Re-designed NB Simulation Parameters

## Setting changes
- K=8, d=10→20, disp=1→3, domain factors [0.8,1.1,1.5,2.0,2.7,3.7,5.0,6.8] (~0.3 log spacing), zero fraction 0.17-0.21.

## Results
| S | n | Zero fraction | ARI(oracle ceiling) | ARI(VI-EM) | VI-EM time |
|---|---|---|---|---|---|
| 250 | 62.5K | 0.214 | 0.857 | 0.661 | 6.8s |
| 1000 | 1M | 0.168 | 0.878 | 0.757 | 94.6s |

## Judgment
- Task identifiability improved from 0.466 → 0.86-0.88 (a substantial improvement, but still <0.9 target).
- The VI-EM vs oracle gap of 0.12-0.20 comes from initialization/convergence (not converged within 300 iterations; MF converges slowly under weak unary terms).
- Next step (h2c): log1p-standardized initialization + multiple starts + relaxed tol + beta grid; target VI-EM/oracle gap <0.05.
- The key argument already holds: the ARI ceiling of the NB scenario is mainly determined by the information content of the data (oracle 0.86), echoing the attribution decomposition of kt2.
