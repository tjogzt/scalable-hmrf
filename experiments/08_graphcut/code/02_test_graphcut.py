"""Unit tests for the alpha-expansion construction.

Author : Tao Zhu <zhutao@tjh.tjmu.edu.cn>
Created: 2026-10-02
Updated: 2026-10-20

Inputs
------
Synthetic toy graphs generated in the test.
Configuration: config/defaults.yaml, loaded through hmrf.config.

Outputs
-------
Console assertions and a CSV summary.

Usage
-----
    python 02_test_graphcut.py [options]
"""

import os
import sys
from pathlib import Path

for _parent in Path(__file__).resolve().parents:
    if (_parent / "config" / "defaults.yaml").exists():
        REPO_ROOT = _parent
        break
else:  # pragma: no cover - repository layout guard
    raise RuntimeError("repository root not found: config/defaults.yaml missing")

sys.path.insert(0, str(REPO_ROOT / "src"))

from hmrf.config import load_config  # noqa: E402
from hmrf.paths import data_dir, results_dir  # noqa: E402

CFG = load_config()
_ROOT = str(REPO_ROOT)  # backwards-compatible alias used by legacy code paths

import sys, itertools
sys.path.insert(0, os.path.join(_ROOT, "src"))
import numpy as np
from hmrf.graphcut import alpha_step, alpha_expansion, potts_energy

rng = np.random.default_rng(0)
def brute_E(b, unary, lab, H, W, beta, alpha):
    b = np.asarray(b).reshape(H, W)
    new = lab.copy()
    new[b == 1] = alpha
    return potts_energy(new, unary, beta, H, W)

fails = 0
# Test 1: 2x2 small grid (4 nodes, 16 assignments), random labels/alpha/unary/beta
for trial in range(300):
    H, W, K = 2, 2, 4
    unary = rng.standard_normal((H, W, K))
    lab = rng.integers(0, K, size=(H, W))
    alpha = int(rng.integers(0, K))
    beta = float(rng.uniform(0.1, 3.0))
    got = alpha_step(unary, lab, H, W, beta, alpha)
    best_E, best_b = None, None
    for bits in itertools.product([0, 1], repeat=4):
        E = brute_E(bits, unary, lab, H, W, beta, alpha)
        if best_E is None or E < best_E:
            best_E, best_b = E, bits
    gE = potts_energy(got, unary, beta, H, W)
    if abs(gE - best_E) > 1e-6:
        fails += 1
        if fails <= 3:
            print("FAIL 2x2 trial", trial, "graph", gE, "brute", best_E, "lab", lab.ravel(), "alpha", alpha)
print("2x2 exhaustive test: ", "PASS" if fails == 0 else f"{fails} FAILS")

# Test 2: 3x3 grid (512 assignments), small sample
fails3 = 0
for trial in range(30):
    H, W, K = 3, 3, 4
    unary = rng.standard_normal((H, W, K))
    lab = rng.integers(0, K, size=(H, W))
    alpha = int(rng.integers(0, K))
    beta = float(rng.uniform(0.1, 2.5))
    got = alpha_step(unary, lab, H, W, beta, alpha)
    best_E = None
    for bits in itertools.product([0, 1], repeat=9):
        E = brute_E(bits, unary, lab, H, W, beta, alpha)
        best_E = E if best_E is None else min(best_E, E)
    gE = potts_energy(got, unary, beta, H, W)
    if abs(gE - best_E) > 1e-6:
        fails3 += 1
        if fails3 <= 3:
            print("FAIL 3x3 trial", trial, "graph", gE, "brute", best_E)
print("3x3 exhaustive test: ", "PASS" if fails3 == 0 else f"{fails3} FAILS")

# Test 3: 2x2 multi-label full enumeration: does alpha_expansion find the global optimum
ok3 = 0; tot = 0
for trial in range(50):
    H, W, K = 2, 2, 3
    unary = rng.standard_normal((H, W, K))
    beta = float(rng.uniform(0.1, 2.0))
    got = alpha_expansion(unary, beta, seed=trial)
    best = None
    for comb in itertools.product(range(K), repeat=4):
        lab = np.asarray(comb).reshape(H, W)
        best = potts_energy(lab, unary, beta, H, W) if best is None else min(best, potts_energy(lab, unary, beta, H, W))
    tot += 1
    if abs(potts_energy(got, unary, beta, H, W) - best) < 1e-6:
        ok3 += 1
print(f"2x2 multi-label global optimum: {ok3}/{tot} (alpha-expansion does not guarantee the global optimum, but the energy must decrease monotonically)")
print("ALL TESTS DONE")
