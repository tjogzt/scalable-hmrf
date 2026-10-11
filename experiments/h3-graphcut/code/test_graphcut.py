# --- portable project root (no hardcoded user paths) ---
import os as _os, pathlib as _pl
def _find_root(_start):
    _p = _pl.Path(_start).resolve()
    for _anc in [_p] + list(_p.parents):
        if (_anc / "src" / "hmrf_solvers.py").exists():
            return str(_anc)
    raise RuntimeError("project root not found; run inside the repository")
_ROOT = _os.environ.get("HMRF_ROOT") or _find_root(__file__)
# ------------------------------------------------------
"""图割构造穷举验证：把 alpha_step 的割结果与 2^9 暴力枚举对照。"""
import sys, itertools
sys.path.insert(0, _os.path.join(_ROOT, "src"))
import numpy as np
from graphcut_hmrf import alpha_step, alpha_expansion, potts_energy

rng = np.random.default_rng(0)
def brute_E(b, unary, lab, H, W, beta, alpha):
    b = np.asarray(b).reshape(H, W)
    new = lab.copy()
    new[b == 1] = alpha
    return potts_energy(new, unary, beta, H, W)

fails = 0
# 测试 1: 2x2 小网格（4 节点，16 种赋值），随机 labels/alpha/unary/beta
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
print("2x2 穷举测试: ", "PASS" if fails == 0 else f"{fails} FAILS")

# 测试 2: 3x3 网格（512 种赋值）小样本
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
print("3x3 穷举测试: ", "PASS" if fails3 == 0 else f"{fails3} FAILS")

# 测试 3: 2x2 多标签全枚举：alpha_expansion 是否找到全局最优
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
print(f"2x2 多标签全局最优: {ok3}/{tot}（α-expansion 不保证全局最优，但能量必须单调下降）")
print("ALL TESTS DONE")
