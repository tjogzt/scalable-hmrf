"""Export the feature pipeline outputs for external baselines.

Author : Tao Zhu <zhutao@tjh.tjmu.edu.cn>
Created: 2026-10-02
Updated: 2026-10-20

Inputs
------
data/dlpfc12/<slice>/ matrices.
Configuration: config/defaults.yaml, loaded through hmrf.config.

Outputs
-------
results/features/ NPZ files.

Usage
-----
    python 14_export_features.py [options]
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

sys.path.insert(0, os.path.join(ROOT, "experiments/17_dlpfc12_benchmark/code"))
from run_ours import load_slice  # noqa


def feature_pipeline(X):
    lx = np.log1p(X / (X.sum(1, keepdims=True) + 1e-9) * 1e4)
    m = lx.mean(0); v = lx.var(0); use = m > 1e-3
    lm = np.log10(m[use]); lv = np.log10(np.maximum(v[use], 1e-12))
    coef = np.polyfit(lm, lv, 2); resid = lv - np.polyval(coef, lm)
    gidx = np.where(use)[0][np.argsort(resid)[::-1][:3000]]
    Y = lx[:, gidx].astype(np.float64)
    Yz = (Y - Y.mean(0, keepdims=True)) / (Y.std(0, keepdims=True) + 1e-6)
    U, S, Vt = np.linalg.svd(Yz, full_matrices=False)
    PC = U[:, :50] * S[:50]
    PC = PC - PC.mean(0, keepdims=True); PC = PC / (PC.std(0, keepdims=True) + 1e-6)
    return PC

OUT = str(results_dir("17_dlpfc12_benchmark") / "features")
os.makedirs(OUT, exist_ok=True)
ALL = CFG["datasets"]["dlpfc12"]["slices"]

for sid in ALL:
    X, rows, cols, truth = load_slice(sid)
    k = int(np.unique(truth).size)
    PC = feature_pipeline(X)
    del X
    np.savetxt(os.path.join(OUT, f"{sid}_pc50.csv"), PC, delimiter=",", fmt="%.5f")
    pos2i = {(r, c): i for i, (r, c) in enumerate(zip(rows, cols))}
    e = []
    for i, (r, c) in enumerate(zip(rows, cols)):
        for dr, dc in ((0, 2), (0, -2), (1, -1), (1, 1), (-1, -1), (-1, 1)):
            j = pos2i.get((r + dr, c + dc))
            if j is not None:
                e.append((i + 1, j + 1))  # 1-based for R
    np.savetxt(os.path.join(OUT, f"{sid}_edges.csv"), np.array(e, dtype=int), delimiter=",", fmt="%d")
    pd.DataFrame({"truth": truth}).to_csv(os.path.join(OUT, f"{sid}_truth.csv"), index=False)
    print(sid, "n=", len(truth), "k=", k, "edges=", len(e), flush=True)
print("EXPORT FEATURES DONE")
