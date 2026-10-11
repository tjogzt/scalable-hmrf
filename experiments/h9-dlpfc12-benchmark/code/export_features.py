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
"""导出 12 切片的本方法特征管线产物（PC50 / 邻接边 / 真值），供 SC.MEB 同管线基线使用。"""
import sys, os
import numpy as np
import pandas as pd

ROOT = _ROOT
sys.path.insert(0, os.path.join(ROOT, "experiments/h9-dlpfc12-benchmark/code"))
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

OUT = os.path.join(ROOT, "experiments/h9-dlpfc12-benchmark/results/features")
os.makedirs(OUT, exist_ok=True)
ALL = ["151507", "151508", "151509", "151510", "151669", "151670",
       "151671", "151672", "151673", "151674", "151675", "151676"]

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
