"""Patch GraphST to accept sparse neighbourhood tensors.

Author : Tao Zhu <zhutao@tjh.tjmu.edu.cn>
Created: 2026-10-02
Updated: 2026-10-20

Inputs
------
Installed GraphST package (BASE path from configuration).
Configuration: config/defaults.yaml, loaded through hmrf.config.

Outputs
-------
Patched package files.

Usage
-----
    python 06_patch_graphst_sparse.py [options]
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

import shutil, re, sys

BASE = CFG["experiments"]["h10"]["graphst_dir"]
if not BASE:
    raise SystemExit("set HMRF_GRAPHST_DIR to the installed GraphST package")
MARK = "SPARSE-PATCH HMRF-bench"


def main():
    for f in ("GraphST.py", "model.py", "preprocess.py"):
        src = open(f"{BASE}/{f}", encoding="utf-8").read()
        if MARK in src:
            print(f"{f}: already patched, skip")
        else:
            shutil.copy(f"{BASE}/{f}", f"{BASE}/{f}.bak-hmrf")
            print(f"{f}: backup -> .bak-hmrf")

    # P1
    s = open(f"{BASE}/preprocess.py", encoding="utf-8").read()
    if MARK not in s:
        new1 = '''def construct_interaction_KNN(adata, n_neighbors=3):
    # [SPARSE-PATCH HMRF-bench] dense n*n impossible at single-cell scale;
    # identical edge semantics, scipy.sparse storage instead of dense ndarray.
    import scipy.sparse as _spx
    position = adata.obsm['spatial']
    n_spot = position.shape[0]
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(position)
    _, indices = nbrs.kneighbors(position)
    x = indices[:, 0].repeat(n_neighbors)
    y = indices[:, 1:].flatten()
    interaction = _spx.coo_matrix((np.ones(len(x), dtype=np.float32), (x, y)),
                                  shape=(n_spot, n_spot)).tocsr()
    adata.obsm['graph_neigh'] = interaction
    adj = interaction + interaction.T
    adj = (adj > 0).astype(np.float32)
    adata.obsm['adj'] = adj
    print('Graph constructed (sparse)!')
'''
        pat1 = re.compile(r"def construct_interaction_KNN\(adata, n_neighbors=3\):.*?print\('Graph constructed!'\)[ \t]*", re.DOTALL)
        s2, n1 = pat1.subn(new1, s, count=1)
        assert n1 == 1, "P1 anchor not found"
        open(f"{BASE}/preprocess.py", "w", encoding="utf-8").write(s2)
        print("P1 applied")

    # P2
    s = open(f"{BASE}/GraphST.py", encoding="utf-8").read()
    if MARK not in s:
        old2 = "        self.graph_neigh = torch.FloatTensor(self.adata.obsm['graph_neigh'].copy() + np.eye(self.adj.shape[0])).to(self.device)"
        new2 = """        _gn = self.adata.obsm['graph_neigh']
        if hasattr(_gn, 'tocsr'):  # [SPARSE-PATCH HMRF-bench] avoid dense n*n expansion
            import scipy.sparse as _spx2
            _gn_ = (_gn + _spx2.eye(_gn.shape[0], format='csr')).tocoo()
            _idx = torch.LongTensor(np.vstack([_gn_.row, _gn_.col]))
            _val = torch.FloatTensor(_gn_.data)
            self.graph_neigh = torch.sparse_coo_tensor(_idx, _val, _gn_.shape, dtype=torch.float32).to(self.device)
        else:
            self.graph_neigh = torch.FloatTensor(_gn.copy() + np.eye(self.adj.shape[0])).to(self.device)"""
        assert old2 in s, "P2 anchor not found"
        open(f"{BASE}/GraphST.py", "w", encoding="utf-8").write(s.replace(old2, new2, 1))
        print("P2 applied")

    # P3
    s = open(f"{BASE}/model.py", encoding="utf-8").read()
    if MARK not in s:
        pat3 = re.compile(r"(def forward\(self, emb, mask=None\):\s*)vsum = torch\.mm\(mask, emb\).*?return F\.normalize\(global_emb, p=2, dim=1\)", re.DOTALL)
        new3 = """def forward(self, emb, mask=None):
        if getattr(mask, 'is_sparse', False):  # [SPARSE-PATCH HMRF-bench]
            vsum = torch.sparse.mm(mask, emb)
            row_sum = torch.sparse.sum(mask, 1).to_dense().unsqueeze(1)
            global_emb = vsum / row_sum
        else:
            vsum = torch.mm(mask, emb)
            row_sum = torch.sum(mask, 1)
            row_sum = row_sum.expand((vsum.shape[1], row_sum.shape[0])).T
            global_emb = vsum / row_sum
        return F.normalize(global_emb, p=2, dim=1)"""
        s2, n3 = pat3.subn(new3, s, count=1)
        assert n3 == 1, "P3 anchor not found"
        open(f"{BASE}/model.py", "w", encoding="utf-8").write(s2)
        print("P3 applied")

    import py_compile
    for f in ("GraphST.py", "model.py", "preprocess.py"):
        py_compile.compile(f"{BASE}/{f}", doraise=True)
    print("all patches in place, compile ok")


if __name__ == "__main__":
    main()
