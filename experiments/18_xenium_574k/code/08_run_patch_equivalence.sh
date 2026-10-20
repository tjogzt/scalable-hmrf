#!/bin/bash
# Numerically verify that the sparse GraphST patch matches the dense reference.
#
# Author : Tao Zhu <zhutao@tjh.tjmu.edu.cn>
# Created: 2026-10-02
# Updated: 2026-10-20
#
# Inputs
#   HMRF_XENIUM_DIR  Xenium output directory
#   GRAPHST_DIR      installed GraphST package directory (required)
#   LOG_DIR          optional log directory (default: logs/e8)
#
# Outputs
#   $LOG_DIR/*.log and experiments/18_xenium_574k/results/patch_equiv_10k/
#
# Usage
#   bash 08_run_patch_equivalence.sh
set -u
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
S="${LOG_DIR:-$ROOT/logs/e8}"
mkdir -p "$S"
BASE="${GRAPHST_DIR:?set GRAPHST_DIR to the installed GraphST package}"
export HMRF_XENIUM_DIR="${HMRF_XENIUM_DIR:?set HMRF_XENIUM_DIR to the Xenium outs directory}"
P="${PYTHON:-python3}"
cd "$ROOT" || exit 1

echo "== leg 1: sparse (patched) =="
"$P" experiments/18_xenium_574k/code/07_test_patch_equivalence.py sparse > "$S/e8_sparse.log" 2>&1
echo "sparse rc=$?"; tail -3 "$S/e8_sparse.log"

echo "== swap in originals =="
for f in GraphST.py model.py preprocess.py; do
  if [ ! -f "$BASE/$f.bak-hmrf" ]; then echo "MISSING BACKUP: $f"; exit 1; fi
  cp "$BASE/$f.bak-hmrf" "$BASE/$f"
done

echo "== leg 2: dense (original) =="
"$P" experiments/18_xenium_574k/code/07_test_patch_equivalence.py dense > "$S/e8_dense.log" 2>&1
echo "dense rc=$?"; tail -3 "$S/e8_dense.log"

echo "== re-apply sparse patch =="
"$P" experiments/18_xenium_574k/code/06_patch_graphst_sparse.py > "$S/e8_repatch.log" 2>&1
echo "repatch rc=$?"; tail -2 "$S/e8_repatch.log"

echo "== compare =="
"$P" experiments/18_xenium_574k/code/07_test_patch_equivalence.py compare > "$S/e8_compare.log" 2>&1
echo "compare rc=$?"; cat "$S/e8_compare.log"
echo "E8 DONE"
