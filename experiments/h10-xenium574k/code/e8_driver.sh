#!/bin/bash
# E8 driver: run sparse leg -> swap in originals -> dense leg -> re-apply patch -> compare.
set -u
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
S="${LOG_DIR:-$ROOT/logs/e8}"
mkdir -p "$S"
BASE="${GRAPHST_DIR:-/opt/anaconda3/lib/python3.12/site-packages/GraphST}"
export HMRF_XENIUM_DIR="${HMRF_XENIUM_DIR:?set HMRF_XENIUM_DIR to the Xenium outs directory}"
P="${PYTHON:-python3}"
cd "$ROOT" || exit 1

echo "== leg 1: sparse (patched) =="
"$P" experiments/h10-xenium574k/code/patch_equivalence_10k.py sparse > "$S/e8_sparse.log" 2>&1
echo "sparse rc=$?"; tail -3 "$S/e8_sparse.log"

echo "== swap in originals =="
for f in GraphST.py model.py preprocess.py; do
  if [ ! -f "$BASE/$f.bak-hmrf" ]; then echo "MISSING BACKUP: $f"; exit 1; fi
  cp "$BASE/$f.bak-hmrf" "$BASE/$f"
done

echo "== leg 2: dense (original) =="
"$P" experiments/h10-xenium574k/code/patch_equivalence_10k.py dense > "$S/e8_dense.log" 2>&1
echo "dense rc=$?"; tail -3 "$S/e8_dense.log"

echo "== re-apply sparse patch =="
"$P" experiments/h10-xenium574k/code/patch_graphst_sparse.py > "$S/e8_repatch.log" 2>&1
echo "repatch rc=$?"; tail -2 "$S/e8_repatch.log"

echo "== compare =="
"$P" experiments/h10-xenium574k/code/patch_equivalence_10k.py compare > "$S/e8_compare.log" 2>&1
echo "compare rc=$?"; cat "$S/e8_compare.log"
echo "E8 DONE"
