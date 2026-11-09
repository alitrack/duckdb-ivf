#!/usr/bin/env bash
# repo-relative paths (env-overridable)
D="$(cd "$(dirname "$0")" && pwd)"
R="$(cd "$D/.." && pwd)"
CORPUS="${ZTG_CORPUS:-$D/data/ag_news_train.parquet}"
NPZ="${ZTG_EMB_NPZ:-$D/data/emb_bge_m3.npz}"
export ZTG_CORPUS="$CORPUS" ZTG_EMB_NPZ="$NPZ"
set -u
SO=$D/libml.so
OUT=$D/ml_onnx.duckdb_extension
# ivf loads with an all-zero 512-byte tail -> mirror that
python3 - "$SO" "$OUT" <<'PY'
import sys
so, out = sys.argv[1], sys.argv[2]
d = open(so,'rb').read()
open(out,'wb').write(d + b"\x00"*512)
print("wrote", out, len(d)+512, "bytes (so + 512 zero tail)")
PY
CAPIDIR=${ORT_CAPIDIR:?set ORT_CAPIDIR to onnxruntime/capi dir}
[ -e "$CAPIDIR/libonnxruntime.so" ] || ln -sf "$CAPIDIR/libonnxruntime.so.1.31.1" "$CAPIDIR/libonnxruntime.so"
export LD_LIBRARY_PATH="$CAPIDIR:${LD_LIBRARY_PATH:-}"
echo "=== end-to-end verify (in-process Rust embedding vs onnxruntime) ==="
${PYTHON:-python3} $D/verify_sentence.py 2>&1
echo "SCRIPT_EXIT=$?"
