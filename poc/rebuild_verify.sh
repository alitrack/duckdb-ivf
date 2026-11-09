#!/usr/bin/env bash
# repo-relative paths (env-overridable)
D="$(cd "$(dirname "$0")" && pwd)"
R="$(cd "$D/.." && pwd)"
CORPUS="${ZTG_CORPUS:-$D/data/ag_news_train.parquet}"
NPZ="${ZTG_EMB_NPZ:-$D/data/emb_bge_m3.npz}"
export ZTG_CORPUS="$CORPUS" ZTG_EMB_NPZ="$NPZ"
set -u
echo "=== rebuild (onnx feature) ==="
cd "${ZTG_ML_WS:?set ZTG_ML_WS to your duckdb-ml checkout}"
CARGO_TARGET_DIR=${ZTG_ML_TARGET:-$HOME/ml-target} cargo build --release --features onnx 2>&1 | grep -E "^error|--> src|Finished" | head -20
echo "build_exit=${PIPESTATUS[0]}"
echo "=== reassemble extension (libml.so + old-ml footer 512) ==="
SO=$D/libml.so
OLDML="${ZTG_ML_WS:?set ZTG_ML_WS to your duckdb-ml checkout}"/build/release/ml.duckdb_extension
OUT=$D/ml.duckdb_extension
python3 - "$SO" "$OLDML" "$OUT" <<'PY'
import sys
so, oldml, out = sys.argv[1:4]
d=open(so,'rb').read(); old=open(oldml,'rb').read()
open(out,'wb').write(d + old[-512:])
print("wrote", out, len(d)+512)
PY
echo "=== end-to-end verify ==="
CAPIDIR=${ORT_CAPIDIR:?set ORT_CAPIDIR to onnxruntime/capi dir}
if [ ! -e "$CAPIDIR/libonnxruntime.so" ]; then
  ln -sf "$CAPIDIR/libonnxruntime.so.1.31.1" "$CAPIDIR/libonnxruntime.so" 2>/dev/null || true
fi
export LD_LIBRARY_PATH="$CAPIDIR:${LD_LIBRARY_PATH:-}"
${PYTHON:-python3} $D/verify_sentence.py 2>&1
echo "SCRIPT_EXIT=$?"
