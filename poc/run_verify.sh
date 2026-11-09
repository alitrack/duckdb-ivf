#!/usr/bin/env bash
# repo-relative paths (env-overridable)
D="$(cd "$(dirname "$0")" && pwd)"
R="$(cd "$D/.." && pwd)"
CORPUS="${ZTG_CORPUS:-$D/data/ag_news_train.parquet}"
NPZ="${ZTG_EMB_NPZ:-$D/data/emb_bge_m3.npz}"
export ZTG_CORPUS="$CORPUS" ZTG_EMB_NPZ="$NPZ"
set -u
SO=$D/libml.so
EXT=$D/ml_onnx.duckdb_extension
# clean copy (no bogus footer)
cp "$SO" "$EXT"
echo "copied -> $EXT  ($(stat -c%s "$EXT") bytes)"
CAPIDIR=${ORT_CAPIDIR:?set ORT_CAPIDIR to onnxruntime/capi dir}
[ -e "$CAPIDIR/libonnxruntime.so" ] || ln -sf "$CAPIDIR/libonnxruntime.so.1.31.1" "$CAPIDIR/libonnxruntime.so"
export LD_LIBRARY_PATH="$CAPIDIR:${LD_LIBRARY_PATH:-}"
${PYTHON:-python3} $D/verify_sentence.py 2>&1
echo "SCRIPT_EXIT=$?"
