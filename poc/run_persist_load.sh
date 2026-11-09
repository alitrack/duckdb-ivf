#!/usr/bin/env bash
# repo-relative paths (env-overridable)
D="$(cd "$(dirname "$0")" && pwd)"
R="$(cd "$D/.." && pwd)"
CORPUS="${ZTG_CORPUS:-$D/data/ag_news_train.parquet}"
NPZ="${ZTG_EMB_NPZ:-$D/data/emb_bge_m3.npz}"
export ZTG_CORPUS="$CORPUS" ZTG_EMB_NPZ="$NPZ"
# C1 load phase: FRESH process, ivf_load (no retrain), compare to saved results
${PYTHON:-python3} $D/verify_persist.py load
