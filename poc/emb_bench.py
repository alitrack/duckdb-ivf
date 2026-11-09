#!/usr/bin/env python3
"""Measure in-process embedding throughput (1k docs), then decide 120k strategy."""
import time, duckdb

# --- repo-relative paths (env-overridable); data files are gitignored ---
import os as _os
_D = _os.path.dirname(_os.path.abspath(__file__))
_R = _os.path.dirname(_D)
def _env(k, d):
    v = _os.environ.get(k)
    return v if v else d
CORPUS = _env("ZTG_CORPUS", _os.path.join(_D, "data", "ag_news_train.parquet"))
NPZ = _env("ZTG_EMB_NPZ", _os.path.join(_D, "data", "emb_bge_m3.npz"))
IDX120 = _env("ZTG_IDX_120K", _os.path.join(_D, "data", "news.ivf.idx"))
IDX5K = _env("ZTG_IDX_5K", _os.path.join(_D, "data", "news_384.idx"))
IVFEXT = _env("ZTG_IVF_EXT", _os.path.join(_R, "build", "release", "ivf.duckdb_extension"))
MLEXT = _env("ZTG_ML_EXT", _os.path.join(_D, "ml.duckdb_extension"))
ONNX = _env("ZTG_MODEL_ONNX", _os.path.join(_D, "model", "model.onnx"))
TOK = _env("ZTG_MODEL_TOK", _os.path.join(_D, "model", "tokenizer.json"))
MODELDIR = _env("ZTG_MODEL_DIR", _os.path.join(_D, "model"))
PERSIST_JSON = _env("ZTG_PERSIST_JSON", _os.path.join(_D, "data", "_persist_check.json"))
MLSO = _env("ZTG_ML_SO", _os.path.join(_D, "libml.so"))
ZTGSQL = _env("ZTG_SQL", _os.path.join(_R, "ztg", "ztg.sql"))
BUNDLE = _env("ZTG_BUNDLE_DIR", _os.path.join(_R, "ztg_test"))
# --- end repo-relative paths ---
con = duckdb.connect(config={"allow_unsigned_extensions": "true"})
con.execute("LOAD '%s'" % MLEXT)
con.execute("SELECT * FROM ml_load_sentence('minilm','%s','%s',128)" % (ONNX, TOK))
con.execute("CREATE TABLE t AS SELECT row_number() OVER () id, text FROM (SELECT * FROM read_parquet('%s') LIMIT 1000)" % CORPUS)
t0 = time.time()
rows = con.execute("SELECT id, ml_embed_sentence('minilm', text) FROM t").fetchall()
dt = time.time() - t0
print("1000 docs in %.1fs -> %.0f docs/s; 120k est %.0f min" % (dt, 1000/dt, 120000/dt/60))
