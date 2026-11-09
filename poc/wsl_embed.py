#!/usr/bin/env python3
"""Embed all 120k ag_news docs via Mac embed-server (bge-m3), save npz."""
import time, json, urllib.request
import duckdb
import numpy as np
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

API = _env("ZTG_BGE_API", "http://10.10.10.81:8765/v1/embeddings")
OUT_NPZ = NPZ
PARQ = CORPUS
OUT = OUT_NPZ
BATCH = 256

def embed(texts):
    payload = json.dumps({"input": texts}).encode()
    req = urllib.request.Request(API, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        j = json.loads(r.read())
    return np.array([d["embedding"] for d in j["data"]], dtype=np.float32)

t0 = time.time()
con = duckdb.connect()
texts = [r[0] for r in con.execute(
    "SELECT text FROM read_parquet(?)", [PARQ]).fetchall()]
N = len(texts)
print("MARK loaded N=%d" % N, flush=True)

D = 1024
emb = np.empty((N, D), dtype=np.float32)
t1 = time.time()
for i in range(0, N, BATCH):
    chunk = texts[i:i+BATCH]
    emb[i:i+len(chunk)] = embed(chunk)
    if i % (BATCH * 20) == 0:
        done = i + len(chunk)
        print("MARK emb %d/%d (%.0f%%) %.0f docs/s" % (done, N, 100*done/N, done/(time.time()-t1)), flush=True)
np.savez(OUT, emb=emb)
print("MARK emb-done %d docs in %.0fs shape=%s saved=%s" % (N, time.time()-t1, emb.shape, OUT), flush=True)
print("MARK norm-sample %.4f" % float(np.linalg.norm(emb[0])), flush=True)
