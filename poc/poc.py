#!/usr/bin/env python3
"""ztg PoC: BM25(fts) + vector KNN + RRF hybrid on real ag_news (120k, bge-m3 1024d).
Embedding entry = live bge-m3 OpenAI-compatible service (10.10.10.81:8765).
Vector path = DuckDB 2.0 list_distance (brute-force kNN, correct at 120k).
fts = layered BM25. Fusion = RRF macro (your hybrid_search.sql).
"""
import time, json, urllib.request
import numpy as np, duckdb, pyarrow as pa

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

PARQ  = CORPUS
NPZ   = NPZ
MACRO = open(_env("ZTG_HYBRID_SQL", _os.path.join(_D, "hybrid_search.sql"))).read()
EMB   = "http://10.10.10.81:8765/v1/embeddings"
D     = 1024

def embed_api(texts):
    payload = json.dumps({"input": texts if isinstance(texts, list) else [texts]}).encode()
    req = urllib.request.Request(EMB, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        j = json.loads(r.read())
    return [d["embedding"] for d in j["data"]]

con = duckdb.connect()
t0 = time.time()
con.execute("INSTALL fts; LOAD fts")
con.execute("CREATE TABLE news AS SELECT row_number() over() AS id, text FROM read_parquet(?)", [PARQ])
con.execute("PRAGMA create_fts_index('news','id','text', stemmer='porter', stopwords='english', layered_search=true)")
emb = np.load(NPZ)["emb"]
ids = np.arange(1, len(emb)+1, dtype=np.int64)
con.register("emb_arr", pa.table({"id": ids, "emb": [emb[i].tolist() for i in range(len(emb))]})
)
con.execute("CREATE OR REPLACE TABLE vec AS SELECT id, emb::FLOAT[%d] AS v FROM emb_arr" % D)
setup = time.time()-t0
print("SETUP %.1fs  news=%d vec=%d" % (setup,
    con.execute("select count(*) from news").fetchone()[0],
    con.execute("select count(*) from vec").fetchone()[0]), flush=True)
con.execute(MACRO)
print("registered: news_hybrid_search (BM25 + vec KNN + RRF)", flush=True)

for q in ["crude oil prices", "indonesia election vote", "tech company layoffs", "iphone sales"]:
    t1=time.time(); qvec = embed_api([q])[0]; t_emb=time.time()-t1
    t2=time.time()
    rows = con.execute("SELECT * FROM news_hybrid_search(?, ?)", [q, qvec]).fetchall()
    ms = (time.time()-t2)*1000
    print("\nQUERY %-24r  emb=%.0fms  search=%.0fms  top%d" % (q, t_emb*1000, ms, len(rows)), flush=True)
    for r in rows[:5]:
        print("   id=%-6d rrf=%.4f bm25=%-4s vec=%-4s %s" % (r[0], r[1], r[2], r[3], (r[4] or '')[:54]), flush=True)
print("\nDONE", flush=True)
