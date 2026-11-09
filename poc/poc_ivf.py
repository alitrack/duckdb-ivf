#!/usr/bin/env python3
"""Step A: replace brute-force list_distance with duckdb-ivf (user's own ANN index).
ivf session-connection constraint: indexed + query tables must be permanent.
Compare: brute-force kNN (list_distance) vs ivf_search(nprobe) — recall + speed.
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
EMB   = "http://10.10.10.81:8765/v1/embeddings"
D     = 1024

def embed_api(texts):
    payload = json.dumps({"input": texts if isinstance(texts, list) else [texts]}).encode()
    req = urllib.request.Request(EMB, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        j = json.loads(r.read())
    return [d["embedding"] for d in j["data"]]

con = duckdb.connect(config={"allow_unsigned_extensions": "true"})
con.execute("INSTALL fts; LOAD fts")
con.execute("LOAD '%s'" % IVFEXT)
t0 = time.time()
con.execute("CREATE TABLE news AS SELECT row_number() over() AS id, text FROM read_parquet(?)", [PARQ])
emb = np.load(NPZ)["emb"]
ids = np.arange(1, len(emb)+1, dtype=np.int64)
con.register("emb_arr", pa.table({"id": ids, "emb": [emb[i].tolist() for i in range(len(emb))]})
)
con.execute("CREATE OR REPLACE TABLE vec AS SELECT id, emb::FLOAT[%d] AS v FROM emb_arr" % D)
print("tables ready %.1fs" % (time.time()-t0), flush=True)

# fts on text (permanent table required)
con.execute("PRAGMA create_fts_index('news','id','text', stemmer='porter', stopwords='english', layered_search=true)")
print("fts ready", flush=True)

# --- train IVF index (single-threaded k-means; expect ~150s for 120k x 1024) ---
t0 = time.time()
con.execute("SELECT * FROM ivf_create_index('news', 'vec', 'id', 'v', 350, 10, true)")
print("ivf_create_index: %.0fs" % (time.time()-t0), flush=True)
print("ivf_list:", con.execute("SELECT * FROM ivf_list()").fetchall(), flush=True)

queries = ["crude oil prices", "indonesia election vote", "tech company layoffs"]
print("\n%-26s %-9s %-9s %-9s %-9s %s" % ("query", "bf_ms", "ivf_ms", "recall@10", "speedup", "top5 match"))
for q in queries:
    qvec = embed_api([q])[0]
    # query vector must live in a permanent single-row table
    con.execute("CREATE OR REPLACE TEMP TABLE _q AS SELECT ?::FLOAT[%d] AS v" % D, [qvec])
    con.execute("CREATE OR REPLACE TABLE qtab AS SELECT * FROM _q")
    t1 = time.time()
    bf = con.execute("""SELECT id, d FROM (
        SELECT id, list_distance(v, (SELECT v FROM qtab LIMIT 1)) AS d FROM vec)
        ORDER BY d ASC LIMIT 10""").fetchall()
    bf_ms = (time.time()-t1)*1000
    t2 = time.time()
    ivf = con.execute("SELECT id, score FROM ivf_search('news', 'qtab', 'v', top_k := 10, nprobe := 8)").fetchall()
    ivf_ms = (time.time()-t2)*1000
    bf_ids = [r[0] for r in bf]; ivf_ids = [r[0] for r in ivf]
    recall = len(set(bf_ids) & set(ivf_ids)) / 10.0
    top5 = all(b in ivf_ids[:5] for b in bf_ids[:5])
    print("%-26s %-9.0f %-9.0f %-9.2f %-9.1fx %s" % (q, bf_ms, ivf_ms, recall, bf_ms/max(ivf_ms,0.1), top5))
print("DONE", flush=True)
