"""Regression: qtab (pre-computed vector) path of ivf_hybrid_search unchanged."""
import duckdb, numpy as np, pyarrow as pa, pyarrow.parquet as pq, time

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

PARQ = CORPUS
NPZ = NPZ
IDX = IDX120
EXT = IVFEXT

con = duckdb.connect(config={"allow_unsigned_extensions": "true"})
con.execute(f"LOAD '{EXT}'")
con.execute(f"LOAD '{EXT.replace('ivf','fts')}'") if False else con.execute("INSTALL fts; LOAD fts")

tbl = pq.read_table(PARQ, columns=["text"])
texts = tbl["text"].to_pylist()
emb = np.load(NPZ)["emb"]
con.execute("CREATE TABLE news(id BIGINT, text VARCHAR, label INT, vec FLOAT[])")
con.execute("FROM df").execute("INSERT INTO news SELECT row_number() OVER (), t, 0, v FROM (SELECT UNNEST(?, ?) AS (t, v))".format(*[None, None])) if False else None

# simpler: build table from python
arr_text = pa.array(texts)
arr_vec = pa.array([emb[i].tolist() for i in range(len(texts))], type=pa.list_(pa.float32()))
arr_id = pa.array(list(range(1, len(texts)+1)), type=pa.int64())
tbl2 = pa.table({"id": arr_id, "text": arr_text, "vec": arr_vec})
con.register("news_view", tbl2)
con.execute("CREATE TABLE news AS SELECT * FROM news_view")
con.execute("CREATE INDEX idx_news_text ON news USING fts(text)")
t0 = time.time()
con.execute("SELECT * FROM ivf_load('news', ?)", [IDX]).fetchall()
print(f"ivf_load {time.time()-t0:.2f}s")

for q in ["crude oil prices", "python software library", "stock market earnings"]:
    qvec = np.mean(emb[[r[0]-1 for r in con.execute(
        "SELECT docname::BIGINT AS id FROM fts_main_news.search_layered_bm25(?, top_k := 5)", [q]).fetchall()]], axis=0)
    qvec = (qvec / np.linalg.norm(qvec)).astype(np.float32)
    con.execute("CREATE TEMP TABLE qv(id BIGINT, vec FLOAT[]) VALUES (1, ?)", [qvec.tolist()])
    t0 = time.time()
    rows = con.execute("""
        SELECT id, round(score,6), bm25_rank, vec_rank FROM ivf_hybrid_search(
          'news','news','id','text',?,'qv','vec', top_k := 10, per_path := 50, nprobe := 8)
    """, [q]).fetchall()
    dt = time.time() - t0
    ok = len(rows) == 10 and all(r[1] > 0 for r in rows)
    print(f"[qtab] {q!r}: n={len(rows)} ok={ok} {dt*1000:.1f}ms  top3={[r[0] for r in rows[:3]]}")
print("REGRESSION OK" if ok else "REGRESSION FAILED")
