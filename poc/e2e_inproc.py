#!/usr/bin/env python3
"""True end-to-end: ONE SQL call, TEXT IN, NO pre-computed query vector.

query text
  -> embed_expr := 'ml_embed_sentence(''minilm'', q)'   [duckdb-ml, in-process ORT]
  -> ivf ANN (384d)  +  FTS BM25  ->  RRF
  -> top-k

Corpus vectors are ALSO embedded in-process by the same model
(ml_embed_sentence over 120k rows), indexed, and persisted.

Modes:
  train - embed 120k docs in-process, build+save 384d index, e2e queries
  load  - fresh process: ivf_load (no retrain) + e2e queries

Per-query checks:
  A = Rust fn with embed_expr (pure text in)
  B = Rust fn with qtab (vector pre-computed in SQL, same call otherwise)
  R = SQL baseline RRF (fts + ivf_search qtab nprobe=8)
  PASS: A ids+scores == B == R
"""
import sys, os, time
import duckdb

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
IDX  = IDX5K
IVF  = IVFEXT
ML   = MLEXT
MODELS = (ONNX,
          TOK)
Q = ["crude oil prices", "python software library", "stock market earnings"]

con = duckdb.connect(config={"allow_unsigned_extensions": "true"})
con.execute("LOAD '%s'" % IVF)
con.execute("LOAD '%s'" % ML)
con.execute("INSTALL fts; LOAD fts")
con.execute("SELECT * FROM ml_load_sentence('minilm','%s','%s',128)" % MODELS)
print("model loaded (in-process ORT)", flush=True)

con.execute("CREATE OR REPLACE TABLE news (id BIGINT, text VARCHAR)")
con.execute("INSERT INTO news SELECT row_number() OVER (), text "
            "FROM (SELECT * FROM read_parquet(?) LIMIT 5000)", [PARQ])
n = con.execute("SELECT count(*) FROM news").fetchone()[0]
print("news=%d" % n, flush=True)

mode = sys.argv[1] if len(sys.argv) > 1 else "train"
if mode == "train":
    t0 = time.time()
    con.execute("CREATE OR REPLACE TABLE vecm AS "
                "SELECT id, ml_embed_sentence('minilm', text)::FLOAT[] AS v FROM news")
    dim = len(con.execute("SELECT v FROM vecm LIMIT 1").fetchone()[0])
    print("[train] in-process embedding of %d docs: %.1fs (dim=%d)" % (n, time.time() - t0, dim), flush=True)
    t0 = time.time()
    con.execute("SELECT * FROM ivf_create_index('newsm','vecm','id','v',200,10,true)")
    print("[train] ivf_create_index(384d): %.1fs" % (time.time() - t0), flush=True)
    con.execute("SELECT * FROM ivf_save('newsm','%s')" % IDX)
    print("[train] saved %s (%.1f MB)" % (IDX, os.path.getsize(IDX) / 1e6), flush=True)
else:
    t0 = time.time()
    print("[load]", con.execute("SELECT * FROM ivf_load('newsm','%s')" % IDX).fetchall(),
          "%.3fs" % (time.time() - t0), flush=True)

con.execute("PRAGMA create_fts_index('news','id','text', stemmer='porter', stopwords='english', layered_search=true)")

con.execute("""
CREATE OR REPLACE MACRO hybrid_ref(qtab_v, top_k := 10, k_rrf := 60, per_path := 50) AS TABLE
  WITH f  AS (SELECT docname::BIGINT AS id, row_number() OVER () AS frk
              FROM fts_main_news.search_layered_bm25(?, top_k := per_path)),
       vr AS (SELECT id, row_number() OVER (ORDER BY score DESC) AS vrk
              FROM ivf_search('newsm', qtabv, 'v', top_k := per_path, nprobe := 8)),
       u  AS (SELECT id, frk, NULL::BIGINT AS vrk FROM f
              UNION ALL SELECT id, NULL, vrk FROM vr),
       rrf AS (SELECT id,
                sum(CASE WHEN frk IS NOT NULL THEN 1.0/(k_rrf + frk) ELSE 0 END)
              + sum(CASE WHEN vrk IS NOT NULL THEN 1.0/(k_rrf + vrk) ELSE 0 END) AS rrf_score
              FROM u GROUP BY id),
       top AS (SELECT id, rrf_score FROM rrf ORDER BY rrf_score DESC, id ASC LIMIT top_k)
  SELECT top.id, round(rrf_score, 6) AS rrf_score FROM top ORDER BY rrf_score DESC, id ASC
""") if False else None

allok = True
for q in Q:
    # pre-compute the SAME in-process embedding into qtab (reference vector)
    qvec = con.execute("SELECT list_transform(ml_embed_sentence('minilm', ?), lambda x: x::FLOAT)", [q]).fetchone()[0]
    con.execute("DROP TABLE IF EXISTS qtab")
    con.execute("CREATE TABLE qtab AS SELECT ?::FLOAT[] AS v", [qvec])

    # A: pure text in via ivf_hybrid_embed (embed_expr is the 6th positional arg).
    # The query text is embedded into the embed_expr string (quotes escaped).
    embed_expr = "ml_embed_sentence('minilm', '%s')" % q.replace("'", "''")
    t0 = time.time()
    a = con.execute(
        "SELECT id, round(score,6), bm25_rank, vec_rank FROM ivf_hybrid_embed("
        "'newsm','news','id','text',?, ?, "
        "top_k:=10, per_path:=50, nprobe:=8, k_rrf:=60)", [q, embed_expr]).fetchall()
    ta = time.time() - t0
    # B: same call with qtab
    b = con.execute(
        "SELECT id, round(score,6), bm25_rank, vec_rank FROM ivf_hybrid_search("
        "'newsm','news','id','text',?, 'qtab','v', top_k:=10, per_path:=50, nprobe:=8, k_rrf:=60)",
        [q]).fetchall()
    # R: SQL baseline RRF with the same branches
    r = con.execute("""
      WITH f  AS (SELECT docname::BIGINT AS id, row_number() OVER () AS frk
                  FROM fts_main_news.search_layered_bm25(?, top_k := 50)),
           vr AS (SELECT id, row_number() OVER (ORDER BY score DESC) AS vrk
                  FROM ivf_search('newsm', 'qtab', 'v', top_k := 50, nprobe := 8)),
           u  AS (SELECT id, frk, NULL::BIGINT AS vrk FROM f
                  UNION ALL SELECT id, NULL, vrk FROM vr),
           rrf AS (SELECT id,
                    sum(CASE WHEN frk IS NOT NULL THEN 1.0/(60 + frk) ELSE 0 END)
                  + sum(CASE WHEN vrk IS NOT NULL THEN 1.0/(60 + vrk) ELSE 0 END) AS s
                  FROM u GROUP BY id),
           top AS (SELECT id, s FROM rrf ORDER BY s DESC, id ASC LIMIT 10)
      SELECT id, round(s,6) FROM top ORDER BY s DESC, id ASC
    """, [q]).fetchall()

    a_set = {x[0]: x[1] for x in a}
    b_set = {x[0]: x[1] for x in b}
    r_set = {x[0]: x[1] for x in r}
    ok_a_b = a == b
    ok_set = set(a_set) == set(r_set)
    ok_score = all(abs(a_set.get(i, 0) - r_set.get(i, 0)) < 1e-6 for i in set(a_set) | set(r_set))
    ok = ok_a_b and ok_set and ok_score
    allok &= ok
    print("[%s] q=%r  A==B:%s  set==R:%s  score==R:%s  (%.0fms)  PASS=%s"
          % (mode, q, ok_a_b, ok_set, ok_score, ta * 1000, ok), flush=True)
    print("       top5: %s" % [(x[0], x[1]) for x in a[:5]], flush=True)

print("%s_E2E_DONE all_ok=%s" % (mode.upper(), allok), flush=True)
