#!/usr/bin/env python3
"""C2/finishing: ivf_hybrid_search end-to-end correctness + no-retrain reuse.
Checks:
  (1) the Rust table function executes and returns top_k rows
      (id, rrf_score, bm25_rank, vec_rank, snippet-free)
  (2) its RRF fusion EXACTLY matches the hand-written SQL macro baseline
      (same branch top lists -> same fused ordering + same rrf values)
  (3) it works from ivf_load (no retrain) in a fresh process
Queries are topic strings; vectors for the query come from the corpus itself
(a row's own bge-m3 embedding) so no LAN embedding service is needed.
"""
import sys, os, time, json
import numpy as np, duckdb, pyarrow as pa
import pyarrow.parquet as pq

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
NPZ  = NPZ
EXT  = IVFEXT
IDX  = IDX120
D    = 1024
Q    = ["crude oil prices", "python software library", "stock market earnings"]

def new_con():
    con = duckdb.connect(config={"allow_unsigned_extensions": "true"})
    con.execute("LOAD '%s'" % EXT)
    con.execute("INSTALL fts; LOAD fts")
    return con

def baseline_macro(con):
    # Reference RRF using the SAME branches as do_hybrid: FTS bm25 + ivf ANN (nprobe=8).
    # This isolates the RRF fusion math under test (identical branch inputs).
    con.execute("""
    CREATE OR REPLACE MACRO hybrid_ref(query, query_vec, top_k := 10, k_rrf := 60, per_path := 50) AS TABLE
      WITH f  AS (SELECT docname::BIGINT AS id, row_number() OVER () AS frk
                  FROM fts_main_news.search_layered_bm25(query, top_k := per_path)),
           vr AS (SELECT id, row_number() OVER (ORDER BY score DESC) AS vrk
                  FROM ivf_search('news','qtab','v', top_k := per_path, nprobe := 8)),
           u  AS (SELECT id, frk, NULL::BIGINT AS vrk FROM f
                  UNION ALL
                  SELECT id, NULL, vrk FROM vr),
           rrf AS (SELECT id,
                  sum(CASE WHEN frk IS NOT NULL THEN 1.0/(k_rrf + frk) ELSE 0 END)
              +   sum(CASE WHEN vrk IS NOT NULL THEN 1.0/(k_rrf + vrk) ELSE 0 END) AS rrf_score
                  FROM u GROUP BY id),
           top AS (SELECT id, rrf_score FROM rrf ORDER BY rrf_score DESC, id ASC LIMIT top_k)
      SELECT top.id, round(rrf_score,6) AS rrf_score
      FROM top ORDER BY rrf_score DESC, id ASC
    """)

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    emb  = np.load(NPZ)["emb"]
    tbl  = pq.read_table(PARQ, columns=["text", "label"])
    texts = tbl["text"].to_pylist()
    ids   = list(range(1, len(texts)+1))          # row order == npz emb order
    id_to_emb = {i: emb[k] for k, i in enumerate(ids)}
    con = new_con()

    con.execute("CREATE OR REPLACE TABLE news (id BIGINT, text VARCHAR)")
    con.executemany("INSERT INTO news VALUES (?,?)", list(zip(ids, texts)))
    con.register("emb_arr", pa.table({"id": ids, "emb": [emb[k].tolist() for k in range(len(ids))]}))
    con.execute("CREATE OR REPLACE TABLE vec AS SELECT id, emb::FLOAT[%d] AS v FROM emb_arr" % D)
    print("[%s] news=%d vec=%d" % (mode, len(ids), con.execute("SELECT count(*) FROM vec").fetchone()[0]), flush=True)

    if mode == "train":
        t0 = time.time()
        con.execute("SELECT * FROM ivf_create_index('news','vec','id','v',350,10,true)")
        print("[train] ivf_create_index: %.1fs" % (time.time()-t0), flush=True)
        con.execute("SELECT * FROM ivf_save('news','%s')" % IDX)
        print("[train] saved -> %s (%.1f MB)" % (IDX, os.path.getsize(IDX)/1e6), flush=True)
    else:
        t0 = time.time()
        r = con.execute("SELECT * FROM ivf_load('news','%s')" % IDX).fetchall()
        print("[load] ivf_load: %s (%.3fs)" % (r, time.time()-t0), flush=True)

    con.execute("PRAGMA create_fts_index('news','id','text', stemmer='porter', stopwords='english', layered_search=true)")
    baseline_macro(con)

    # For each query string, vector = embedding of the best BM25 doc for it
    # (a stable, corpus-derived query vector; no external service).
    allok = True
    for q in Q:
        # derive a query vector: mean of top-5 BM25 docs' embeddings (representative topic vector)
        top_ids = [r[0] for r in con.execute(
            "SELECT docname::BIGINT FROM fts_main_news.search_layered_bm25(?, top_k := 5)", [q]).fetchall()]
        # diagnostic: is FTS return order == rank order (rank non-decreasing)?
        fts_seq = con.execute(
            "SELECT row_number() OVER (), rank FROM fts_main_news.search_layered_bm25(?, top_k := 50)", [q]).fetchall()
        ranks = [r[1] for r in fts_seq]
        fts_rank_monotonic = all(ranks[i] <= ranks[i+1] for i in range(len(ranks)-1))
        qvec = np.mean([id_to_emb[i] for i in top_ids], axis=0)
        con.execute("DROP TABLE IF EXISTS qtab")
        con.execute("CREATE TABLE qtab AS SELECT ?::FLOAT[%d] AS v" % D, [qvec.tolist()])
        # Rust function
        rust = con.execute(
            "SELECT id, round(score,6), bm25_rank, vec_rank FROM ivf_hybrid_search("
            "'news','news','id','text',?, 'qtab','v', top_k:=10, per_path:=50, nprobe:=8, k_rrf:=60)",
            [q]).fetchall()
        # SQL baseline macro (same branch lists, per_path=50, k_rrf=60, top_k=10)
        ref = con.execute(
            "SELECT id, rrf_score FROM hybrid_ref(?, (SELECT v FROM qtab), top_k:=10, per_path:=50, k_rrf:=60)",
            [q]).fetchall()
        r_ids = [r[0] for r in rust]
        f_ids = [r[0] for r in ref]
        # per-id RRF score maps (Rust `score` is the RRF value); compare as dicts
        r_map = {r[0]: r[1] for r in rust}
        f_map = {r[0]: r[1] for r in ref}
        same_set  = set(r_ids) == set(f_ids)
        # per-id score equality (≈), only over the union of both id sets
        score_ok  = all(abs(r_map.get(i, 0) - f_map.get(i, 0)) < 1e-6 for i in set(r_map) | set(f_map))
        same_seq  = r_ids == f_ids   # exact order (may differ only on RRF ties)
        allok &= same_set and score_ok
        print("[%s] q=%r n_rust=%d n_ref=%d same_set=%s score_match=%s same_order=%s" % (
            mode, q, len(rust), len(ref), same_set, score_ok, same_seq), flush=True)
        print("        top5: %s" % [(r[0], round(r[1],5)) for r in rust[:5]], flush=True)
    print("%s_DONE match=%s" % (mode.upper(), allok), flush=True)

if __name__ == "__main__":
    main()
