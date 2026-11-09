#!/usr/bin/env python3
"""E2E nDCG@10 with REAL query embeddings (LAN bge-m3), same metric as ndcg_e2e.py.

Only difference vs ndcg_e2e.py: the query vector is NOT constructed (mean of BM25
top-5 doc embeddings) — it is the real bge-m3 embedding of the query phrase,
fetched from the LAN embedding service. Everything else (exact-vector GT, hybrid
RRF GT, nprobe=8 ANN under test) is identical, so this isolates the effect of
"real query vectors vs constructed ones" on the nDCG@10 number.

Corpus = ag_news 120k, embeddings bge-m3 1024-d (emb_bge_m3.npz), index = the
persisted news.ivf.idx (ivf_load, no retrain).
"""
import time, json, urllib.request
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
EMBED = "http://10.10.10.81:8765/v1/embeddings"

QUERIES = [
    "middle east diplomacy", "united nations summit", "european union election",
    "ukraine conflict news", "japan prime minister",
    "football transfer rumors", "olympic games", "tennis championship",
    "soccer league standings", "basketball finals",
    "crude oil prices", "stock market earnings", "corporate merger deal",
    "bank interest rates", "quarterly revenue report",
    "python software library", "machine learning research", "space exploration",
    "genetic engineering study", "smartphone release",
]

def embed_one(q):
    body = json.dumps({"model": "bge-m3", "input": q}).encode()
    req = urllib.request.Request(EMBED, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as r:
        d = json.loads(r.read())
    v = np.asarray(d["data"][0]["embedding"], dtype=np.float32)
    assert v.shape == (D,), (q, v.shape)
    return v

def dcg_at_k(ranked_ids, rel, k=10):
    s = 0.0
    for i, doc in enumerate(ranked_ids[:k]):
        s += rel.get(doc, 0.0) / np.log2(i + 2)
    return s

def ndcg_at_k(ranked_ids, gt_ranks, k=10):
    rel = {d: 1.0 / r for d, r in gt_ranks.items()}
    ideal = [d for d, _ in sorted(gt_ranks.items(), key=lambda x: x[1])[:k]]
    ideal_dcg = dcg_at_k(ideal, rel, k)
    if ideal_dcg <= 0:
        return 1.0
    return dcg_at_k(ranked_ids, rel, k) / ideal_dcg

def main():
    tbl = pq.read_table(PARQ, columns=["text"])
    texts = tbl["text"].to_pylist()
    ids   = list(range(1, len(texts)+1))

    # real query embeddings from LAN bge-m3 (the whole point of this run)
    qvecs = {}
    for q in QUERIES:
        qvecs[q] = embed_one(q)
    print("[ndcg-real] embedded %d queries via LAN bge-m3" % len(qvecs), flush=True)

    emb = np.load(NPZ)["emb"]
    con = duckdb.connect(config={"allow_unsigned_extensions": "true"})
    con.execute("LOAD '%s'" % EXT)
    con.execute("INSTALL fts; LOAD fts")
    con.execute("CREATE OR REPLACE TABLE news (id BIGINT, text VARCHAR)")
    con.executemany("INSERT INTO news VALUES (?,?)", list(zip(ids, texts)))
    con.register("emb_arr", pa.table({"id": ids, "emb": [emb[k].tolist() for k in range(len(ids))]})
)
    con.execute("CREATE OR REPLACE TABLE vec AS SELECT id, emb::FLOAT[%d] AS v FROM emb_arr" % D)
    t0 = time.time()
    con.execute("SELECT * FROM ivf_load('news','%s')" % IDX)
    print("[ndcg-real] ivf_load %.3fs" % (time.time()-t0), flush=True)
    con.execute("PRAGMA create_fts_index('news','id','text', stemmer='porter', stopwords='english', layered_search=true)")

    vec_scores, hyd_scores = [], []
    for q in QUERIES:
        qvec = qvecs[q]
        con.execute("DROP TABLE IF EXISTS qtab")
        con.execute("CREATE TABLE qtab AS SELECT ?::FLOAT[%d] AS v" % D, [qvec.tolist()])
        exact = [r[0] for r in con.execute(
            "SELECT id FROM ivf_search_exact('news','qtab','v', top_k:=50)").fetchall()]
        ann   = [r[0] for r in con.execute(
            "SELECT id FROM ivf_search('news','qtab','v', top_k:=50, nprobe:=8)").fetchall()]
        vec_scores.append(ndcg_at_k(ann, {d: r+1 for r, d in enumerate(exact)}))
        fts   = [r[0] for r in con.execute(
            "SELECT docname::BIGINT FROM fts_main_news.search_layered_bm25(?, top_k:=50)", [q]).fetchall()]
        rrf = {}
        for lst in (exact, fts):
            for r, d in enumerate(lst):
                rrf[d] = rrf.get(d, 0.0) + 1.0/(60.0 + r + 1.0)
        gt_h = [d for d, _ in sorted(rrf.items(), key=lambda x: (-x[1], x[0]))[:50]]
        hyd = [r[0] for r in con.execute(
            "SELECT id FROM ivf_hybrid_search('news','news','id','text',?, 'qtab','v', "
            "top_k:=50, per_path:=50, nprobe:=8, k_rrf:=60)", [q]).fetchall()]
        hyd_scores.append(ndcg_at_k(hyd, {d: r+1 for r, d in enumerate(gt_h)}))

    print("QUERIES=%d  (real bge-m3 query vectors)" % len(QUERIES), flush=True)
    print("ANN_vec  nDCG@10  mean=%.4f  min=%.4f" % (np.mean(vec_scores), np.min(vec_scores)), flush=True)
    print("HYBRID   nDCG@10  mean=%.4f  min=%.4f" % (np.mean(hyd_scores), np.min(hyd_scores)), flush=True)
    print("per-query ANN:  %s" % ["%.3f" % s for s in vec_scores], flush=True)
    print("per-query HYB:  %s" % ["%.3f" % s for s in hyd_scores], flush=True)
    print("NDCG_REAL_DONE", flush=True)

if __name__ == "__main__":
    main()
