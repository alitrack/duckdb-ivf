#!/usr/bin/env python3
"""Verify the ztg single-call bundle: ztg_search('text') == ivf_hybrid_embed.

Reuses the 5k all-MiniLM-384d setup (news table, news_384.idx, fts index,
'minilm' model). cd's into the bundle dir so ztg.sql's relative LOAD works.
"""
import os, time
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

BUNDLE = BUNDLE
os.makedirs(BUNDLE, exist_ok=True)
os.system("cp -f '%s' '%s'/ 2>/dev/null || true" % (IVFEXT, BUNDLE))
os.system("cp -f '%s' '%s'/ 2>/dev/null || true" % (MLEXT, BUNDLE))
os.chdir(BUNDLE)

con = duckdb.connect(config={"allow_unsigned_extensions": "true"})
# Load the three in-process extensions by absolute path (the bundle's
# ztg.sql is portable entry-point SQL only; deployment paths are external).
sql_probe = open(ZTGSQL).read()
for _ph, _real in [("/abs/path/to/ivf.duckdb_extension", IVFEXT),
                   ("/abs/path/to/ml.duckdb_extension", MLEXT),
                   ("/abs/path/to/model.onnx", ONNX),
                   ("/abs/path/to/tokenizer.json", TOK),
                   ("/abs/path/to/news.idx", IDX5K)]:
    sql_probe = sql_probe.replace(_ph, _real)
if "/abs/path/to/" in sql_probe:
    open("ztg_local.sql", "w").write(sql_probe)
con.execute("LOAD '%s'" % IVFEXT)
con.execute("LOAD '%s'" % MLEXT)
con.execute("INSTALL fts; LOAD fts")
# ztg.sql section 5 (create_fts_index) needs the corpus table to exist first
con.execute("CREATE OR REPLACE TABLE news (id BIGINT, text VARCHAR)")
con.execute("INSERT INTO news SELECT row_number() OVER (), text "
            "FROM (SELECT * FROM read_parquet('%s') LIMIT 5000)" % CORPUS)
# strip comment lines, fill (EDIT) placeholders with this deployment's paths
lines = [ln for ln in sql_probe.splitlines() if not ln.lstrip().startswith("--")]
for stmt in [s.strip() for s in "\n".join(lines).split(";") if s.strip()]:
    con.execute(stmt)
print("ztg bundle loaded", flush=True)

# model + index + fts (corpus table was already created before ztg.sql)
con.execute("SELECT * FROM ml_load_sentence('minilm','%s','%s',128)" % (ONNX, TOK))
t0 = time.time()
con.execute("SELECT * FROM ivf_load('newsm',?)", [IDX5K])
print("ivf_load %.3fs" % (time.time()-t0), flush=True)
con.execute("PRAGMA create_fts_index('news','id','text', stemmer='porter', stopwords='english', layered_search=true, overwrite=1)")
print("fts index ready", flush=True)

Q = ["crude oil prices", "python software library", "stock market earnings"]
allok = True
for q in Q:
    t0 = time.time()
    a = con.execute("SELECT id, round(score,6) FROM ztg_search_news(?)", [q]).fetchall()
    ta = time.time() - t0
    _embed_expr = "ml_embed_sentence('minilm', '%s')" % q.replace("'", "''")
    b = con.execute(
        "SELECT id, round(score,6) FROM ivf_hybrid_embed('newsm','news','id','text',?, ?, "
        "top_k:=10, per_path:=50, nprobe:=8, k_rrf:=60)",
        [q, _embed_expr]).fetchall()
    ok = a == b
    allok &= ok
    print("[%s] ztg_search == ivf_hybrid_embed: %s  (%.0fms)" % (q, ok, ta*1000), flush=True)
    print("       top3: %s" % a[:3], flush=True)
print("ZTG_BUNDLE_OK all_ok=%s" % allok, flush=True)
