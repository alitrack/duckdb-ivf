#!/usr/bin/env python3
"""C1: ivf index persistence — save / load without retraining.
mode=train : build vec table, train ivf (slow), save to file, run queries -> JSON
mode=load  : FRESH process, ivf_load (no retrain), run same queries, compare to JSON
Self-contained: query vectors come from the npz itself (no LAN embedding needed).
"""
import sys, os, time, json
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

PARQ = CORPUS
NPZ  = NPZ
EXT  = IVFEXT
IDX  = IDX120
CHK  = PERSIST_JSON
D = 1024
QLABELS = ["q0(row0)", "q1(row5000)", "q2(row99999)"]
QROWS   = [0, 5000, 99999]

def new_con():
    con = duckdb.connect(config={"allow_unsigned_extensions": "true"})
    con.execute("LOAD '%s'" % EXT)
    return con

def run_queries(con, emb):
    """3 queries from the corpus itself; return {label: [(id,score)...]}."""
    out = {}
    for lab, r in zip(QLABELS, QROWS):
        qvec = emb[r]
        con.execute("DROP TABLE IF EXISTS qtab")
        con.execute("CREATE TABLE qtab AS SELECT ?::FLOAT[%d] AS v" % D, [qvec.tolist()])
        rows = con.execute(
            "SELECT id, round(score,6) FROM ivf_search('news','qtab','v', top_k:=10, nprobe:=8)"
        ).fetchall()
        out[lab] = rows
    return out

def build_vec_table(con, emb):
    t0 = time.time()
    con.register("emb_arr", pa.table({
        "id": np.arange(1, len(emb)+1, dtype=np.int64),
        "emb": [emb[i].tolist() for i in range(len(emb))],
    }))
    con.execute("CREATE OR REPLACE TABLE vec AS SELECT id, emb::FLOAT[%d] AS v FROM emb_arr" % D)
    print("[train] vec table ready %.1fs" % (time.time()-t0), flush=True)

if sys.argv[1] == "train":
    emb = np.load(NPZ)["emb"]
    con = new_con()
    build_vec_table(con, emb)
    t0 = time.time()
    con.execute("SELECT * FROM ivf_create_index('news','vec','id','v',350,10,true)")
    train_s = time.time()-t0
    print("[train] ivf_create_index: %.1fs" % train_s, flush=True)
    print("[train] ivf_list:", con.execute("SELECT * FROM ivf_list()").fetchall(), flush=True)
    # save
    t0 = time.time()
    res = con.execute("SELECT * FROM ivf_save('news','%s')" % IDX).fetchall()
    save_s = time.time()-t0
    print("[train] ivf_save -> %s (%.3fs)" % (str(res), save_s), flush=True)
    if os.path.exists(IDX):
        print("[train] file size: %.1f MB" % (os.path.getsize(IDX)/1e6), flush=True)
    res_q = run_queries(con, emb)
    json.dump(res_q, open(CHK, "w"))
    for lab in QLABELS:
        print("[train] %s top10 ids: %s" % (lab, [r[0] for r in res_q[lab]]), flush=True)
    print("TRAIN_DONE train=%.1fs" % train_s, flush=True)

else:  # load  — FRESH process, no index in memory yet
    emb = np.load(NPZ)["emb"]
    con = new_con()
    # sanity: nothing loaded yet
    print("[load] ivf_list before:", con.execute("SELECT count(*) FROM ivf_list()").fetchone(), flush=True)
    t0 = time.time()
    res = con.execute("SELECT * FROM ivf_load('news','%s')" % IDX).fetchall()
    load_s = time.time()-t0
    print("[load] ivf_load -> %s (%.3fs)" % (str(res), load_s), flush=True)
    print("[load] ivf_list after:", con.execute("SELECT * FROM ivf_list()").fetchall(), flush=True)
    res_q = run_queries(con, emb)
    prev = json.load(open(CHK))
    all_match = True
    for lab in QLABELS:
        a = [r[0] for r in prev[lab]]
        b = [r[0] for r in res_q[lab]]
        same = a == b
        all_match &= same
        print("[load] %s match=%s" % (lab, same), flush=True)
        if not same:
            print("   train: %s" % a, flush=True)
            print("   load : %s" % b, flush=True)
    print("LOAD_DONE load=%.3fs identical=%s" % (load_s, all_match), flush=True)
