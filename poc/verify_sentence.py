#!/usr/bin/env python3
"""B verification: duckdb-ml in-process sentence embedding == onnxruntime reference.
Load MiniLM ONNX + tokenizer into the extension (ml_load_sentence), embed texts
via ml_embed_sentence, and compare against a standalone onnxruntime run of the
SAME model on the SAME text. Cosine similarity ~1.0 = the Rust path is correct.
"""
import duckdb, numpy as np, onnxruntime as ort, time, os

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

MODEL = MODELDIR
EXT   = MLSO
assert os.path.exists(EXT), f"missing {EXT}"

def ref_embed(text, max_len=128):
    sess = ort.InferenceSession(MODEL + "/model.onnx", providers=["CPUExecutionProvider"])
    from tokenizers import Tokenizer
    tz = Tokenizer.from_file(MODEL + "/tokenizer.json")
    enc = tz.encode(text, add_special_tokens=True)
    ids = enc.ids[:max_len]
    mask = [1]*len(ids)
    ttids = [0]*len(ids)
    a = np.array([ids], dtype=np.int64); m = np.array([mask], dtype=np.int64); t = np.array([ttids], dtype=np.int64)
    out = sess.run(None, {"input_ids": a, "attention_mask": m, "token_type_ids": t})[0]
    h = out[0]  # (seq, 384)
    v = h.mean(axis=0)
    v = v / (np.linalg.norm(v)+1e-12)
    return v.astype(np.float64)

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-12))

texts = ["the cat sat on the mat",
         "oil prices surged on supply fears in the market",
         "the python library loads the parquet file into a table"]

con = duckdb.connect(config={"allow_unsigned_extensions": "true"})
EXTD  = MLEXT
t0 = time.time()
con.execute("LOAD '%s'" % EXTD)
print("LOAD ext: %.2fs" % (time.time()-t0), flush=True)

t0 = time.time()
row = con.execute(
    "SELECT * FROM ml_load_sentence('minilm', ?, ?, 128)",
    [MODEL + "/model.onnx", MODEL + "/tokenizer.json"]
).fetchall()
print("ml_load_sentence ->", row, "(%.2fs)" % (time.time()-t0), flush=True)

print("\n%-45s %-6s %-10s %s" % ("text", "dim", "cos(vs ORT)", "match"), flush=True)
for txt in texts:
    t1 = time.time()
    v = con.execute("SELECT ml_embed_sentence('minilm', ?)", [txt]).fetchone()[0]
    ms = (time.time()-t1)*1000
    ref = ref_embed(txt)
    c = cosine(np.array(v), ref)
    print("%-45s %-6d %-10.6f %s" % (repr(txt[:44]), len(v), c, "OK" if c > 0.999 else "MISMATCH"))

# sanity: self-similarity high, unrelated low
va = con.execute("SELECT ml_embed_sentence('minilm','machine learning model')").fetchone()[0]
vb = con.execute("SELECT ml_embed_sentence('minilm','crude oil barrel price')").fetchone()[0]
print("\ncos(same-topic pair vs diff-topic): ", end="")
vc = con.execute("SELECT ml_embed_sentence('minilm','deep learning neural network')").fetchone()[0]
print("ML-vs-ML=%.4f  ML-vs-oil=%.4f" % (cosine(va, vc), cosine(va, vb)), flush=True)
print("DONE", flush=True)
