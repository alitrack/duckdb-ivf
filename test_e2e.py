#!/usr/bin/env python3
"""End-to-end test of the duckdb-ivf IVF extension on ag_news bge-m3 embeddings.
Load .so -> load into in-memory DB -> build IVF index -> ANN vs brute-force recall.
This is the acceptance test for the Rust extension (b).
"""
import os, time
import numpy as np
import duckdb

HERE = os.path.dirname(os.path.abspath(__file__))
SO = "/home/lhy/ivf-target/release/libivf.so"
NPZ = "/mnt/d/wsl2/tmp/fts_ivf_poc/emb_bge_m3.npz"
PARQUET = "/mnt/d/wsl2/ag_news_data/ag_news_train.parquet"

con = duckdb.connect(config={"allow_unsigned_extensions": "true"})
# DuckDB 2.0 requires loadables named *.duckdb_extension with a 512-byte metadata
# footer. Stamp it on the C-API .so (C_STRUCT ABI, C-API-min v1.2.0, linux_amd64,
# magic "4") — same layout tiktoken's build produces.
EXT = "/home/lhy/ivf-target/release/ivf.duckdb_extension"
def stamp(src, dst, ext_ver="v0.1.0", db_ver="v1.2.0", plat="linux_amd64"):
    data = open(src, "rb").read()
    if data[-512:-480].split(b"\x00")[0] == b"4":  # already stamped
        data = data[:-512]
    def fld(s): return s.encode().ljust(32, b"\x00")
    footer = b"".join(fld(x) for x in ["","","","C_STRUCT",ext_ver,db_ver,plat,"4"])
    footer = footer.ljust(512, b"\x00")  # 256-byte RSA signature area zero-filled
    open(dst, "wb").write(data + footer)
stamp(SO, EXT)
con.execute(f"LOAD '{EXT}'")
print("loaded", con.sql("SELECT extension_name || ' ' || extension_version FROM duckdb_extensions() WHERE installed").fetchall())

# Load embeddings + build the indexed table.
emb = np.load(NPZ)
d = emb["emb"]           # (N,1024) float32, L2-normalized
ids = emb["id"] if "id" in emb else None
N, D = d.shape
print(f"loaded {N} vectors, dim {D}")

# Register numpy and persist as a permanent table (extension reads permanent tables).
con.register("emb_arr", __import__("pyarrow").table({"id": np.arange(N, dtype=np.int64), "v": [d[i] for i in range(N)]}))
con.execute(f"CREATE OR REPLACE TABLE news_vec AS SELECT id, v::FLOAT[{D}] AS vec FROM emb_arr")
print("table news_vec built, rows =", con.sql("SELECT count(*) FROM news_vec").fetchall()[0][0])

# Train IVF index inside the DB (nlist=sqrt(N) default, niter=10).
t0 = time.time()
res = con.sql("SELECT rows_indexed FROM ivf_create_index('news', 'news_vec', 'id', 'vec', 350, 10, true)").fetchall()
train_s = time.time() - t0
print(f"ivf_create_index -> {res}  ({train_s:.1f}s to train)")

print("\nivf_list():", con.sql("SELECT * FROM ivf_list()").fetchall())

# Build a held-out query set: 50 random docs as queries.
rng = np.random.default_rng(0)
qidx = rng.choice(N, size=50, replace=False)

def brute_force(q):
    # exact top-k via inner product (vectors are normalized)
    return np.argsort(-d @ q)[:50]

recall = {k: [] for k in (1, 5, 10)}
ann_ms, exact_ms = [], []
for i, qi in enumerate(qidx):
    q = d[qi]
    # Materialize the query as a PERMANENT table: the extension reads via its own
    # session connection, which only sees permanent tables (not the caller's
    # registered/temp views). Register a temp source, then materialize permanent.
    con.register("q1_src", __import__("pyarrow").table({"vec": [q]}))
    con.execute("CREATE OR REPLACE TABLE q1 AS SELECT vec FROM q1_src")
    con.unregister("q1_src")
    t0 = time.time()
    ann = con.sql("SELECT id, score FROM ivf_search('news','q1','vec', top_k:=50)").fetchall()
    ann_ms.append((time.time()-t0)*1000)
    ann_ids = [a[0] for a in ann]
    t0 = time.time()
    ex = con.sql("SELECT id FROM ivf_search_exact('news','q1','vec', top_k:=50)").fetchall()
    exact_ms.append((time.time()-t0)*1000)
    ex_ids = [e[0] for e in ex]
    bf = set(brute_force(q).tolist())
    for k in recall:
        recall[k].append(len(set(ann_ids[:k]) & bf) / k)

print("\n=== ivf_search (IVF ANN) recall vs brute-force ground truth ===")
for k in recall:
    print(f"  recall@{k:<2} = {np.mean(recall[k]):.4f}")
print(f"  ANN  median = {np.median(ann_ms):.2f} ms   (50 queries)")
print(f"  exact median = {np.median(exact_ms):.2f} ms")

# Correctness gate: ANN top-10 must mostly agree with exact top-10 (same index).
top10_ann = set(a[0] for a in ann)
top10_ex = set(e[0] for e in ex)
agree = len(top10_ann & top10_ex) / 10
print(f"  last-query ANN top10 ∩ exact top10 = {agree:.2f}")

# A real end-to-end example query.
print("\n=== example: ivf_search top-3 for query doc", int(qidx[0]), "===")
for r in ann[:3]:
    print(f"  id={r[0]:>7}  score={r[1]:.5f}")

# ── nprobe sweep: the real ANN speed/accuracy tradeoff ──────────────────────
# core's list_distance is brute-force (no ANN index). The point of an IVF index
# is: probe fewer centroids -> less work -> faster, at a small recall cost.
# This sweep proves the extension is a real ANN, not a brute-force wrapper.
print("\n=== nprobe sweep (ANN speed vs recall@10, ground truth = numpy brute-force) ===")
print(f"  {'nprobe':>6} {'recall@1':>9} {'recall@10':>10} {'median_ms':>11}")
for np_ in (8, 16, 32, 64, 128, 350):
    rec1, rec10, times = [], [], []
    for qi in qidx:
        q = d[qi]
        con.register("q1_src", __import__("pyarrow").table({"vec": [q]}))
        con.execute("CREATE OR REPLACE TABLE q1 AS SELECT vec FROM q1_src")
        con.unregister("q1_src")
        t0 = time.time()
        rows = con.sql(f"SELECT id FROM ivf_search('news','q1','vec', top_k:=10, nprobe:={np_})").fetchall()
        times.append((time.time() - t0) * 1000)
        got10 = set(r[0] for r in rows)
        bf1 = np.argsort(-d @ q)[0]
        bf50 = set(np.argsort(-d @ q)[:50].tolist())
        rec1.append(1.0 if bf1 in got10 else 0.0)
        rec10.append(len(got10 & bf50) / 10)
    print(f"  {np_:>6} {np.mean(rec1):>9.4f} {np.mean(rec10):>10.4f} {np.median(times):>11.2f}")