import duckdb
import numpy as np

con = duckdb.connect(config={"allow_unsigned_extensions": "true"})
con.execute("LOAD '/home/lhy/vss-target/release/vss.duckdb_extension'")

DIM = 16
VEC_EXPR = ", ".join(f"sin(r.i * {0.7 * (d + 1)}) * cos(r.i * {0.3 * (d + 2)})" for d in range(DIM))
con.execute(f"CREATE OR REPLACE TABLE t AS SELECT r.i AS id, [{VEC_EXPR}] AS vec FROM range(1000) r(i)")
con.execute("CREATE OR REPLACE TABLE q AS SELECT vec FROM t WHERE id = 37")

# numpy ground truth (L2-normalized, nearest = smallest distance)
arr = np.array(con.execute("SELECT id, vec FROM t").fetchall(), dtype=object)
ids = arr[:, 0].astype(int)
vecs = np.stack([np.asarray(v, dtype=np.float64) for v in arr[:, 1]])
qv = vecs[37]
qn = qv / np.linalg.norm(qv)
D = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
d = np.linalg.norm(D - qn, axis=1)
gt = np.argsort(d)[:5]
print("GT top-5 (id, dist):", [(int(ids[i]), round(float(d[i]), 6)) for i in gt])
print("dist to self (id 37):", round(float(d[37]), 6))
print("num ids with dist < dist(id37):", int((d < d[37]).sum()), "  (0 => self is unique nearest)")

con.execute("SELECT * FROM vss_create_index('idx', 'main.t', 'id', 'vec', 8, 10, true)")
ann = con.execute("SELECT id, score FROM vss_search('idx', 'main.q', 'vec', top_k := 20, nprobe := 8) ORDER BY score DESC").fetchall()
print("ANN nprobe=8 (=nlist, exhaustive) top-10:", ann[:10])

ex = con.execute("SELECT id, score FROM vss_search_exact('idx', 'main.q', 'vec', top_k := 5) ORDER BY score DESC").fetchall()
print("vss_search_exact top-5:", ex)
