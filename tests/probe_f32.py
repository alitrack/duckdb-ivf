import duckdb, numpy as np
con = duckdb.connect(config={"allow_unsigned_extensions": "true"})
con.execute("LOAD '/home/lhy/vss-target/release/vss.duckdb_extension'")

DIM = 16
# explicit FLOAT elements (avoid any DOUBLE[] -> FLOAT[] cast entirely)
VEC_EXPR = ", ".join(f"(sin(r.i * {0.7 * (d + 1)}) * cos(r.i * {0.3 * (d + 2)}))::FLOAT" for d in range(DIM))
con.execute(f"CREATE OR REPLACE TABLE t2 AS SELECT r.i AS id, [{VEC_EXPR}] AS vec FROM range(1000) r(i)")
print("col type:", con.execute("DESCRIBE t2").fetchall())

con.execute("CREATE OR REPLACE TABLE q2 AS SELECT vec FROM t2 WHERE id = 37")
con.execute("SELECT * FROM vss_create_index('idx2', 'main.t2', 'id', 'vec', 8, 10, true)")
ann = con.execute("SELECT id, score FROM vss_search('idx2', 'main.q2', 'vec', top_k := 5, nprobe := 8) ORDER BY score DESC").fetchall()
print("scores (explicit FLOAT):", ann)

# numpy ground truth
arr = con.execute("SELECT id, vec FROM t2").fetchall()
ids = [int(a[0]) for a in arr]
vecs = np.stack([np.asarray(a[1], dtype=np.float64) for a in arr])
D = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
qn = D[37]
d = np.linalg.norm(D - qn, axis=1)
print("GT top-5:", [(int(ids[i]), round(float(d[i]),4)) for i in np.argsort(d)[:5]])
