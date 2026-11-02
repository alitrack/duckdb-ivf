import duckdb, numpy as np

con = duckdb.connect(config={"allow_unsigned_extensions": "true"})
con.execute("LOAD '/home/lhy/vss-target/release/vss.duckdb_extension'")

DIM = 16
VEC_EXPR = ", ".join(f"sin(r.i * {0.7 * (d + 1)}) * cos(r.i * {0.3 * (d + 2)})" for d in range(DIM))

# DOUBLE[] source table (the failing case)
con.execute(f"CREATE OR REPLACE TABLE td AS SELECT r.i AS id, [{VEC_EXPR}] AS vec FROM range(100) r(i)")
con.execute("CREATE OR REPLACE TABLE qd AS SELECT vec FROM td WHERE id = 37")

# A: list_transform per-element ::FLOAT (materialize) — both table and query
con.execute("""
CREATE OR REPLACE TABLE tA AS
SELECT id, list_transform(vec, lambda x: x::FLOAT) AS vec FROM td
""")
con.execute("""
CREATE OR REPLACE TABLE qA AS
SELECT list_transform(vec, lambda x: x::FLOAT) AS vec FROM qd
""")
con.execute("SELECT * FROM vss_create_index('idxA', 'main.tA', 'id', 'vec', 8, 10, true)")
a = con.execute("SELECT id, score FROM vss_search('idxA', 'main.qA', 'vec', top_k := 3, nprobe := 8) ORDER BY score DESC").fetchall()
print("A list_transform both:", a)

# B: inline list_transform in the create query path is not user-controllable;
# instead: does list_transform change the physical child? (compare to raw DOUBLE read via cast)
arr = con.execute("SELECT id, vec FROM td").fetchall()
vecs = np.stack([np.asarray(a_[1], dtype=np.float64) for a_ in arr])
D = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
qn = D[37]
d = np.linalg.norm(D - qn, axis=1)
print("GT top-3:", [(int(arr[i][0]), round(float(d[i]), 4)) for i in np.argsort(d)[:3]])
