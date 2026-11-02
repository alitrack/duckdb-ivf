#!/usr/bin/env python3
"""SQL-level e2e for duckdb-ivf (the CI harness).

Host = a real DuckDB process (the `duckdb` Python package); the artifact under
test is the built+stamped ivf.duckdb_extension passed as argv[1]. This mirrors
what a user does (`LOAD` + SQL) and needs no embedded/C++ test host, so it runs
on every CI platform that has a duckdb wheel (linux/amd64, osx/arm64+amd64,
windows/amd64).

Run:  make test   (builds + stamps, then calls this script)
"""
import os
import sys

import duckdb

ARTIFACT = os.path.abspath(sys.argv[1])
assert os.path.exists(ARTIFACT), f"artifact not found: {ARTIFACT}"
N = 1000          # rows
DIM = 16
NLIST = 8
TOPK = 10

# Deterministic pseudo-random vectors: a smooth per-dim sinusoid mix so the
# nearest-neighbour structure is real (neighbours in i are close in space).
VEC_EXPR = ", ".join(f"sin(r.i * {0.7 * (d + 1)}) * cos(r.i * {0.3 * (d + 2)})" for d in range(DIM))

con = duckdb.connect(config={"allow_unsigned_extensions": "true"})
con.execute(f"LOAD '{ARTIFACT}'")
print(f"loaded: {con.execute('SELECT ivf_version()').fetchall()[0][0]}")

# ── 1. build a deterministic table ────────────────────────────────────────
con.execute(f"""
CREATE TABLE t AS
SELECT r.i AS id, [{VEC_EXPR}] AS vec FROM range({N}) r(i)
""")

# ── 2. create index ───────────────────────────────────────────────────────
rows = con.execute(
    f"SELECT rows_indexed FROM ivf_create_index('idx', 'main.t', 'id', 'vec', {NLIST}, 10, true)"
).fetchall()
assert rows == [(N,)], f"expected all {N} rows indexed, got {rows}"
print(f"[ok] ivf_create_index indexed {N} rows")

# ── 3. ivf_list metadata ──────────────────────────────────────────────────
meta = {r[0]: r for r in con.execute("SELECT name, dim, n, nlist FROM ivf_list()").fetchall()}
assert meta.get("idx") == ("idx", DIM, N, NLIST), f"ivf_list metadata wrong: {meta.get('idx')}"
print("[ok] ivf_list reports name/dim/n/nlist")

# ── 4. self-search: a vector is its own nearest neighbour ─────────────────
for qid in (0, 37, 555, 999):
    con.execute(f"CREATE OR REPLACE TABLE q AS SELECT vec FROM t WHERE id = {qid}")
    top = con.execute(
        f"SELECT id FROM ivf_search('idx', 'main.q', 'vec', top_k := 1, nprobe := {NLIST})"
    ).fetchall()[0][0]
    assert top == qid, f"self-search for {qid} returned {top}"
print("[ok] self-search (nprobe=nlist) returns the query vector itself, 4/4")

# ── 5. exhaustive probe (nprobe=nlist) must equal exact brute force ───────
diff = con.execute(f"""
WITH a AS (SELECT id FROM ivf_search('idx', 'main.q', 'vec', top_k := {TOPK}, nprobe := {NLIST})),
     b AS (SELECT id FROM ivf_search_exact('idx', 'main.q', 'vec', top_k := {TOPK}))
SELECT count(*) FROM (
  (SELECT id FROM a EXCEPT SELECT id FROM b)
  UNION ALL
  (SELECT id FROM b EXCEPT SELECT id FROM a)
)
""").fetchall()[0][0]
assert diff == 0, f"exhaustive IVF probe disagrees with exact search on {diff} ids"
print(f"[ok] nprobe=nlist search set == ivf_search_exact set (top-{TOPK})")

# ── 6. regression: FLOAT[N] (ArrayVector) must be normalized, not segfault
con.execute(f"""
CREATE TABLE ta AS
SELECT r.i AS id, [r.i::FLOAT, (r.i+1)::FLOAT, (r.i+2)::FLOAT, (r.i+3)::FLOAT]::FLOAT[4] AS vec
FROM range(100) r(i)
""")
n2 = con.execute(
    "SELECT rows_indexed FROM ivf_create_index('idx_arr', 'main.ta', 'id', 'vec', 4, 4, true)"
).fetchall()[0][0]
assert n2 == 100
con.execute("CREATE TABLE qa AS SELECT [1.0, 2.0, 3.0, 4.0]::FLOAT[4] AS vec")
top = con.execute(
    "SELECT id FROM ivf_search('idx_arr', 'main.qa', 'vec', top_k := 1, nprobe := 4)"
).fetchall()[0][0]
assert top == 1, f"expected id 1 nearest to [1,2,3,4], got {top}"
print("[ok] FLOAT[4] fixed-array column: index + search (no segfault)")

# ── 7. error paths surface as DuckDB errors, not crashes ──────────────────
con.execute("CREATE TABLE qd AS SELECT [1.0, 2.0, 3.0] AS vec")
try:
    con.execute("SELECT id FROM ivf_search('idx', 'main.qd', 'vec', top_k := 5)").fetchall()
    raise AssertionError("dim mismatch did not error")
except duckdb.Error as e:
    assert "dim" in str(e), f"expected dim error, got: {e}"
try:
    con.execute("SELECT id FROM ivf_search('no_such_index', 'main.q', 'vec', top_k := 5)").fetchall()
    raise AssertionError("unknown index did not error")
except duckdb.Error as e:
    assert "no ivf index" in str(e), f"expected unknown-index error, got: {e}"
print("[ok] dim-mismatch + unknown-index surface as clean DuckDB errors")

print("\nALL e2e CHECKS PASSED")
