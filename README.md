# duckdb-vss

DuckDB loadable extension: **IVF-Flat approximate nearest-neighbor vector search** in pure Rust.

This is the (b) deliverable of the SereneDB feasibility study (`D:\wsl2\tmp\fts_vss_poc`).
The study's conclusion: DuckDB's `fts` + `vss` extensions + SQL-layer RRF hybrid can reproduce
~80–90% of SereneDB's *functional* surface. The one thing DuckDB **core genuinely lacks** is a
vector **index** — core's `list_distance` is brute-force kNN, and this build has no HNSW/IVF/VSS
index type. `duckdb-vss` fills exactly that gap: a real, in-DB ANN index you create and query from SQL.

> Honest framing: this is a **reusable skeleton with a working data plane**, not a production
> vector engine. Training is single-threaded k-means (no SIMD, no persistence, no parallel build).
> See [Limitations](#limitations).

## Why an IVF index and not HNSW

- **IVF-Flat** (coarse k-means centroids + inverted lists) is the simplest index that is *actually
  sublinear*: at query time you probe `nprobe` of `nlist` lists instead of scanning all `n` vectors.
  It needs no external dependency (we implement k-means ourselves with a deterministic SplitMix64
  PRNG) → **builds fully offline**, which matters in this environment (WSL→pypi times out).
- **HNSW** would give better recall/time, but there's no offline-cached Rust HNSW crate available
  here (`usearch`/`hnswlib` are not in the cargo cache), so it would force a network build. IVF
  keeps the skeleton dependency-free and honest.

## Functions

| Function | Kind | Description |
|---|---|---|
| `vss_create_index(name, table, id_column, vector_column, nlist, niter, normalize)` | table | Train + register an IVF index over a table's vector column. Returns 1 row `(rows_indexed BIGINT)`. |
| `vss_search(name, query_table, query_column, top_k := 10, nprobe := 0)` | table | Approximate search. Returns `(id BIGINT, score DOUBLE)` for the top-k. |
| `vss_search_exact(name, query_table, query_column, top_k := 10)` | table | Exhaustive ground-truth search (brute force over all `n`). |
| `vss_list()` | table | List registered indexes: `(name, dim, n, nlist)`. |
| `vss_version()` | scalar | Extension version. |

- `nlist = 0` → default `sqrt(n)`, clamped `[1, 4096]`.
- `niter = 0` → 16 k-means iterations.
- `normalize = true` → L2-normalize vectors + centroids (cosine / inner-product semantics).
- `nprobe = 0` (or negative) → probe all `nlist` lists (exhaustive; recall = exact).

### Usage

```sql
LOAD 'vss.duckdb_extension';

-- Index the vector column (a LIST(FLOAT) or a fixed ARRAY column; the latter is
-- auto-normalized to a LIST inside the extension).
SELECT * FROM vss_create_index('news', 'main.news_vec', 'id', 'vec', 350, 10, true);

-- The query vector is a 1-row table the extension can read (see the
-- "session connection" note below).
CREATE TABLE q1 AS SELECT [0.1, 0.2, ...] AS vec;   -- 1024 floats

SELECT id, score FROM vss_search('news', 'q1', 'vec', top_k := 10, nprobe := 32);
```

### Important: the session connection

A C-API extension **cannot see the caller's connection**. `duckdb-vss` opens its own
*session connection* (bound to the first database that `LOAD`s it) to read the indexed
table and the query table. Consequences:

- The indexed table and the query table must be **permanent tables** (the session
  connection cannot see the caller's temp tables or `register()`ed in-memory views).
- The index lives in a process-global registry for the process lifetime, addressed by
  `name`. It is not persisted across DuckDB restarts (re-run `vss_create_index`).
- Multi-database / re-entrant use is guarded (a `ReentryMark` prevents a table
  callback from running a nested query on the very connection it's scanning).

## Build

```bash
# C-API extension: quack-rs 0.18.0 + libduckdb-sys (loadable). All deps offline-cached.
CARGO_TARGET_DIR=/home/lhy/vss-target cargo build --release --lib --offline
# → target/release/libvss.so
```

DuckDB 2.0 requires a 512-byte metadata footer on loadables. Stamp the `.so`:

```python
def stamp(so, dst, ext="v0.1.0", db="v1.2.0", plat="linux_amd64"):
    data = open(so, "rb").read()
    if data[-512:-480].split(b"\x00")[0] == b"4":
        data = data[:-512]
    f = b"".join(s.encode().ljust(32, b"\x00") for s in
                 ["", "", "", "C_STRUCT", ext, db, plat, "4"])
    open(dst, "wb").write(data + f.ljust(512, b"\x00"))
```

Then `SET allow_unsigned_extensions=1; LOAD 'vss.duckdb_extension';`.

## Verified (ag_news, bge-m3 120,000 × 1024, L2-normalized)

`test_e2e.py` loads the extension into DuckDB 2.0.0 and (latest run, `e2e_final.out`):

- `vss_create_index` trains an IVF index over all 120,000 vectors — **no crash** (157s,
  single-threaded k-means, nlist=350, niter=10).
- `vss_list()` → `(news, 1024, 120000, 350)`.
- `vss_search` recall@1/5/10 vs a numpy brute-force ground truth = **1.000 / 1.000 / 1.000**.
- **nprobe sweep** (the proof it's a real ANN, not a brute-force wrapper — core
  `list_distance` always scans all `n` and has no equivalent):

  | nprobe | recall@1 | recall@10 | median_ms | fraction scanned |
  |---:|---:|---:|---:|---|
  | 8   | 1.0000 | 1.0000 | **5.68**  | ~2.3% |
  | 16  | 1.0000 | 1.0000 | 10.34 | ~4.6% |
  | 32  | 1.0000 | 1.0000 | 14.70 | ~9% |
  | 64  | 1.0000 | 1.0000 | 25.05 | ~18% |
  | 128 | 1.0000 | 1.0000 | 36.68 | ~37% |
  | 350 | 1.0000 | 1.0000 | 82.69 | 100% |

  nprobe 350→8 is a **14.5× speedup** (82.69ms→5.68ms) at unchanged recall@10.
  Search time scales with `nprobe` (sublinear in the dataset), which is the property
  core's brute-force `list_distance` structurally lacks (always O(n), no knob).

**Honest caveats on these numbers:**
- recall@10 stays 1.0000 even at nprobe=8 *because of the eval design* — queries are the
  docs' own embeddings, so the ground-truth top-10 are the doc itself + near-duplicates that
  cluster tightly; even 8 lists catch them. Real unseen-text queries would degrade recall at
  low nprobe. Same caveat as the (a) evaluation.
- This does **not** claim IVF beats core's vectorized brute-force at 120k — that's a separate
  performance question a single-threaded skeleton isn't positioned to answer. The claim is:
  the *tunable (speed, recall) knob* exists in-DB, and core has no equivalent.

## Two real bugs found and fixed while making this work

1. **Array vs LIST segfault.** `FLOAT[1024]` is a fixed-size **ARRAY** (`ArrayVector` in the
   C API), not a LIST. Calling `duckdb_list_vector_get_size` on an array vector segfaults.
   Fix: the extension normalizes the vector column with `vec_col::FLOAT[]` (→ variable-length
   LIST) in its nested query, so both array and list input columns read correctly.
2. **k-means centroid buffer.** `centroids` was allocated as `nlist` floats instead of
   `nlist * dim` → out-of-bounds on any non-trivial build. Fixed to `vec![0.0; nlist*dim]`.

(An earlier design had `vss_create_index` as a *scalar* function; a side-effectful
index-building function must be a **table function** so the optimizer doesn't treat it as a
pure constant-foldable scalar. It is now a table function.)

## Limitations (be honest about these)

- **Training is single-threaded** k-means: ~157 s for 120k × 1024 with nlist=350, niter=10.
  No SIMD, no parallel build, no incremental updates.
- **No persistence**: the index is in-memory for the process lifetime.
- **nprobe=350 is exhaustive**, so default recall = exact; you must set a small `nprobe`
  to get the sublinear speedup (see the sweep).
- **Memory**: stores all `n` vectors in a flat `Vec<f32>` (120k × 1024 × 4 B ≈ 480 MB).
- IVF recall degrades for very small `nprobe` on high-dim data; tune `nlist`/`nprobe`.

## Files

- `src/vss.rs` — pure-Rust IVF engine (k-means + search), zero deps.
- `src/lib.rs` — quack-rs C-API binding: 5 functions, session connection, LIST reader.
- `test_e2e.py` — acceptance test on real ag_news bge-m3 embeddings.
- `Makefile`, `Cargo.toml` — C-API extension build (tiktoken skeleton, zero new deps).
