# duckdb-ivf

DuckDB loadable extension: **IVF-Flat approximate nearest-neighbor vector search** in pure Rust.

This is the (b) deliverable of the SereneDB feasibility study (`D:\wsl2\tmp\fts_ivf_poc`).
The study's conclusion: DuckDB's `fts` + `ivf` extensions + SQL-layer RRF hybrid can reproduce
~80–90% of SereneDB's *functional* surface. The one thing DuckDB **core genuinely lacks** is a
vector **index** — core's `list_distance` is brute-force kNN, and this build has no HNSW/IVF/VSS
index type. `duckdb-ivf` fills exactly that gap: a real, in-DB ANN index you create and query from SQL.

> Honest framing: this is a **reusable skeleton with a working data plane**, not a production
> vector engine. Training is single-threaded k-means (no SIMD, no parallel build).
> The index **can** be persisted to a file and reloaded without retraining
> (`ivf_save` / `ivf_load`). See [Limitations](#limitations-be-honest-about-these).

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
| `ivf_create_index(name, table, id_column, vector_column, nlist, niter, normalize)` | table | Train + register an IVF index over a table's vector column. Returns 1 row `(rows_indexed BIGINT)`. |
| `ivf_search(name, query_table, query_column, top_k := 10, nprobe := 0)` | table | Approximate search. Returns `(id BIGINT, score DOUBLE)` for the top-k. |
| `ivf_search_exact(name, query_table, query_column, top_k := 10)` | table | Exhaustive ground-truth search (brute force over all `n`). |
| `ivf_hybrid_search(name, table, id_column, text_column, query, query_table, query_column, top_k := 10, per_path := 50, nprobe := 0, k_rrf := 60)` | table | **Hybrid** FTS BM25 (`fts` extension, `fts_main_<table>.search_layered_bm25`) + IVF vector, fused by Reciprocal Rank Fusion; ties broken by `id` for reproducible results. Returns `(id BIGINT, score DOUBLE, bm25_rank INT, vec_rank INT)`. `id_column` may be empty (positions are used). The query vector comes from a 1-row `query_table.query_column`. |
| `ivf_hybrid_embed(name, table, id_column, text_column, query, embed_expr, top_k := 10, per_path := 50, nprobe := 0, k_rrf := 60)` | table | **Single-call, text-in** hybrid search: same RRF fusion as `ivf_hybrid_search`, but the query vector is embedded **in-process** from `query` via `embed_expr` — an arbitrary scalar SQL expression returning a `LIST`/`ARRAY`, e.g. `ml_embed_sentence('minilm','crude oil')` (duckdb-ml). No precomputed query table needed. Same result columns. |
| `ivf_save(name, path)` | table | Serialize a registered index to a file (vectors + centroids + list assignment + id map). Returns `(name, dim, n, nlist, bytes, message)`. |
| `ivf_load(name, path)` | table | Rebuild an index from a file written by `ivf_save` **without retraining** (lists are deterministically rebuilt from the assignment). Returns the same columns as `ivf_save`; `message = 'ok (no retrain)'`. |
| `ivf_list()` | table | List registered indexes: `(name, dim, n, nlist)`. |
| `ivf_version()` | scalar | Extension version. |
| `ivf_levenshtein(a, b)` / `(a, b, maxd)` | scalar | OSA Damerau-Levenshtein edit distance (transposition = 1), saturated at 3 / `maxd`. Building block for the fts typo-tolerance macro layer, see [Typo tolerance](#typo-tolerance--synonyms-meilisearch-style-on-top-of-fts). |

- `nlist = 0` → default `sqrt(n)`, clamped `[1, 4096]`.
- `niter = 0` → 16 k-means iterations.
- `normalize = true` → L2-normalize vectors + centroids (cosine / inner-product semantics).
- `nprobe = 0` (or negative) → probe all `nlist` lists (exhaustive; recall = exact).

### Usage

```sql
LOAD 'ivf.duckdb_extension';

-- Index the vector column (a LIST(FLOAT) or a fixed ARRAY column; the latter is
-- auto-normalized to a LIST inside the extension).
SELECT * FROM ivf_create_index('news', 'main.news_vec', 'id', 'vec', 350, 10, true);

-- The query vector is a 1-row table the extension can read (see the
-- "session connection" note below).
CREATE TABLE q1 AS SELECT [0.1, 0.2, ...] AS vec;   -- 1024 floats

SELECT id, score FROM ivf_search('news', 'q1', 'vec', top_k := 10, nprobe := 32);
```

### Persistence (no retraining)

```sql
-- After training once (one-off ~157s for 120k × 1024):
SELECT * FROM ivf_save('news', '/path/to/news.ivf.idx');   -- 1 row, message = 'ok'

-- Later, in a FRESH process — no k-means, ~1.1s, bit-identical search results:
SELECT * FROM ivf_load('news', '/path/to/news.ivf.idx');   -- message = 'ok (no retrain)'
SELECT id, score FROM ivf_search('news', 'q1', 'vec', top_k := 10, nprobe := 8);
```
File format (v1, little-endian, magic `0x4346_5649`/`'IVFC'`):
header (`magic`, `version`, `dim`, `nlist`, `n`) + `data` (n×dim f32) +
`centroids` (nlist×dim f32) + `list_of` (n u32) + `ids` (n i64). The inverted
`lists` are rebuilt deterministically from `list_of`, so a loaded index searches
**bit-identical** to the trained one — verified top-10 id-for-id.

### Hybrid search (BM25 + vector, RRF fusion)

```sql
-- Table must have an FTS index (fts extension, layered_search=true):
PRAGMA create_fts_index('news', 'id', 'text', stemmer := 'porter',
                        stopwords := 'english', layered_search := true);

-- query_table.query_column holds the query's embedding (1-row table); `query`
-- is the raw text for the BM25 arm. RRF fuses both arms; ties broken by id.
SELECT id, score, bm25_rank, vec_rank
FROM ivf_hybrid_search('news', 'news', 'id', 'text',
                       'crude oil prices', 'q1', 'vec',
                       top_k := 10, per_path := 50, nprobe := 8, k_rrf := 60);
```
- `per_path` = how many docs each arm feeds into the fusion (default 50).
- `nprobe` = IVF probing for the vector arm (0 = exhaustive).
- `k_rrf` = RRF constant (60 is the Meilisearch convention).

### Single-call text-in search (no precomputed vector)

`ivf_hybrid_embed` removes the last external step — you hand it the raw query
text and an in-process embedding expression; the vector arm is computed
in-process (duckdb-ml), so no `query_table` and no network embedding call:

```sql
-- corpus vectors were also embedded in-process (ml_embed_sentence) + indexed;
-- now a query is ONE call, text in, top-k out:
SELECT id, score, bm25_rank, vec_rank
FROM ivf_hybrid_embed('news', 'news', 'id', 'text', 'crude oil prices',
                      "ml_embed_sentence('minilm', 'crude oil prices')",
                      top_k := 10, per_path := 50, nprobe := 8, k_rrf := 60);
```

- `embed_expr` is any scalar SQL expression returning a `LIST`/`ARRAY` of the
  same dimension as the index; the extension evaluates it on its session
  connection (so the embedding model must already be `ml_load_sentence`'d).
- Verified: `ivf_hybrid_embed` (text in) == `ivf_hybrid_search` (qtab) == a
  hand-written SQL RRF baseline, id-for-id on 3 queries (fresh process,
  `ivf_load` no-retrain, ~45 ms/query).
- If the table has no FTS index, the text arm degrades gracefully to
  vector-only (RRF just scores the vector arm) instead of erroring.

### `ztg` single entry point (text in, top-k out)

For the zg-style *one call* experience, `ztg/ztg.sql` is **one file that deploys
and searches** — no shell script, no separate deploy step. Edit the literals
marked `(EDIT)` (extension paths, model paths, corpus, index path, profile)
once, execute the whole file, and the only thing a search caller writes is:

```sql
SELECT * FROM ztg_search_news('crude oil prices');            -- top_k := 10
SELECT * FROM ztg_search_news('crude oil prices', 20);        -- explicit top_k
```

The file contains, in order: LOAD ivf/ml/fts → `ml_load_sentence` → corpus
table → `ivf_load` (with the fresh-train variant commented alongside) →
`create_fts_index` → the `ztg_search_*` macro. Connect with unsigned
extensions allowed. Verified: full file deploys in ~1s on the 5k PoC corpus,
3/3 queries == `ivf_hybrid_embed`.

**Design constraint found while building this**: a C-API table function binds
its arguments as *constants* — subqueries/macros-in-args raise
`Table function cannot contain subqueries`, so the profile (index/table/model)
is baked into one macro per deployment (one index/table ⇒ one macro name).
Quotes in the embedded expression are built with `chr(39)` because `''`
escapes are swallowed inside table-function macros. Known limitation: query
text containing `'` does not round-trip.

### Chinese (中文)

Verified end-to-end (400-doc Chinese corpus, 8 topics):

- **Native fts tokenizer is character-level for CJK** — `原油价格上涨`
  tokenizes to `原|油|价|格|上|涨`. It "works" (returns hits) but without word
  semantics, so BM25 ranking quality degrades.
- **Fix: jieba pre-tokenization** (one lib in
  `alitrack/duckdb-luajit-libs`, `luajit_module(mode:='install', sql_name:='jieba')`):
  index the tokenized column and run every query through the same pass
  (`jieba('北京大学在北京市') → 北京大学|在|北京市`). With jieba in the fts
  arm, the full hybrid (jieba-BM25 + in-process vector, RRF) returns
  topic-correct top-5 on Chinese queries — verified with 4 topic queries,
  3–4/5 correct each.
- The **vector arm is language-agnostic only as far as the model is**:
  MiniLM (English) is weak on Chinese; a Chinese-capable embedding model
  (e.g. bge-m3, which is multilingual) is the right choice for the vector arm
  of a Chinese deployment. Tokenizer fix + model choice are the two knobs.

### Important: the session connection

A C-API extension **cannot see the caller's connection**. `duckdb-ivf` opens its own
*session connection* (bound to the first database that `LOAD`s it) to read the indexed
table and the query table. Consequences:

- The indexed table and the query table must be **permanent tables** (the session
  connection cannot see the caller's temp tables or `register()`ed in-memory views).
- The index lives in a process-global registry for the process lifetime, addressed by
  `name`. It is **not** persisted across DuckDB restarts automatically — use
  `ivf_save(name, path)` / `ivf_load(name, path)` to persist and reload (no retraining).
- Multi-database / re-entrant use is guarded (a `ReentryMark` prevents a table
  callback from running a nested query on the very connection it's scanning).

## Build

```bash
# C-API extension: quack-rs 0.18.0 + libduckdb-sys (loadable).
# Build on the Linux toolchain — on a Windows host `cargo build` produces a
# .dll and pollutes the shared target/ dir.
CARGO_HOME=/home/lhy/.cargo make release
# → build/release/ivf.duckdb_extension
```

`make release` uses the official `extension-ci-tools` Makefile, which compiles
with the DuckDB-1.2-ABI Rust toolchain and stamps the proper metadata footer via
`append_extension_metadata.py` (534-byte footer; DuckDB 2.0 requires it on
loadables). No manual footer hack needed. Then
`SET allow_unsigned_extensions=1; LOAD 'ivf.duckdb_extension';`.

## Verified (ag_news, bge-m3 120,000 × 1024, L2-normalized)

`test_e2e.py` loads the extension into DuckDB 2.0.0 and (latest run, `e2e_final.out`):

- `ivf_create_index` trains an IVF index over all 120,000 vectors — **no crash** (157s,
  single-threaded k-means, nlist=350, niter=10).
- `ivf_list()` → `(news, 1024, 120000, 350)`.
- `ivf_search` recall@1/5/10 vs a numpy brute-force ground truth = **1.000 / 1.000 / 1.000**.
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

## E2E nDCG@10 (hybrid vs ANN, 20 queries)

`ztg_poc/ndcg_e2e.py` / `ndcg_real.py` — full-stack ranking quality on 20 query
phrases, ground truth = exact scoring (vector: `ivf_search_exact` brute force;
hybrid: RRF of exact-vector top-50 + FTS BM25 top-50). Under test: ANN
nprobe=8 / hybrid RRF nprobe=8.

| Query vectors | ANN (nprobe=8) | Hybrid (RRF) |
|---|---:|---:|
| Constructed (mean of BM25 top-5 doc embeddings) | 0.9337 | 0.9543 |
| **Real bge-m3 phrase embeddings** (LAN service) | **0.8491** | **0.8720** |

The real-embedding row is the honest one: constructed query vectors are derived
from the corpus itself, so they align with the index (optimistic). With true
query phrases, IVF nprobe=8's approximation error shows (0.85, with the worst
single query at 0.15), and — the point of the hybrid design — **hybrid beats
pure ANN on real queries** (0.87 vs 0.85; 13/20 queries ≥ ANN), e.g.
`corporate merger deal` where ANN nDCG is 0.152 but the BM25 arm lifts the
fusion to 0.417.

## Typo tolerance + synonyms (Meilisearch-style, on top of `fts`)

`ivf_levenshtein(a, b[, maxd])` — scalar, **OSA Damerau-Levenshtein** (an
adjacent transposition costs 1, the dominant human typo class — `sahcin`→
`sachin`), banded with max-distance early exit, UTF-8.

`test/fts_typo.sql` (v1, **recommended**) is a SQL macro layer that turns any
layered-`fts` index into a typo-tolerant search. For table `T` with fts schema
`fts_main_<T>`:

```sql
LOAD 'ivf.duckdb_extension';
CREATE TABLE synonyms (a VARCHAR, b VARCHAR);  -- optional, exact, both directions
-- register the macros from test/fts_typo.sql (they reference fts_main_<T>)
SELECT * FROM fts_main_<T>.search_layered_bm25(fts_expand_query_t('cruide refinery'), top_k := 10);
```

How it works: per query word — (1) exact `synonyms` pair wins, (2) exact
dictionary hit passes through, (3) otherwise the closest `fts_main_<T>.raw_dict`
term within a **length-tiered edit budget** (≤3 letters: never; 4-5: 1 edit;
≥6: 2 edits — short words must not be "corrected" into random 1-edit
lookalikes). Matching is lowercase-aware. The entry point is
**expand-not-replace**: original query + corrected query searched together, so
a wrong correction can only lose, never destroy the original ranking.

`test/fts_typo_v2.sql` adds experimental arms, **measured and found not to
beat v1 on ag_news** (120k, bge-m3, 60 short queries, two scenarios):

| feature | verdict (nDCG@10) |
|---|---|
| configurable thresholds `t_mid/t_short/t_long` | additive, back-compatible — keep |
| explicit edit tiers `fts_word_tier_t` (0/1/2) | tier-weighted RRF ≈ v1, no gain |
| prefix expansion (3-char trie, top-5 by df) | **net negative** in both 1-typo (0.827→0.822) and truncated-word (0.766→0.757) scenarios: strong for high-distinctiveness truncations (`Namibia`→`Nam` 0.000→0.643), diluted for common words (`Profit`→`Pro`) by the df-ranked expansion |

So v1's `fts_expand_query_t` remains the default; the v2 macros are provided
for callers who want the knobs (e.g. an autocomplete UI can weight the prefix
arm itself).

Verified numbers (ag_news, real runs, `D:\wsl2\tmp\fts_vss_poc\eval_*.json`):
50 short queries each with 1 transposition/substitution typo —
`expand` 0.644 → **0.705** nDCG@10 (OSA upgrade), `expand+vec` 0.889 → 0.909
vs `nofix+vec` 0.892. Honest ceiling: correction recall is 13/50 — words the
corpus never contained (proper names, new words) cannot be corrected by any
dictionary-based scheme; the corpus's `raw_dict` is 61k terms.

## Two real bugs found and fixed while making this work

1. **Array vs LIST segfault.** `FLOAT[1024]` is a fixed-size **ARRAY** (`ArrayVector` in the
   C API), not a LIST. Calling `duckdb_list_vector_get_size` on an array vector segfaults.
   Fix: the extension normalizes the vector column with `vec_col::FLOAT[]` (→ variable-length
   LIST) in its nested query, so both array and list input columns read correctly.
2. **k-means centroid buffer.** `centroids` was allocated as `nlist` floats instead of
   `nlist * dim` → out-of-bounds on any non-trivial build. Fixed to `vec![0.0; nlist*dim]`.

(An earlier design had `ivf_create_index` as a *scalar* function; a side-effectful
index-building function must be a **table function** so the optimizer doesn't treat it as a
pure constant-foldable scalar. It is now a table function.)

## Limitations (be honest about these)

- **Training is single-threaded** k-means: ~157 s for 120k × 1024 with nlist=350, niter=10.
  No SIMD, no parallel build, no incremental updates. `ivf_save`/`ivf_load` makes the
  cost one-off: save once (~1 s, ~494 MB for 120k × 1024), reload in fresh processes in
  ~1.1 s with bit-identical search results (verified top-10 id-for-id).
- **nprobe=350 is exhaustive**, so default recall = exact; you must set a small `nprobe`
  to get the sublinear speedup (see the sweep).
- **Memory**: stores all `n` vectors in a flat `Vec<f32>` (120k × 1024 × 4 B ≈ 480 MB).
- IVF recall degrades for very small `nprobe` on high-dim data; tune `nlist`/`nprobe`.

## Files

- `src/ivf.rs` — pure-Rust IVF engine (k-means + search + `load_from` for no-retrain
  rebuild), zero deps.
- `src/lib.rs` — quack-rs C-API binding: 9 table/scalar functions (create/search/
  search_exact/hybrid_search/hybrid_embed/save/load/list/version), session connection,
  LIST reader, RRF fusion.
- `test_e2e.py` — acceptance test on real ag_news bge-m3 embeddings
  (recall@k, nprobe sweep, e2e nDCG@10 via `ztg_poc/ndcg_e2e.py`,
  real-query-vector variant in `ztg_poc/ndcg_real.py`).
- `ztg/ztg.sql` — single file: deploys + searches (`ztg_search_*`, text in → top-k).
- `Makefile`, `Cargo.toml` — C-API extension build (tiktoken skeleton, zero new deps).
