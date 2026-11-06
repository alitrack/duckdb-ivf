-- ============================================================================
-- ztg.sql — zg-clone hybrid search: ONE file deploys, ONE call searches.
--
--   SELECT * FROM ztg_search_news('crude oil prices');
--
-- Everything runs in-process — zero external services:
--   query text -> ml_embed_sentence (duckdb-ml)  -> IVF ANN (duckdb-ivf) [vector arm]
--   query text -> fts BM25 (duckdb fts)                               [text arm]
--   arms fused by RRF -> top-k
--
-- Usage:
--   Connect with unsigned extensions allowed:
--     CLI:    duckdb -unsigned < ztg.sql
--     Python: duckdb.connect(config={"allow_unsigned_extensions":"true"})
--   then execute this whole file (or source it). Done.
--
-- DEPLOYMENT: edit the literals marked (EDIT) in sections 1-6 — one time.
-- After that, the only thing a search caller writes is the SELECT below.
-- ============================================================================

-- ── 1. Extensions (EDIT: absolute paths to your builds) ─────────────────────
LOAD '/abs/path/to/ivf.duckdb_extension';
LOAD '/abs/path/to/ml.duckdb_extension';
INSTALL fts; LOAD fts;

-- ── 2. Embedding model (EDIT: model name + ONNX/tokenizer paths) ────────────
SELECT * FROM ml_load_sentence(
  'minilm', '/abs/path/to/model.onnx', '/abs/path/to/tokenizer.json', 128);

-- ── 3. Corpus table (EDIT: your data; must be a PERMANENT table with
--      id BIGINT + text VARCHAR — the C-API extension cannot see temp tables)
-- For the ag_news PoC:
-- CREATE OR REPLACE TABLE news AS
--   SELECT row_number() OVER() AS id, text
--   FROM read_parquet('/abs/path/to/corpus.parquet');

-- ── 4. Vector index (EDIT: index name + .idx path) ──────────────────────────
-- 4a. FRESH TRAIN — once, on first deployment (in-process embedding of the
--     whole corpus; 5000 docs ≈ 50s, 120k ≈ 3h at CPU ORT batch=1):
-- CREATE OR REPLACE TABLE vecm AS
--   SELECT id, ml_embed_sentence('minilm', text)::FLOAT[] AS v FROM news;
-- SELECT * FROM ivf_create_index('newsm', 'vecm', 'id', 'v', 200, 10, true);
-- SELECT * FROM ivf_save('newsm', '/abs/path/to/news.idx');
-- DROP TABLE vecm;
-- 4b. LOAD pre-built index (deployment default; no retraining):
SELECT * FROM ivf_load('newsm', '/abs/path/to/news.idx');

-- ── 5. FTS index (EDIT: table / id / text columns).
--      Skip if the index already exists in a persistent database file. ───────
PRAGMA create_fts_index('news', 'id', 'text',
  stemmer='porter', stopwords='english', layered_search=true);

-- ── 6. THE single entry point (EDIT: the 5 literals to your deployment) ─────
-- C-API table functions bind their arguments as constants, so the profile is
-- fixed in the macro: one macro per (index, table) deployment. Different
-- dataset => different macro name (ztg_search_docs, ...).
CREATE OR REPLACE MACRO ztg_search_news(q, top_k := 10) AS TABLE
  SELECT id, score, bm25_rank, vec_rank
  FROM ivf_hybrid_embed(
    'newsm', 'news', 'id', 'text', q,
    concat('ml_embed_sentence(', chr(39), 'minilm', chr(39), ', ', chr(39), q, chr(39), ')'),
    top_k := top_k, per_path := 50, nprobe := 8, k_rrf := 60);

-- ── Done. Search: ────────────────────────────────────────────────────────────
--   SELECT * FROM ztg_search_news('crude oil prices');
--   SELECT * FROM ztg_search_news('crude oil prices', 20);
--
-- Known limitation: query text containing a single quote will not round-trip
-- (the embedded expression is built via chr(39) concatenation and is not
-- quote-escaped — a DuckDB table-function-macro limitation).
