-- ============================================================================
-- ztg_tantivy.sql — zg-clone hybrid search: BM25 arm on the duckdb-tantivy
-- persistent index. ONE file deploys, ONE call searches.
--
--   CREATE OR REPLACE TABLE qtab AS SELECT * FROM ztg_qvec_news('原油价格 上涨');
--   SELECT * FROM ztg_search_news_tantivy('原油价格 上涨', 'qtab');
--
-- Everything runs in-process — zero external services:
--   query text -> ml_embed_sentence (duckdb-ml)  -> IVF ANN (duckdb-ivf)      [vector arm]
--   query text -> tantivy BM25 (duckdb-tantivy persistent index)             [text arm]
--   arms fused by RRF in this macro -> top-k
--
-- vs ztg.sql (the fts-arm sibling) — three deliberate differences:
--   * the text arm is a PERSISTENT on-disk tantivy index (tantivy_create_index
--     / tantivy_load_index): survives process restarts, millisecond queries,
--     no re-index per connection;
--   * CHINESE: language:='chinese' is native jieba WORD-level segmentation
--     ('北京大学在北京市' -> ['北京大学','在','北京市']) and lowercases the
--     Latin parts of mixed text, so ONE index serves a zh/en-mixed corpus and
--     the RAW text column is indexed directly — NO jieba pre-tokenized column
--     is needed (the fts arm required one; see ztg.sql section 5);
--   * the RRF fusion is computed in THIS SQL macro (1/(k_rrf+rank) summed per
--     arm) — the ivf extension's internal fts-BM25 paths (ivf_hybrid_search /
--     ivf_hybrid_embed) are NOT used; the vector arm is plain ivf_search over
--     a one-row query-vector table.
--
-- Usage:
--   Connect with unsigned extensions allowed:
--     CLI:    duckdb -unsigned < ztg_tantivy.sql
--     Python: duckdb.connect(config={"allow_unsigned_extensions":"true"})
--   then execute this whole file (or source it). Done.
--
-- DEPLOYMENT: edit the literals marked (EDIT) in sections 1-6 — one time.
-- ============================================================================

-- ── 1. Extensions (EDIT: absolute paths to your builds) ─────────────────────
LOAD '/abs/path/to/ivf.duckdb_extension';
LOAD '/abs/path/to/ml.duckdb_extension';
LOAD '/abs/path/to/tantivy.duckdb_extension';
-- NOTE: no `INSTALL fts; LOAD fts;` anywhere — the text arm no longer touches fts.

-- ── 2. Embedding model (EDIT: model name + ONNX/tokenizer paths) ────────────
SELECT * FROM ml_load_sentence(
  'minilm', '/abs/path/to/model.onnx', '/abs/path/to/tokenizer.json', 128);

-- Query-vector convenience for the search call below. The vector arm is
-- ivf_search over a one-row table, and a table-function macro cannot CREATE
-- that table inside its own SELECT, so the caller materializes it first:
--   CREATE OR REPLACE TABLE qtab AS SELECT * FROM ztg_qvec_news('...');
CREATE OR REPLACE MACRO ztg_qvec_news(q) AS TABLE
  SELECT ml_embed_sentence('minilm', q) AS v;

-- ── 3. Corpus table (EDIT: your data; must be a PERMANENT table with
--      id BIGINT + text VARCHAR — both extensions read tables through their
--      own dedicated session connections: temp tables and uncommitted writes
--      are invisible to them)
-- For the ag_news PoC:
-- CREATE OR REPLACE TABLE news AS
--   SELECT row_number() OVER() AS id, text
--   FROM read_parquet('/abs/path/to/corpus.parquet');

-- ── 4. Vector index (EDIT: index name + .idx path) — unchanged from ztg.sql ─
-- 4a. FRESH TRAIN — once, on first deployment (in-process embedding of the
--     whole corpus; 5000 docs ≈ 50s, 120k ≈ 3h at CPU ORT batch=1):
-- CREATE OR REPLACE TABLE vecm AS
--   SELECT id, ml_embed_sentence('minilm', text)::FLOAT[] AS v FROM news;
-- SELECT * FROM ivf_create_index('newsm', 'vecm', 'id', 'v', 200, 10, true);
-- SELECT * FROM ivf_save('newsm', '/abs/path/to/news.idx');
-- DROP TABLE vecm;
-- 4b. LOAD pre-built index (deployment default; no retraining):
SELECT * FROM ivf_load('newsm', '/abs/path/to/news.idx');

-- ── 5. Text index (EDIT: name / table / columns / language / path) ──────────
-- Persistent on-disk index with rebuild semantics (the directory is wiped and
-- rewritten each time; default location $TANTIVY_INDEX_DIR/<name> or
-- .tantivy_index/<name> when path is left empty).
--
-- CHINESE (the active statement below): language:='chinese' = native jieba
-- word segmentation for CJK plus lowercased Latin tokens for the mixed parts.
-- The raw text column is indexed directly — no jieba pre-tokenized column,
-- no luajit tokenization step (the simplification vs the fts arm, see
-- ztg.sql section 5).
SELECT * FROM tantivy_create_index('newsm_t', 'news',
  id_column := 'id', text_column := 'text',
  language := 'chinese',
  path := '/abs/path/to/newsm_tantivy');
-- ENGLISH deployment variant (Snowball stemmer, word-level Latin tokens —
-- use for an English-only corpus; note: no stemmer bridges inflections the
-- way porter does, so choose per corpus):
-- SELECT * FROM tantivy_create_index('newsm_t', 'news',
--   id_column := 'id', text_column := 'text',
--   language := 'english',
--   path := '/abs/path/to/newsm_tantivy');
--
-- FRESH PROCESS (deployment default — no rebuild): load the index built
-- above. The __tantivy_meta.json sidecar records the build-time language and
-- the analyzer is re-registered from it, so the language must simply be
-- re-resolvable:
-- [LOAD-VARIANT] SELECT * FROM tantivy_load_index('newsm_t', '/abs/path/to/newsm_tantivy');

-- ── 6. THE single entry point (EDIT: the literals to your deployment) ───────
-- Arms:    BM25   = tantivy_search_index(persistent index, q)  -> (doc_id, score)
--          vector = ivf_search(ivf index, qtab.vcol)           -> (id, score)
-- Fusion:  RRF at THIS macro layer — rank within each arm's per_path list,
--          rrf_score = sum(1/(k_rrf + rank)) over the two arms, ties broken
--          by id for reproducible output. bm25_rank / vec_rank are NULL for
--          a doc missed by that arm (NULL = a pure hit from the other arm).
-- C-API table functions bind their arguments as constants, so the profile is
-- fixed in the macro: one macro per (index, table) deployment.
CREATE OR REPLACE MACRO ztg_search_news_tantivy(
  q, qtab, vcol := 'v', top_k := 10, per_path := 50, nprobe := 8, k_rrf := 60) AS TABLE
  WITH t AS (
      -- BM25 arm: the persistent index emits hits in final ranked order
      -- (tie-break included), so the row order IS the rank.
      SELECT doc_id AS id, row_number() OVER () AS trk
      FROM tantivy_search_index('newsm_t', q, top_k := per_path)),
       v AS (
      -- vector arm: ANN over the one-row query table (qtab.vcol).
      SELECT id, row_number() OVER (ORDER BY score DESC) AS vrk
      FROM ivf_search('newsm', qtab, vcol, top_k := per_path, nprobe := nprobe)),
       u AS (
      SELECT id, trk, NULL::BIGINT AS vrk FROM t
      UNION ALL
      SELECT id, NULL::BIGINT AS trk, vrk FROM v),
       rrf AS (
      -- max() recovers the per-arm rank (a doc ranks in each arm at most
      -- once; max over {value, NULL} = value, over {NULL} = NULL)
      SELECT id,
             sum(CASE WHEN trk IS NOT NULL THEN 1.0/(k_rrf + trk) ELSE 0 END)
         +   sum(CASE WHEN vrk IS NOT NULL THEN 1.0/(k_rrf + vrk) ELSE 0 END) AS rrf_score,
             max(trk) AS trk,
             max(vrk) AS vrk
      FROM u GROUP BY id)
  SELECT id, rrf_score AS score, trk AS bm25_rank, vrk AS vec_rank
  FROM rrf
  ORDER BY rrf_score DESC, id ASC
  LIMIT top_k;

-- ── Done. Search (two statements — see section 2 for why qtab is explicit):
--   CREATE OR REPLACE TABLE qtab AS SELECT * FROM ztg_qvec_news('原油价格 上涨');
--   SELECT * FROM ztg_search_news_tantivy('原油价格 上涨', 'qtab');
--   SELECT * FROM ztg_search_news_tantivy('原油价格 上涨', 'qtab', top_k := 20);
--
-- Known limitations:
--   * qtab must be a committed, permanent one-row table — both extensions
--     read tables through their own session connections (temp tables and
--     uncommitted writes are invisible to them);
--   * queries are passed as parameters, so quotes in the query text are safe
--     for both arms (unlike ztg.sql's embedded-expression workaround).
