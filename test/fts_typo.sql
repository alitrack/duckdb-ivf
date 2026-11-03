-- ============================================================
-- fts typo tolerance + synonyms — macro layer (Meilisearch-style)
-- DuckDB 2.0.0, layered fts + ivf_levenshtein (duckdb-ivf extension)
--
-- DESIGN (verified on WSL, 2026-11-03):
--   * correction runs on RAW terms (fts_main_<T>.raw_dict), because
--     search_layered_bm25 re-stems the query internally·
--   * length-tiered threshold keeps short words exact
--     (<=3: no typos, 4-5: 1, >=6: 2) — Meilisearch does the same,
--     otherwise "oil" would "correct" to a random 1-edit word·
--   * synonyms = user table synonyms(a,b) (exact, both directions),
--     applied BEFORE typo correction so curated pairs always win·
--   * unknown words pass through unchanged (never invent a fix).
--
-- PREREQUISITES for table <T> with fts index fts_main_<T>:
--   LOAD ivf.duckdb_extension·            -- provides ivf_levenshtein
--   CREATE OR REPLACE TABLE synonyms (a VARCHAR, b VARCHAR)·  -- optional
--
-- USAGE:
--   SELECT ... FROM fts_main_<T>.search_layered_bm25(
--     fts_correct_query_t('cruide refineri'), top_k := 10)·
-- ============================================================

CREATE OR REPLACE MACRO fts_lev_thresh_t(w) AS
  CASE WHEN length(w) <= 3 THEN 0 WHEN length(w) <= 5 THEN 1 ELSE 2 END;

CREATE OR REPLACE MACRO fts_correct_word_t(w) AS
  CASE
    WHEN w IS NULL OR length(w) = 0 THEN w
    -- synonym first (exact, both directions, case-insensitive)
    WHEN EXISTS (SELECT 1 FROM synonyms WHERE (a = lower(w) OR b = lower(w))) THEN
      COALESCE(
        (SELECT b FROM synonyms WHERE a = lower(w) ORDER BY rowid LIMIT 1),
        (SELECT a FROM synonyms WHERE b = lower(w) ORDER BY rowid LIMIT 1))
    -- exact dictionary hit: keep as-is (dictionary is lowercased by the
    -- fts tokenizer, so match on lower(w) — a capitalised query word must
    -- not count as a typo of its own lowercase form)
    WHEN EXISTS (SELECT 1 FROM fts_main_t.raw_dict WHERE raw_term = lower(w)) THEN w
    -- closest dictionary term within the length-tiered budget
    ELSE COALESCE(
      (SELECT raw_term FROM fts_main_t.raw_dict
       WHERE ivf_levenshtein(lower(w), raw_term) <= fts_lev_thresh_t(w)
       ORDER BY ivf_levenshtein(lower(w), raw_term), df DESC LIMIT 1),
      w)
  END;

CREATE OR REPLACE MACRO fts_correct_query_t(q) AS
  (SELECT string_agg(fts_correct_word_t(word), ' ')
   FROM (SELECT UNNEST(string_split(q, ' ')) AS word));

-- Recommended entry point: search BOTH the original and the corrected query.
-- The corrected copy rescues typos that BM25 misses· words the corrector
-- mangles stay harmless because the original copy still ranks normally and
-- RRF fusion combines both lists. (Eval v2: replacing the query outright
-- LOST nDCG on ag_news - see eval_typo2_result.json.)
CREATE OR REPLACE MACRO fts_expand_query_t(q) AS
  q || (CASE WHEN fts_correct_query_t(q) = q THEN ''
             ELSE ' ' || fts_correct_query_t(q) END);
