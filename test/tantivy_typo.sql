-- ============================================================
-- typo tolerance macro layer — TANTIVY backend (optional, works on
-- DuckDB 1.4.4 - 2.0.0). Sibling of fts_typo_v2.sql, which is the
-- fts/layered backend (DuckDB >= 2.0 only).
--
-- Backend-agnostic core (identical in both files): ivf_levenshtein
-- (the ivf extension), length-tiered edit budgets, synonyms, and
-- expand-not-replace fusion. The ONLY difference is where the
-- correction dictionary and the search come from:
--
--   fts backend (fts_typo_v2.sql):
--     dict  = fts_main_<T>.raw_dict          (built by the fts index)
--     search= fts_main_<T>.search_layered_bm25
--   tantivy backend (this file):
--     dict  = typo_dict  (user-materialized, see below)
--     search= bm25_search('<table>','<col>', ...) from duckdb-tantivy
--
-- SETUP (one-time, after the table exists):
--   INSTALL tantivy· LOAD tantivy·      -- or LOAD 'tantivy.duckdb_extension'
--   CREATE TABLE typo_dict AS
--     SELECT term, df FROM tantivy_terms('<table>', '<text_column>')·
--   LOAD 'ivf.duckdb_extension'·         -- ivf_levenshtein
--   CREATE TABLE synonyms (a VARCHAR, b VARCHAR)·  -- optional
--   -- then register the macros from this file (no placeholder to replace)
--
-- USAGE:
--   WITH c AS (SELECT typo_correct_query_t('cruide refinery') AS cq)
--   SELECT * FROM typo_search_t('cruide refinery', c.cq, 'news', 'body',
--                               top_k := 10);
--
-- SEMANTIC NOTES (honest differences vs the fts backend):
--   * The tantivy dict is lowercased surface forms (unstemmed) — the same
--     correction space as fts raw_dict· bm25_search stems the QUERY, so a
--     corrected surface form bridges back to the stemmed index automatically.
--   * bm25_search is AND semantics (every query word must match). That KILLS
--     the fts backend's expand-not-replace trick (original + corrected words
--     in one query string: the misspelled word matches nothing → 0 hits).
--     The equivalent here is two-arm RRF: the original query and the
--     corrected query run as SEPARATE bm25_search calls, fused by rank.
--   * tantivy_terms scans the table per materialization (in-RAM, no
--     persistent index) — refresh typo_dict when the table changes.
-- ============================================================

-- threshold by word length: <=3 never correct, 4..t_mid -> t_short, >t_mid -> t_long
CREATE OR REPLACE MACRO typo_lev_thresh_t(w, t_mid := 5, t_short := 1, t_long := 2) AS
  CASE WHEN length(w) <= 3 THEN 0
       WHEN length(w) <= t_mid THEN t_short
       ELSE t_long END;

-- explicit edit-distance tier to the closest dict term, or null if none within budget
CREATE OR REPLACE MACRO typo_word_tier_t(w, t_mid := 5, t_short := 1, t_long := 2) AS
  CASE
    WHEN w IS NULL OR length(w) <= 3 THEN NULL
    WHEN EXISTS (SELECT 1 FROM synonyms WHERE (a = lower(w) OR b = lower(w))) THEN NULL
    WHEN EXISTS (SELECT 1 FROM typo_dict WHERE term = lower(w)) THEN 0
    ELSE (SELECT ivf_levenshtein(lower(w), term)
          FROM typo_dict
          WHERE ivf_levenshtein(lower(w), term) <= typo_lev_thresh_t(w, t_mid, t_short, t_long)
          ORDER BY ivf_levenshtein(lower(w), term), df DESC LIMIT 1)
  END;

CREATE OR REPLACE MACRO typo_correct_word_t(w, t_mid := 5, t_short := 1, t_long := 2) AS
  CASE
    WHEN w IS NULL OR length(w) = 0 THEN w
    WHEN EXISTS (SELECT 1 FROM synonyms WHERE (a = lower(w) OR b = lower(w))) THEN
      COALESCE(
        (SELECT b FROM synonyms WHERE a = lower(w) ORDER BY rowid LIMIT 1),
        (SELECT a FROM synonyms WHERE b = lower(w) ORDER BY rowid LIMIT 1))
    WHEN EXISTS (SELECT 1 FROM typo_dict WHERE term = lower(w)) THEN w
    ELSE COALESCE(
      (SELECT term FROM typo_dict
       WHERE ivf_levenshtein(lower(w), term) <= typo_lev_thresh_t(w, t_mid, t_short, t_long)
       ORDER BY ivf_levenshtein(lower(w), term), df DESC LIMIT 1),
      w)
  END;

CREATE OR REPLACE MACRO typo_correct_query_t(q, t_mid := 5, t_short := 1, t_long := 2) AS
  (SELECT string_agg(typo_correct_word_t(word, t_mid, t_short, t_long), ' ')
   FROM (SELECT UNNEST(string_split(q, ' ')) AS word));

-- prefix expansion: dict terms sharing the 3-char prefix of w (top 5 by df),
-- excluding w itself. No trie needed — a real table + LIKE does it.
CREATE OR REPLACE MACRO typo_prefix_terms_t(w) AS
  (SELECT string_agg(term, ' ')
   FROM (SELECT term FROM typo_dict
         WHERE term LIKE substring(lower(w), 1, 3) || '%'
           AND lower(term) <> lower(w)
           AND length(term) > 3
         ORDER BY df DESC LIMIT 5) sel);

-- prefix-expanded query: original word + its prefix terms, per word
CREATE OR REPLACE MACRO typo_prefix_query_t(q) AS
  (SELECT string_agg(item, ' ')
   FROM (SELECT UNNEST(string_split(q, ' ')) AS word) w
   CROSS JOIN LATERAL (
     SELECT w.word AS item
     UNION ALL
     SELECT UNNEST(string_split(COALESCE(typo_prefix_terms_t(w.word), ''), ' ')) AS item
   ) ex
   WHERE item <> '');

-- RECOMMENDED ENTRY POINT (fts backend): original + corrected merged
CREATE OR REPLACE MACRO typo_expand_query_t(q, t_mid := 5, t_short := 1, t_long := 2) AS
  q || (CASE WHEN typo_correct_query_t(q, t_mid, t_short, t_long) = q THEN ''
             ELSE ' ' || typo_correct_query_t(q, t_mid, t_short, t_long) END);

-- ======================================================================
-- TANTIVY-BACKEND SEARCH: two-arm RRF over bm25_search
--   Arm 1: q_orig — the original (possibly misspelled) query
--   Arm 2: q_corr — the corrected query (only used when it differs from
--   q_orig; NULL or identical → single arm)
-- Each arm runs bm25_search independently (AND semantics per arm), and
-- document ids are fused by reciprocal rank fusion (k=60, Meilisearch's
-- default).
--
-- Why two string params instead of correcting inside? DuckDB TABLE macros
-- cannot call other macros whose body contains subqueries ("Table function
-- cannot contain subqueries") — and the word-correction chain needs
-- subqueries to scan the dictionary. So correction is computed AT THE CALL
-- SITE (top level / CTE, where subqueries are fine) and passed in as a
-- plain string. This is also the documented usage:
--
--   WITH c AS (SELECT typo_correct_query_t('cruide refinery') AS cq)
--   SELECT * FROM typo_search_t('cruide refinery', c.cq, 'news', 'body',
--                               top_k := 10)
--   → doc_id BIGINT, rrf_score DOUBLE, n_arms INTEGER
--   (TABLE macro: it returns rows, not a scalar — usable in FROM)
--   The table must have an id column (BIGINT) for bm25_search's id_column;
--   pass id_column := 'your_col' if it is not named 'id'.
-- ======================================================================
CREATE OR REPLACE MACRO typo_search_t(q_orig, q_corr, table_name, text_column,
                                      top_k := 10, id_column := 'id') AS TABLE
  SELECT doc_id, sum(1.0 / (60 + rank0)) AS rrf_score,
         count(*) AS n_arms
  FROM (
    SELECT doc_id, row_number() OVER (ORDER BY score DESC) - 1 AS rank0
    FROM bm25_search(table_name, q_orig, text_column := text_column,
                     id_column := id_column, top_k := top_k)
    UNION ALL
    SELECT doc_id, row_number() OVER (ORDER BY score DESC) - 1 AS rank0
    FROM bm25_search(table_name, q_corr, text_column := text_column,
                     id_column := id_column, top_k := top_k)
    WHERE q_corr IS NOT NULL AND q_corr <> q_orig
  ) arms
  GROUP BY doc_id
  ORDER BY rrf_score DESC, doc_id
  LIMIT top_k;
