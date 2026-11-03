-- ============================================================
-- fts typo tolerance + synonyms + prefix — macro layer v2
-- DuckDB 2.0.0, layered fts + ivf_levenshtein (duckdb-ivf)
--
-- v2 over v1 (a9f0235):
--   * configurable thresholds (t_mid/t_short/t_long) threaded through
--   * fts_word_tier_t(w): EXPLICIT edit-distance tier 0/1/2/null
--   * prefix expansion: short query words expand to dict terms that
--     share their 3-char prefix (the fts trie stores only 2/3-char
--     prefixes, so prefix search is exact-on-3-char then filtered)
--   * weighted RRF helper for tier-aware fusion
--
-- MEASURED (ag_news 120k, bge-m3, 60 short queries, two scenarios): the v2
-- arms do NOT beat v1's fts_expand_query_t default — tier-weighted RRF ties,
-- prefix arm net negative (diluted for common words, strong only for
-- high-distinctiveness truncations like Namibia->Nam). v1 stays the
-- recommended default; use the v2 macros for the knobs (see README).
-- ============================================================

-- threshold by word length: <=3 never correct· 4..t_mid -> t_short· >t_mid -> t_long
CREATE OR REPLACE MACRO fts_lev_thresh_t(w, t_mid := 5, t_short := 1, t_long := 2) AS
  CASE WHEN length(w) <= 3 THEN 0
       WHEN length(w) <= t_mid THEN t_short
       ELSE t_long END;

-- explicit edit-distance tier to the closest dict term, or null if none within budget
CREATE OR REPLACE MACRO fts_word_tier_t(w, t_mid := 5, t_short := 1, t_long := 2) AS
  CASE
    WHEN w IS NULL OR length(w) <= 3 THEN NULL
    WHEN EXISTS (SELECT 1 FROM synonyms WHERE (a = lower(w) OR b = lower(w))) THEN NULL
    WHEN EXISTS (SELECT 1 FROM fts_main_t.raw_dict WHERE raw_term = lower(w)) THEN 0
    ELSE (SELECT ivf_levenshtein(lower(w), raw_term)
          FROM fts_main_t.raw_dict
          WHERE ivf_levenshtein(lower(w), raw_term) <= fts_lev_thresh_t(w, t_mid, t_short, t_long)
          ORDER BY ivf_levenshtein(lower(w), raw_term), df DESC LIMIT 1)
  END;

CREATE OR REPLACE MACRO fts_correct_word_t(w, t_mid := 5, t_short := 1, t_long := 2) AS
  CASE
    WHEN w IS NULL OR length(w) = 0 THEN w
    WHEN EXISTS (SELECT 1 FROM synonyms WHERE (a = lower(w) OR b = lower(w))) THEN
      COALESCE(
        (SELECT b FROM synonyms WHERE a = lower(w) ORDER BY rowid LIMIT 1),
        (SELECT a FROM synonyms WHERE b = lower(w) ORDER BY rowid LIMIT 1))
    WHEN EXISTS (SELECT 1 FROM fts_main_t.raw_dict WHERE raw_term = lower(w)) THEN w
    ELSE COALESCE(
      (SELECT raw_term FROM fts_main_t.raw_dict
       WHERE ivf_levenshtein(lower(w), raw_term) <= fts_lev_thresh_t(w, t_mid, t_short, t_long)
       ORDER BY ivf_levenshtein(lower(w), raw_term), df DESC LIMIT 1),
      w)
  END;

CREATE OR REPLACE MACRO fts_correct_query_t(q, t_mid := 5, t_short := 1, t_long := 2) AS
  (SELECT string_agg(fts_correct_word_t(word, t_mid, t_short, t_long), ' ')
   FROM (SELECT UNNEST(string_split(q, ' ')) AS word));

-- prefix expansion: dict terms sharing the 3-char prefix of w (top 5 by df),
-- excluding w itself. Only meaningful for short-ish query words· for full
-- words it mostly re-finds derivatives (crude -> crudes/crudely).
CREATE OR REPLACE MACRO fts_prefix_terms_t(w) AS
  (SELECT string_agg(raw_term, ' ')
   FROM (SELECT rd.raw_term
         FROM fts_main_t.term_prefixes tp
         JOIN fts_main_t.raw_dict rd ON tp.rawtermid = rd.rawtermid
         WHERE tp.prefix = substring(lower(w), 1, 3)
           AND lower(rd.raw_term) <> lower(w)
           AND length(rd.raw_term) > 3
         ORDER BY rd.df DESC LIMIT 5) sel);

-- prefix-expanded query: original word + its prefix terms, per word
CREATE OR REPLACE MACRO fts_prefix_query_t(q) AS
  (SELECT string_agg(item, ' ')
   FROM (SELECT UNNEST(string_split(q, ' ')) AS word) w
   CROSS JOIN LATERAL (
     SELECT w.word AS item
     UNION ALL
     SELECT UNNEST(string_split(COALESCE(fts_prefix_terms_t(w.word), ''), ' ')) AS item
   ) ex
   WHERE item <> '');

-- v1-compatible expand entry point (original + corrected, unchanged semantics)
CREATE OR REPLACE MACRO fts_expand_query_t(q, t_mid := 5, t_short := 1, t_long := 2) AS
  q || (CASE WHEN fts_correct_query_t(q, t_mid, t_short, t_long) = q THEN ''
             ELSE ' ' || fts_correct_query_t(q, t_mid, t_short, t_long) END);
