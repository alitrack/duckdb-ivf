-- ============================================================
-- fts + vector hybrid search — reusable SQL macro layer
-- DuckDB 2.0.0 (layered fts). Verified on 120k ag_news (bge-m3 1024-dim).
--
-- CONTRACT (for table <T> with id <ID>, text <TXT>):
--   1) CREATE TABLE <T> ...                         -- your data
--   2) PRAGMA create_fts_index('<T>','<ID>','<TXT>',
--          stemmer='porter', stopwords='english', layered_search=true);
--   3) CREATE TABLE <V> (<ID>, <VEC> FLOAT[<DIM>]);  -- your embeddings (L2-norm recommended)
--   4) register the macro below with {table}=<T> {fts_schema}=fts_main_<T>
--      {vec_table}=<V> {vec_col}=<VEC> {id_col}=<ID> {text_col}=<TXT>
--
-- USAGE:
--   SELECT * FROM <T>_hybrid_search('crude oil', ? /*query_vec*/, top_k := 10);
-- TUNING: top_k (final rows, 10) / k_rrf (RRF const, 60) / per_path (per-branch, 50)
-- ============================================================
CREATE OR REPLACE MACRO news_hybrid_search(query, query_vec, top_k := 10, k_rrf := 60, per_path := 50) AS TABLE
  WITH f  AS (SELECT docname::INT AS id, rank AS frk
              FROM fts_main_news.search_layered_bm25(query, top_k := per_path)),
       vr AS (SELECT id, row_number() OVER (ORDER BY list_distance(v, query_vec)) AS vrk
              FROM vec LIMIT per_path),
       u  AS (SELECT id, frk, NULL::INT AS vrk FROM f
              UNION ALL
              SELECT id AS id, NULL AS frk, vrk FROM vr),
       rrf AS (SELECT id,
               sum(CASE WHEN frk IS NOT NULL THEN 1.0/(k_rrf + frk) ELSE 0 END)
           +   sum(CASE WHEN vrk IS NOT NULL THEN 1.0/(k_rrf + vrk) ELSE 0 END) AS rrf_score
               FROM u GROUP BY id),
       top AS (SELECT id, round(rrf_score,6) AS rrf_score FROM rrf ORDER BY rrf_score DESC LIMIT top_k)
  SELECT top.id,
         top.rrf_score,
         f.frk AS bm25_rank,
         vr.vrk AS vec_rank,
         substring(news.text, 1, 80) AS snippet
  FROM top
  JOIN news   ON news.id = top.id
  LEFT JOIN f    ON f.id = top.id
  LEFT JOIN vr   ON vr.id = top.id
  ORDER BY top.rrf_score DESC
