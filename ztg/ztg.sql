-- ztg: single-call text-in hybrid search (zg-clone, DuckDB-native).
--
-- Per-query call — text in, top-k out, zero precomputed vectors, zero network:
--
--   SELECT * FROM ztg_search_news('crude oil prices');
--   SELECT * FROM ztg_search_news('crude oil prices', 20);
--
-- Pipeline (all in-process):
--   query text --duckdb-ml--> vector --ivf ANN (this ext.)--> arm 1
--   query text --fts BM25---> arm 2            -> RRF fusion -> top-k
--
-- Deployment (once per index; machine-specific paths NOT in this file):
--   1. Connect with config allow_unsigned_extensions=true; LOAD ivf, ml, fts.
--   2. Train + persist: ivf_create(...) ; ivf_save(...).
--   3. FTS: PRAGMA create_fts_index('<tbl>','<id_col>','<text_col>', ...).
--   4. Model: ml_load_sentence('<model>','<onnx>','<tok>', 128).
--   5. Create the deployment macro below, editing the FIVE literals to your
--      deployment: index_name, table, id_col, text_col, model_name. One macro
--      per index/table — the C-API table function (ivf_hybrid_embed) binds its
--      arguments as constants, so the profile is fixed here, not read from a
--      config table at query time.
--
-- KNOWN LIMITATION: query text containing a single quote will not round-trip
-- (the embedded expression is not quote-escaped inside the table-function
-- macro — a DuckDB limitation). Avoid ' in queries, or escape in the caller.

CREATE OR REPLACE MACRO ztg_search_news(q, top_k := 10) AS TABLE
  SELECT id, score, bm25_rank, vec_rank
  FROM ivf_hybrid_embed(
    'newsm', 'news', 'id', 'text', q,                       -- <== edit these 5
    concat('ml_embed_sentence(', chr(39), 'minilm', chr(39), ', ', chr(39), q, chr(39), ')'),
    top_k := top_k, per_path := 50, nprobe := 8, k_rrf := 60);
