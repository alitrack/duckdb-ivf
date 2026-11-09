import duckdb
# Verify ztg.sql deploys end-to-end: substitute the (EDIT) placeholder paths
# with the real PoC paths, then execute the whole file and call the macro.
sql = open(ZTGSQL).read()
sql = sql.replace("/abs/path/to/ivf.duckdb_extension",
                  IVFEXT)
sql = sql.replace("/abs/path/to/ml.duckdb_extension",
                  MLEXT)
sql = sql.replace("/abs/path/to/model.onnx", ONNX)
sql = sql.replace("/abs/path/to/tokenizer.json", TOK)
sql = sql.replace("/abs/path/to/news.idx", IDX5K)
# section 3 corpus build: use the real parquet (fresh table each run)
sql = sql.replace(
    "-- CREATE OR REPLACE TABLE news AS\n"
    "--   SELECT row_number() OVER() AS id, text\n"
    "--   FROM read_parquet('/abs/path/to/corpus.parquet');",
    "CREATE OR REPLACE TABLE news AS\n"
    "  SELECT row_number() OVER() AS id, text\n"
    "  FROM read_parquet('ag_news_train.parquet') LIMIT 5000;")
assert "/abs/path/" not in sql, "leftover placeholder"

c = duckdb.connect(config={"allow_unsigned_extensions": "true"})
import time
t0 = time.time()
c.execute(sql)
print("FULL ztg.sql DEPLOYED in %.2fs" % (time.time() - t0))
r = c.execute("SELECT id, round(score,4) FROM ztg_search_news('crude oil prices')").fetchall()
print("CALL OK:", r[:5], "total", len(r))
r2 = c.execute("SELECT id FROM ztg_search_news('stock market earnings', 5)").fetchall()
print("top_k=5 total", len(r2))
