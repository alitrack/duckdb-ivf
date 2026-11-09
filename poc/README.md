# poc — ztg hybrid search end-to-end (reproducible numbers)

Reproduction scripts for the article numbers: in-process hybrid search
(duckdb-ivf + duckdb-ml + fts) on ag_news 120k. All paths are env-overridable;
data files are NOT in git.

## Pipeline

| Step | Script | Produces |
|---|---|---|
| 1. embed corpus (bge-m3 1024d, LAN service) | `wsl_embed.py` | `data/emb_bge_m3.npz` |
| 2. build IVF index, recall check (14.5x / recall 1.00) | `../test_e2e.py` | index + recall numbers |
| 3. IVF + persistence verification (156s -> 1.1s load) | `verify_persist.py`, `run_persist_*.sh` | persistence numbers |
| 4. constructed-vector nDCG (the 0.934 / 0.954 column) | `ndcg_e2e.py` | nDCG@10 |
| 5. **real-vector nDCG (the honest 0.849 / 0.872)** | `ndcg_real.py`, `run_ndcg.sh` | nDCG@10 |
| 6. in-process e2e, text-in (97 docs/s, train/load) | `e2e_inproc.py`, `emb_bench.py` | throughput + e2e |
| 7. ztg.sql single-file deployment check | `ztg_deploy_test.py`, `ztg_bundle_test.py` | ZTG_BUNDLE_OK |
| 8. hybrid correctness vs SQL-baseline RRF | `verify_hybrid.py`, `poc.py`, `poc_ivf.py` | equivalence |
| 9. duckdb-ml embedding vs onnxruntime reference | `verify_sentence.py` | cosine == 1.0 |
| 10. regression: qtab path unchanged | `e2e_regression.py`, `run_e2e_regression.sh` | golden output |

`hybrid_search.sql` = the standalone RRF macro layer (pre-extension baseline).

## Environment variables (defaults are repo-relative)

```
ZTG_CORPUS     corpus parquet        (default poc/data/ag_news_train.parquet)
ZTG_EMB_NPZ    bge-m3 embeddings npz (default poc/data/emb_bge_m3.npz)
ZTG_IDX_120K   120k x 1024d index    (default poc/data/news.ivf.idx)
ZTG_IDX_5K     5k x 384d index       (default poc/data/news_384.idx)
ZTG_IVF_EXT    ivf extension binary  (default ../build/release/ivf.duckdb_extension)
ZTG_ML_EXT     ml extension binary   (default poc/ml.duckdb_extension)
ZTG_MODEL_ONNX / ZTG_MODEL_TOK / ZTG_MODEL_DIR   MiniLM model files
ZTG_BGE_API    bge-m3 embedding service (default http://10.10.10.81:8765/v1/embeddings)
ZTG_SQL        ztg.sql               (default ../ztg/ztg.sql)
ZTG_BUNDLE_DIR scratch bundle dir    (default ../ztg_test)
```

## Data (gitignored, obtain via)

- `ag_news_train.parquet` — Kaggle ag_news; `wsl_embed.py` embeds it via the
  bge-m3 service (`ZTG_BGE_API`) into `data/emb_bge_m3.npz`
- `news.ivf.idx` / `news_384.idx` — built by `../test_e2e.py` /
  `e2e_inproc.py train` (fresh-train section of `ztg/ztg.sql` shows the SQL)
- model.onnx / tokenizer.json — all-MiniLM-L6-v2 ONNX export

## Smoke (verified 2026-11-09)

```
ZTG_CORPUS=... ZTG_IDX_5K=... ZTG_ML_EXT=... ZTG_MODEL_ONNX=... ZTG_MODEL_TOK=... \
  python3 ztg_bundle_test.py
# -> ZTG_BUNDLE_OK all_ok=True   (ztg_search_news == ivf_hybrid_embed, ~45ms/query)
```

Build/rebuild of the `ml` extension (onnx feature) happens in the
[duckdb-ml](https://github.com/alitrack/duckdb-ml) repo — set `ZTG_ML_WS`
for `rebuild_verify.sh` / `run_verify*.sh`.
