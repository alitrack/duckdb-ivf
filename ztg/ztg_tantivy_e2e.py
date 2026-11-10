#!/usr/bin/env python3
"""ztg_tantivy e2e — the ztg hybrid-search BM25 arm on the duckdb-tantivy
persistent index (sibling of poc/verify_hybrid.py, which validated the fts arm
inside the ivf extension; here the fts extension is NOT loaded at all).

Corpus: synthetic zh/en-mixed, 240 docs / 12 topics (deterministic, no
external data). Vectors: synthetic 64-dim, topic-clustered (the sandbox has
no ONNX model, so the query vector is derived from the corpus itself — the
macro's real deployment uses ml_embed_sentence, see ztg_tantivy.sql
section 2).

Arms (per ztg_tantivy.sql section 6):
  BM25   = tantivy_search_index('newsm_t', q, top_k := per_path)
  vector = ivf_search('newsm', qtab, 'v', top_k := per_path, nprobe := nprobe)
Fusion  = the macro's RRF: 1/(k_rrf+rank) summed over the two arms.

Checks (per query; top_k=30, per_path=50, nprobe=8, k_rrf=60):
  A0  tantivy_search_index emits in non-increasing score order (so
      row_number() OVER () inside the macro IS the BM25 rank)
  A1  pure BM25 list == the EXACT expected match set of the controlled
      corpus (AND semantics + jieba word segmentation both verified)
  A2  pure vector list contains all 17 in-topic docs (14 keyword + 3
      paraphrase) — the topic cluster dominates the ANN top-50
  A3  both arms contribute to the fused output: >=1 pure-BM25 row (a
      keyword-tagged doc of the query topic, vec_rank NULL) and ALL of the
      topic's paraphrase docs present as pure-vector rows (bm25_rank NULL)
  A4  every dual-arm doc outranks every single-arm doc (k_rrf=60 >
      per_path=50 => 2/(60+per_path) > 1/(60+1)) and output is score-sorted
  A5  exact RRF arithmetic: score == sum of 1/(60+rank) over the arms the
      doc ranks in (<= 1e-15)
  A6  determinism: two identical calls -> identical (id, score, bm25_rank,
      vec_rank) sequences
  A7  fused ids == subset of (pure BM25 list ∪ pure vector list)

Modes (like poc/verify_hybrid.py):
  train — build ivf (create+save) and tantivy (create) indexes, then run the
          macro file (its ivf_load re-loads the just-saved index; its active
          tantivy_create_index does the one and only tantivy build), then run
          all checks
  load  — fresh process, no rebuild: the file's active tantivy_create_index
          is swapped for its [LOAD-VARIANT] tantivy_load_index line
Run both:  python3 ztg/ztg_tantivy_e2e.py train && python3 ztg/ztg_tantivy_e2e.py load
"""
import os, sys, time
import numpy as np
import duckdb

# --- repo-relative paths (env-overridable); data files are gitignored ---
_D = os.path.dirname(os.path.abspath(__file__))
_R = os.path.dirname(_D)
def _env(k, d):
    v = os.environ.get(k)
    return v if v else d

IVFEXT     = _env("ZTG_IVF_EXT",     os.path.join(_R, "build", "release", "ivf.duckdb_extension"))
MLEXT      = _env("ZTG_ML_EXT",      os.path.join(_R, "ztg_test", "ml.duckdb_extension"))
TANTIVYEXT = _env("ZTG_TANTIVY_EXT", "/mnt/d/wsl2/duckdb-tantivy/build/release/tantivy.duckdb_extension")
ONNX       = _env("ZTG_MODEL_ONNX",  os.path.join(_D, "model", "model.onnx"))
TOK        = _env("ZTG_MODEL_TOK",   os.path.join(_D, "model", "tokenizer.json"))
ZTGSQL     = _env("ZTG_SQL",         os.path.join(_D, "ztg_tantivy.sql"))
WORK       = _env("ZTG_TANTIVY_WORK", os.path.join(_R, "ztg_test", "ztg_tantivy_work"))
IDXPATH    = os.path.join(WORK, "news_zt.idx")
TPATH      = os.path.join(WORK, "newsm_tantivy")

DIM    = 64
NLIST  = 16
NITER  = 10
# search parameters shared by the macro call and the pure-arm calls
TOP_K, PER_PATH, NPROBE, K_RRF = 30, 50, 8, 60

BOILER = "简报, 编辑部 每日 整理, 内容 仅供 参考, 读者 请 注意 数据 来源。"
# (zh keyword words, en keyword words, zh keyword sentence, en keyword
#  sentence, zh paraphrase sentence (contains NO topic keyword),
#   en paraphrase sentence (contains NO topic keyword))
TOPICS = [
    (["原油", "价格", "上涨"],
     ["crude", "oil", "prices"],
     "今天 原油 价格 继续 上涨, 市场 反应 强烈",
     "Crude oil prices rose again on strong demand.",
     "能源 板块 今日 走势 活跃, 分析 人士 预计 波动 持续",
     "Energy sector sentiment improved over the past week."),
    (["天然气", "供应", "紧张"],
     ["natural", "gas", "supply"],
     "近期 天然气 供应 出现 紧张, 进口 量 有所 增加",
     "Natural gas supply tightened during the quarter.",
     "冬季 取暖 需求 上升, 管网 负荷 达到 高位",
     "Winter heating demand is climbing across the region."),
    (["光伏", "装机", "规模"],
     ["solar", "panel", "output"],
     "今年 光伏 装机 规模 创 下 纪录, 行业 保持 高 增长",
     "Solar panel output reached a record this year.",
     "新能源 发电 占比 提升, 电网 消纳 压力 增大",
     "Renewable share of generation keeps rising steadily."),
    (["电动", "汽车", "销量"],
     ["electric", "vehicle", "sales"],
     "上个月 电动 汽车 销量 再 创新高, 充电 设施 快速 铺设",
     "Electric vehicle sales set another monthly record.",
     "出行 方式 持续 更新, 城市 交通 结构 不断 调整",
     "Urban transport structures keep shifting toward efficiency."),
    (["芯片", "制造", "产能"],
     ["chip", "factory", "capacity"],
     "先进 芯片 制造 产能 稳步 扩张, 设备 采购 活跃",
     "Chip factory capacity expanded steadily this year.",
     "半导体 投资 稳步 回暖, 下游 需求 逐步 恢复",
     "Semiconductor investment recovered gradually last quarter."),
    (["软件", "库", "开源"],
     ["software", "library", "open", "source"],
     "新版 软件 库 已经 开源 发布, 社区 贡献 活跃",
     "The new software library was released as open source.",
     "程序 员 工具 链 不断 更新, 开发 效率 明显 提升",
     "Developer toolchains keep improving release after release."),
    (["医疗", "研究", "成果"],
     ["medical", "research", "results"],
     "最新 医疗 研究 成果 发表, 临床 数据 令人 关注",
     "New medical research results were published last month.",
     "患者 康复 周期 缩短, 治疗 方案 更加 成熟",
     "Patient care periods shortened under the new protocol."),
    (["股市", "收益", "报告"],
     ["stock", "market", "earnings"],
     "本季度 股市 收益 报告 密集 发布, 机构 预期 分化",
     "Stock market earnings were reported across the whole sector.",
     "投资者 情绪 趋于 平稳, 资金 流向 出现 新 动向",
     "Investor sentiment stabilized while capital flows shifted."),
    (["足球", "联赛", "赛季"],
     ["football", "league", "season"],
     "新赛季 足球 联赛 赛程 公布, 焦点 赛事 引发 期待",
     "The football league season opens with a busy fixture list.",
     "球队 阵容 更新, 球迷 关注 度 持续 上升",
     "Squad renewals drew growing attention from supporters."),
    (["旅游", "行业", "复苏"],
     ["travel", "industry", "recovery"],
     "暑期 旅游 行业 明显 复苏, 出行 人次 大幅 增长",
     "The travel industry recovery accelerated over the summer.",
     "景区 服务 升级, 消费 结构 逐步 改善",
     "Destination services upgraded while spending patterns improved."),
    (["气候", "变化", "预测"],
     ["climate", "change", "forecast"],
     "今年 气候 变化 预测 显示, 极端 天气 事件 可能 增多",
     "The climate change forecast points to more extreme events.",
     "碳排放 管理 收紧, 绿色 转型 步伐 加快",
     "Carbon management tightened as the green transition accelerated."),
    (["银行", "信贷", "政策"],
     ["bank", "credit", "policy"],
     "银行 信贷 政策 出现 微调, 市场 流动性 保持 合理",
     "Bank credit policy shifted slightly as liquidity stayed ample.",
     "企业 融资 成本 下行, 经济 主体 信心 恢复",
     "Financing costs fell while business confidence recovered."),
]

# test queries: (topic idx, query text, which keyword side the query uses)
QUERIES = [
    (0, "原油 价格 上涨", "zh"),
    (5, "软件 库 开源", "zh"),
    (7, "stock market earnings", "en"),
]


def _norm(v):
    return v / np.linalg.norm(v)


def gen_corpus():
    """Deterministic corpus: 12 topics x 20 docs = 240 (ids 1..240 in topic
    order). Per topic: 10 bilingual keyword docs, 2 zh-only keyword, 2
    en-only keyword, 3 paraphrase (in-topic vector, NO topic keywords),
    3 keyword-tagged docs (text carries the topic's keywords, vector parked
    in another topic's cluster -> pure-BM25 candidates)."""
    ids, texts, clusters = [], [], []
    for t, (zkw, ekw, zsent, esent, zpara, epara) in enumerate(TOPICS):
        for _ in range(10):
            ids.append(None); texts.append(zsent + " " + esent + " " + BOILER); clusters.append(t)
        for _ in range(2):
            ids.append(None); texts.append(zsent + " " + BOILER); clusters.append(t)
        for _ in range(2):
            ids.append(None); texts.append(esent + " " + BOILER); clusters.append(t)
        for _ in range(3):
            ids.append(None); texts.append(zpara + " " + epara + " " + BOILER); clusters.append(t)
        for i in range(3):
            tag = "关键词: " + " ".join(zkw) + " / " + " ".join(ekw) + " " + BOILER
            ids.append(None); texts.append(tag)
            clusters.append((t + 1 + 2 * i) % len(TOPICS))
    ids = list(range(1, len(texts) + 1))
    # topic doc id maps (offsets within each 20-doc block)
    kw_ids, para_ids, tag_ids = {}, {}, {}
    for t in range(len(TOPICS)):
        base = t * 20 + 1
        kw_ids[t]   = list(range(base, base + 14))
        para_ids[t] = list(range(base + 14, base + 17))
        tag_ids[t]  = list(range(base + 17, base + 20))
    # vectors: per-topic base direction + small per-doc noise
    rng_b = np.random.default_rng(20261110)
    base_v = [_norm(rng_b.standard_normal(DIM)) for _ in TOPICS]
    rng_d = np.random.default_rng(42)
    vecs = [_norm(base_v[c] + 0.08 * rng_d.standard_normal(DIM)) for c in clusters]
    # query vector per topic: mean of that topic's 14 keyword docs
    qvecs = {}
    for t in range(len(TOPICS)):
        qvecs[t] = _norm(np.mean(np.array([vecs[t * 20 + i] for i in range(14)]), axis=0))
    return ids, texts, vecs, kw_ids, para_ids, tag_ids, qvecs


def exec_ztg_file(con, mode):
    """Load ztg_tantivy.sql: replace (EDIT) placeholders with real paths,
    strip comments, split on ';', execute. In load mode the active
    tantivy_create_index statement is swapped for the [LOAD-VARIANT]
    tantivy_load_index line. Returns (n_executed, skipped_descriptions)."""
    with open(ZTGSQL) as f:
        sql = f.read()
    load_stmt = None
    for ln in sql.splitlines():
        s = ln.strip()
        if s.startswith("-- [LOAD-VARIANT]"):
            load_stmt = s[len("-- [LOAD-VARIANT]"):].strip()
    # work on non-comment content only (documentation comments may carry
    # example placeholders that never execute)
    raw = "\n".join(l for l in sql.splitlines() if not l.lstrip().startswith("--"))
    subs = [
        ("/abs/path/to/ivf.duckdb_extension", IVFEXT),
        ("/abs/path/to/ml.duckdb_extension", MLEXT),
        ("/abs/path/to/tantivy.duckdb_extension", TANTIVYEXT),
        ("/abs/path/to/model.onnx", ONNX),
        ("/abs/path/to/tokenizer.json", TOK),
        ("/abs/path/to/news.idx", IDXPATH),
        ("/abs/path/to/newsm_tantivy", TPATH),
    ]
    for ph, real in subs:
        raw = raw.replace(ph, real)
        if load_stmt:
            load_stmt = load_stmt.replace(ph, real)
    if "/abs/path/to/" in raw or (load_stmt and "/abs/path/to/" in load_stmt):
        raise SystemExit("unreplaced /abs/path/to/ placeholder in ztg_tantivy.sql")
    stmts = [s.strip() for s in raw.split(";") if s.strip()]
    if mode == "load":
        if not load_stmt:
            raise SystemExit("no [LOAD-VARIANT] line found in ztg_tantivy.sql")
        stmts = [load_stmt if st.startswith("SELECT * FROM tantivy_create_index(") else st
                 for st in stmts]
    executed, skipped = 0, []
    for st in stmts:
        first = st.splitlines()[0]
        if st.startswith("LOAD '%s'" % MLEXT) and not os.path.exists(MLEXT):
            skipped.append(first + "  (ml extension file missing)")
            continue
        if st.startswith("SELECT * FROM ml_load_sentence") and not (os.path.exists(ONNX) and os.path.exists(TOK)):
            skipped.append(first + "  (no ONNX model in this sandbox)")
            continue
        con.execute(st)
        executed += 1
    return executed, skipped


def check_query(con, tag, t, q, side, kw_ids, para_ids, tag_ids, qvec):
    print("[%s] q=%r (topic %d, %s query)" % (tag, q, t, side), flush=True)
    con.execute("DROP TABLE IF EXISTS qtab")
    con.execute("CREATE TABLE qtab AS SELECT ?::FLOAT[%d] AS v" % DIM, [qvec.tolist()])

    # pure arms — same index / parameters as the macro's two arms
    pure_b = con.execute(
        "SELECT doc_id, score, row_number() OVER () AS rk FROM "
        "tantivy_search_index('newsm_t', ?, top_k := %d)" % PER_PATH, [q]).fetchall()
    pure_v = con.execute(
        "SELECT id, score, row_number() OVER (ORDER BY score DESC) AS rk FROM "
        "ivf_search('newsm', 'qtab', 'v', top_k := %d, nprobe := %d)" % (PER_PATH, NPROBE)).fetchall()
    fused1 = con.execute(
        "SELECT id, score, bm25_rank, vec_rank FROM ztg_search_news_tantivy(?, 'qtab', "
        "top_k := %d, per_path := %d, nprobe := %d, k_rrf := %d)"
        % (TOP_K, PER_PATH, NPROBE, K_RRF), [q]).fetchall()
    fused2 = con.execute(
        "SELECT id, score, bm25_rank, vec_rank FROM ztg_search_news_tantivy(?, 'qtab', "
        "top_k := %d, per_path := %d, nprobe := %d, k_rrf := %d)"
        % (TOP_K, PER_PATH, NPROBE, K_RRF), [q]).fetchall()

    bmap = {r[0]: r[2] for r in pure_b}   # id -> bm25 rank (1-based)
    vmap = {r[0]: r[2] for r in pure_v}   # id -> vec rank (1-based)
    # expected exact BM25 match set for the controlled corpus
    zkw = TOPICS[t][0]
    matched_kw = []
    for i in range(14):  # 10 bilingual, 2 zh-only, 2 en-only
        doc_side = "both" if i < 10 else ("zh" if i < 12 else "en")
        if doc_side != "both" and doc_side != side:
            continue
        matched_kw.append(kw_ids[t][i])
    expected_b = set(matched_kw) | set(tag_ids[t])

    checks = {}
    # A0 rank fidelity of tantivy_search_index output
    checks["A0_bm25_rank_order"] = all(
        pure_b[i][1] >= pure_b[i + 1][1] for i in range(len(pure_b) - 1))
    # A1 exact match set (AND semantics + segmentation)
    checks["A1_bm25_exact_set"] = set(bmap) == expected_b
    if not checks["A1_bm25_exact_set"]:
        print("        A1 detail: got=%s expected=%s" %
              (sorted(bmap), sorted(expected_b)), flush=True)
        # diagnostic: what does the analyzer actually produce for the query?
        try:
            print("        A1 tokenize(q, chinese)=", flush=True,
                  end="")
            print(con.execute("SELECT tokenize(?, 'chinese')", [q]).fetchall(), flush=True)
        except Exception as e:
            print("        (tokenize unavailable: %s)" % e, flush=True)
    # A2 vector arm found the whole in-topic cluster
    checks["A2_vec_in_topic"] = (set(kw_ids[t]) | set(para_ids[t])) <= set(vmap)
    if not checks["A2_vec_in_topic"]:
        missing = (set(kw_ids[t]) | set(para_ids[t])) - set(vmap)
        print("        A2 missing from vector list: %s" % sorted(missing), flush=True)
    # A3 both arms contribute
    pure_bm25_rows = [r for r in fused1 if r[3] is None]
    pure_vec_rows  = [r for r in fused1 if r[2] is None]
    checks["A3a_pure_bm25"] = (
        len(pure_bm25_rows) >= 1
        and all(r[0] in set(tag_ids[t]) for r in pure_bm25_rows))
    checks["A3b_pure_vec"] = set(para_ids[t]) <= {r[0] for r in pure_vec_rows}
    if not checks["A3a_pure_bm25"]:
        print("        A3a pure-bm25 rows: %s (tag ids %s)" %
              ([r[0] for r in pure_bm25_rows], tag_ids[t]), flush=True)
    if not checks["A3b_pure_vec"]:
        print("        A3b missing paraphrase: %s" %
              sorted(set(para_ids[t]) - {r[0] for r in pure_vec_rows}), flush=True)
    # A4 dual-arm docs outrank single-arm docs; output score-sorted
    # single = exactly ONE arm (XOR); dual = both arms
    dual   = [r for r in fused1 if r[2] is not None and r[3] is not None]
    single = [r for r in fused1 if (r[2] is None) != (r[3] is None)]
    checks["A4_dual_gt_single"] = (
        len(dual) > 0 and len(single) > 0
        and min(r[1] for r in dual) > max(r[1] for r in single))
    if not checks["A4_dual_gt_single"]:
        md = min(dual, key=lambda r: r[1]); ms = max(single, key=lambda r: r[1])
        print("        A4 debug: min_dual=%s (b=%s v=%s)  max_single=%s (b=%s v=%s)" %
              (round(md[1], 6), md[2], md[3], round(ms[1], 6), ms[2], ms[3]), flush=True)
    checks["A4_sorted"] = all(
        fused1[i][1] >= fused1[i + 1][1] for i in range(len(fused1) - 1))
    # A5 exact RRF arithmetic
    checks["A5_rrf_math"] = all(
        abs(r[1] - ((1.0 / (K_RRF + r[2]) if r[2] is not None else 0.0)
                    + (1.0 / (K_RRF + r[3]) if r[3] is not None else 0.0))) <= 1e-15
        for r in fused1)
    # A6 determinism (same q, same qtab table, back to back)
    checks["A6_deterministic"] = (fused1 == fused2)
    # A7 fused ids are the union of the two arm lists (nothing from nowhere)
    checks["A7_union_subset"] = set(r[0] for r in fused1) <= set(bmap) | set(vmap)

    for k, ok in checks.items():
        print("        %-18s %s" % (k + ":", "OK" if ok else "FAIL"), flush=True)
    print("        n_pure_b=%d n_pure_v=%d n_fused=%d (dual=%d single=%d; "
          "pure-bm25=%d pure-vec=%d)" %
          (len(pure_b), len(pure_v), len(fused1), len(dual), len(single),
           len(pure_bm25_rows), len(pure_vec_rows)), flush=True)
    print("        top5: %s" %
          [(r[0], round(r[1], 5), r[2], r[3]) for r in fused1[:5]], flush=True)
    return all(checks.values())


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    if mode not in ("train", "load"):
        raise SystemExit("usage: ztg_tantivy_e2e.py train|load")
    assert os.path.exists(IVFEXT),     "missing %s" % IVFEXT
    assert os.path.exists(TANTIVYEXT), "missing %s" % TANTIVYEXT
    os.makedirs(WORK, exist_ok=True)

    con = duckdb.connect(config={"allow_unsigned_extensions": "true"})
    print("[%s] duckdb %s; loading ivf + tantivy (no fts)" %
          (mode, con.execute("select version()").fetchone()[0]), flush=True)
    con.execute("LOAD '%s'" % IVFEXT)
    con.execute("LOAD '%s'" % TANTIVYEXT)

    ids, texts, vecs, kw_ids, para_ids, tag_ids, qvecs = gen_corpus()
    print("[%s] corpus: %d docs, dim=%d, nlist=%d" %
          (mode, len(ids), DIM, NLIST), flush=True)

    if mode == "train":
        con.execute("CREATE OR REPLACE TABLE news (id BIGINT, text VARCHAR)")
        con.executemany("INSERT INTO news VALUES (?, ?)", list(zip(ids, texts)))
        con.execute("CREATE OR REPLACE TABLE vec (id BIGINT, v FLOAT[%d])" % DIM)
        con.executemany("INSERT INTO vec VALUES (?, ?)",
                        list(zip(ids, [v.tolist() for v in vecs])))
        t0 = time.time()
        n = con.execute(
            "SELECT * FROM ivf_create_index('newsm', 'vec', 'id', 'v', %d, %d, true)"
            % (NLIST, NITER)).fetchone()[0]
        print("[%s] ivf_create_index: %d rows in %.2fs" % (mode, n, time.time() - t0), flush=True)
        t0 = time.time()
        con.execute("SELECT * FROM ivf_save('newsm', '%s')" % IDXPATH)
        print("[%s] ivf_save -> %s (%.1f KB, %.2fs)" %
              (mode, IDXPATH, os.path.getsize(IDXPATH) / 1e3, time.time() - t0), flush=True)

    t0 = time.time()
    executed, skipped = exec_ztg_file(con, mode)
    print("[%s] ztg_tantivy.sql: %d statements executed in %.2fs" %
          (mode, executed, time.time() - t0), flush=True)
    for s in skipped:
        print("[%s] skipped: %s" % (mode, s), flush=True)
    print("[%s] tantivy_list: %s" %
          (mode, con.execute("SELECT * FROM tantivy_list()").fetchall()), flush=True)

    allok = True
    for (t, q, side) in QUERIES:
        allok &= check_query(con, mode, t, q, side, kw_ids, para_ids, tag_ids, qvecs[t])
    print("%s_DONE ok=%s" % (mode.upper(), allok), flush=True)
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
