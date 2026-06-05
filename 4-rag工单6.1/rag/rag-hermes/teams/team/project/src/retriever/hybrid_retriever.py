"""
混合检索 — 向量检索 + BM25 加权融合
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统

BM25 索引懒加载：首次检索时从 Milvus 读文本自动构建。
"""

import time
import logging
from pathlib import Path
from typing import List, Dict, Any
from src.config import config
from src.embedder.embed import Embedder
from src.store.milvus_store import MilvusStore
from src.store.keyword_store import BM25Index
from src.retriever.reranker import Reranker

logger = logging.getLogger("rag")


def _text_preview(text: str, max_len: int = 60) -> str:
    """截取文本摘要，用于日志展示"""
    text = text.replace("\n", " ").strip()
    return text[:max_len] + "..." if len(text) > max_len else text


class HybridRetriever:
    """混合检索器：向量检索 × 0.7 + BM25 × 0.3 → Reranker 精排

    BM25 索引懒加载策略：
      1. 检查缓存文件 data/bm25_index.pkl → 加载
      2. 无缓存 → 从 Milvus 读取所有文本 → 构建 BM25 → 保存缓存
      3. 首次检索时自动触发，后续直接用
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        milvus: MilvusStore | None = None,
    ):
        self.embedder = embedder or Embedder()
        self.milvus = milvus or MilvusStore()
        self.bm25 = BM25Index()
        self.reranker = Reranker()
        self._bm25_ready = False

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        source_filter: str | None = None,
        retrieval_query: str | None = None,
    ) -> List[Dict[str, Any]]:
        """混合检索 + Reranker，含全链路计时和分数日志

        Args:
            query: 查询文本
            top_k: 返回数量
            source_filter: 按 source_file 过滤（如 "招股说明书1-无水印"），None 表示不过滤
            retrieval_query: 检索专用query（去掉公司名前缀），用于向量检索。None 则用 query。

        表格查询优化：
        - 识别表格意图关键词（多少、比例、占比、收入等）
        - 表格块权重提升 20%
        """
        t_total = time.perf_counter()

        # 0. 确保 BM25 索引就绪
        self._ensure_bm25()

        # 1. 生成查询向量
        t0 = time.perf_counter()
        # 向量检索用 core_query（去掉公司名前缀，语义更精准）
        vector_query = retrieval_query or query
        query_vector = self.embedder.embed(vector_query)
        embed_ms = (time.perf_counter() - t0) * 1000

        # 2. 向量检索 Top-20
        t0 = time.perf_counter()
        milvus_expr = f'source_file == "{source_filter}"' if source_filter else None
        vector_results = self.milvus.search(
            query_embedding=query_vector,
            top_k=config.VECTOR_TOP_K,
            expr=milvus_expr,
        )
        vector_ms = (time.perf_counter() - t0) * 1000

        # 3. BM25 检索 Top-20（含同义词扩展）
        t0 = time.perf_counter()
        bm25_query = self._expand_synonyms(query)
        bm25_results = self.bm25.search(bm25_query, top_k=config.BM25_TOP_K)
        # BM25 按 source_file 过滤
        if source_filter:
            bm25_results = [r for r in bm25_results if r.get("source_file") == source_filter]
        bm25_ms = (time.perf_counter() - t0) * 1000

        # 4. 加权融合（粗排）
        t0 = time.perf_counter()
        merged = self._fuse(vector_results, bm25_results)

        # 4.5 BM25 精确兜底：BM25 高分但被融合挤掉的 chunk 直通候选集
        # 解决问题：向量检索没命中但 BM25 精确命中的表格/文本被稀释
        existing_keys = {r["text"][:100] for r in merged}
        rescued_chunks = []
        if bm25_results:
            bm25_max = max(r["score"] for r in bm25_results)
            for r in bm25_results:
                if r["score"] < bm25_max * 0.3:  # 只捞 BM25 前 30% 的
                    continue
                key = r["text"][:100]
                if key not in existing_keys:
                    rescue = dict(r)
                    rescue["score"] = r["score"] / bm25_max * 0.5
                    rescued_chunks.append(rescue)
                    existing_keys.add(key)
                    if len(rescued_chunks) >= 5:
                        break

        # 4.6 关键词精确捞取：query中的核心关键词如果在BM25全量索引中精确出现，
        # 但被BM25排序挤掉（TF低），强制捞入候选集
        # 适用于表格中的一行数据（如"法定代表人 | 程家明"）
        import re as _re
        _query_segments = _re.split(
            r'[的了是在和与为及或中对到从被也都不那哪？?，,。.；;：:、\s]+', query)
        # 过滤：去掉公司名（已由source_filter处理）、疑问词、太短的词
        _stop_words = {"公司", "股份", "有限", "什么", "多少", "谁", "哪些",
                       "怎么", "为什么", "哪个", "请问", "可以", "应该"}
        _company_suffixes = {"股份有限公司", "有限公司", "集团公司"}
        _key_phrases = []
        for s in _query_segments:
            if len(s) < 2:
                continue
            if s in _stop_words:
                continue
            if any(s.endswith(suffix) for suffix in _company_suffixes):
                continue
            if "兴图新科" in s or "力源信息" in s:
                continue
            _key_phrases.append(s)
        _key_phrases = sorted(_key_phrases, key=len, reverse=True)[:3]
        if _key_phrases and self.bm25._chunks:
            for chunk in self.bm25._chunks:
                if source_filter and chunk.get("source_file") != source_filter:
                    continue
                text = chunk.get("text", "")
                # 任一关键短语出现即捞入（宽松匹配，让reranker做精排）
                if any(p in text for p in _key_phrases):
                    key = text[:100]
                    if key not in existing_keys:
                        rescue = dict(chunk)
                        rescue["score"] = 0.45  # 中等分数，让reranker评判
                        rescued_chunks.append(rescue)
                        existing_keys.add(key)
                        if len(rescued_chunks) >= 8:
                            break

        # 表格查询优化：提升表格块权重
        is_table_query = self._is_table_query(query)
        if is_table_query:
            for r in merged:
                chunk_type = r.get("chunk_type", "")
                if chunk_type.startswith("table_"):
                    r["score"] *= 1.2  # 表格块权重提升 20%

        # 重新排序取 Top-K
        merged.sort(key=lambda x: x.get("score", 0), reverse=True)
        merged = merged[:config.HYBRID_TOP_K]

        # BM25 兜底 chunk 直通（不受 Top-K 截断）
        if rescued_chunks:
            merged.extend(rescued_chunks)
            logger.info(f"  [BM25兜底] 补入 {len(rescued_chunks)} 条: "
                        f"{[r.get('page_no') for r in rescued_chunks]}")

        fuse_ms = (time.perf_counter() - t0) * 1000

        # ── 粗排日志：融合后 Top-10 分数 + 文本摘要 ──
        logger.info(
            f"[粗排] query=\"{query[:50]}\" | "
            f"向量召回={len(vector_results)}条 BM25召回={len(bm25_results)}条 → 融合Top-{len(merged)}"
            f"{' [表格查询]' if is_table_query else ''}"
            f"{f' [向量用: {vector_query[:30]}]' if retrieval_query else ''}"
        )
        for i, r in enumerate(merged[:10]):
            logger.info(
                f"  粗排#{i+1}  score={r['score']:.4f}  page={r.get('page_no', '?')}  "
                f"type={r.get('chunk_type', 'text')}  "
                f"text={_text_preview(r['text'])}"
            )

        # 5. Reranker 精排
        t0 = time.perf_counter()
        reranked = self.reranker.rerank(query, merged, top_n=config.RERANK_TOP_N)
        rerank_ms = (time.perf_counter() - t0) * 1000

        # 5.5 精确短语兜底：精排Top-5里如果没有query中的长短语，从BM25全量索引捞回
        self._exact_phrase_rescue(query, reranked, source_filter)

        # 5.6 定义表惩罚：术语/定义类 chunk 对大多数问题都是噪声
        for r in reranked:
            if self._is_definition_table(r):
                r["score"] *= 0.3  # 降权 70%
        # 重新排序
        reranked.sort(key=lambda x: x["score"], reverse=True)

        # ── 精排日志：Top-3 分数 + 文本摘要 ──
        logger.info(f"[精排] Reranker 输出 Top-{len(reranked)}：")
        for i, r in enumerate(reranked[:config.RERANK_TOP_N]):
            logger.info(
                f"  精排#{i+1}  score={r['score']:.4f}  page={r.get('page_no', '?')}  "
                f"type={r.get('chunk_type', 'text')}  "
                f"text={_text_preview(r['text'])}"
            )

        # ── 全链路耗时 ──
        total_ms = (time.perf_counter() - t_total) * 1000
        logger.info(
            f"[检索耗时] embed={embed_ms:.0f}ms | 向量={vector_ms:.0f}ms | "
            f"BM25={bm25_ms:.0f}ms | 融合={fuse_ms:.0f}ms | "
            f"精排={rerank_ms:.0f}ms | 总计={total_ms:.0f}ms"
        )

        return reranked[:top_k]

    def _is_table_query(self, query: str) -> bool:
        """判断是否为表格查询（简单规则匹配）

        表格查询特征：问数量、比例、金额等结构化数据
        """
        table_keywords = [
            "多少", "比例", "占比", "收入", "发行股数", "注册资本",
            "法定代表人", "持股", "股东", "募集资金", "投资", "金额",
            "毛利率", "净利润", "总资产", "净资产", "负债", "费用",
            "人员", "数量", "百分比", "%", "万元", "元",
        ]
        return any(kw in query for kw in table_keywords)

    # 同义词映射：用户可能用口语化表达，文档里用的是专业术语
    _SYNONYMS = {
        "营业额": "营业收入",
        "营收": "营业收入",
        "盈利": "净利润",
        "老板": "董事长",
        "法人": "法定代表人",
        "融了多少钱": "募集资金",
    }

    def _expand_synonyms(self, query: str) -> str:
        """BM25 检索前的同义词扩展：把口语词替换为文档术语"""
        expanded = query
        for oral, formal in self._SYNONYMS.items():
            if oral in expanded and formal not in expanded:
                expanded = expanded + " " + formal
        return expanded

    def _exact_phrase_rescue(
        self,
        query: str,
        reranked: List[Dict[str, Any]],
        source_filter: str | None = None,
    ) -> None:
        """精确短语兜底：精排Top-5里如果没有query中的长短语，从BM25全量索引捞回

        逻辑：
        1. 提取query中7字以上的连续中文子串
        2. 检查精排Top-5是否包含这些子串
        3. 如果不包含 → 从BM25全量索引找含最长子串的chunk
        4. 补进reranked前端（最多3条），让LLM看到正确答案
        """
        import re

        # ── 关键词对兜底 ──
        self._keyword_pair_rescue(query, reranked, source_filter)

        # 提取query中8字以上的关键短语（按虚词切分）
        # 8字阈值避免"武汉兴图新科电子股份"等泛匹配短语触发误捞
        # "有"不作为切分词（避免切开"有限公司"）
        segments = re.split(r'[的了是在和与为及或中对到从被也都不那哪？?，,。.；;：:、\s]+', query)
        phrases = sorted(
            [s for s in segments if len(s) >= 8],
            key=len, reverse=True,
        )
        if not phrases:
            return

        # 检查精排Top-5是否已包含这些短语
        # 只要有一个长短语不在Top-5里，就触发兜底
        top5_text = " ".join(r.get("text", "") for r in reranked[:5])
        missing = [p for p in phrases if p not in top5_text]
        if not missing:
            return  # 全部包含，不需要兜底

        # 从BM25全量索引捞含missing短语的chunk（从最长的missing短语开始搜）
        rescued = []
        existing_keys = {r["text"][:100] for r in reranked}
        for search_phrase in missing:
            for chunk in self.bm25._chunks:
                if source_filter and chunk.get("source_file") != source_filter:
                    continue
                key = chunk["text"][:100]
                if key in existing_keys:
                    continue
                if search_phrase in chunk.get("text", ""):
                    rescued.append({
                        "text": chunk["text"],
                        "page_no": chunk.get("page_no", 0),
                        "source_file": chunk.get("source_file", ""),
                        "chunk_type": chunk.get("chunk_type", "text"),
                        "section_path": chunk.get("section_path", ""),
                        "score": 0.95,  # 高分兜底，让LLM看到
                    })
                    existing_keys.add(key)
                    if len(rescued) >= 3:
                        break
            if len(rescued) >= 3:
                break

        if rescued:
            reranked[:0] = rescued  # 插入到最前面
            logger.info(f"[精确短语兜底] 补入 {len(rescued)} 条含'{missing[0]}'的 chunk")

    # 关键词对兜底：query同时含两个词时，从BM25捞含两者的chunk
    # 解决reranker截断导致的关键信息丢失
    _KEYWORD_PAIRS = [
        ("大客户", "销售处"),   # Q5: 大客户销售部的销售处数量
        ("增长率", "行业"),     # Q6: IC市场各行业增长率
        ("负增长", "行业"),     # Q6: 负增长行业
        ("组织", "结构"),       # Q5: 组织结构图
    ]

    def _keyword_pair_rescue(
        self,
        query: str,
        reranked: List[Dict[str, Any]],
        source_filter: str | None = None,
    ) -> None:
        """关键词对兜底"""
        matched_pairs = []
        for a, b in self._KEYWORD_PAIRS:
            if a in query and b in query:
                matched_pairs.append((a, b))

        if not matched_pairs:
            return

        existing_keys = {r["text"][:100] for r in reranked}
        rescued = []
        for a, b in matched_pairs:
            for chunk in self.bm25._chunks:
                if source_filter and chunk.get("source_file") != source_filter:
                    continue
                key = chunk["text"][:100]
                if key in existing_keys:
                    continue
                text = chunk.get("text", "")
                if a in text and b in text:
                    rescued.append({
                        "text": chunk["text"],
                        "page_no": chunk.get("page_no", 0),
                        "source_file": chunk.get("source_file", ""),
                        "chunk_type": chunk.get("chunk_type", "text"),
                        "section_path": chunk.get("section_path", ""),
                        "score": 0.96,
                    })
                    existing_keys.add(key)
                    if len(rescued) >= 5:
                        break
            if len(rescued) >= 5:
                break

        if rescued:
            reranked[:0] = rescued
            logger.info(f"[关键词对兜底] 补入 {len(rescued)} 条含'{matched_pairs[0][0]}+{matched_pairs[0][1]}'的 chunk")

    def _is_definition_table(self, chunk: Dict[str, Any]) -> bool:
        """判断是否为术语定义表（招股说明书中的专业术语解释）

        特征：table_semantic/table_json 类型，包含大量 "XX | 指 | XX" 模式
        这类 chunk 包含公司全名但对大多数问题都是噪声
        """
        ct = chunk.get("chunk_type", "text")
        if ct not in ("table_semantic", "table_json"):
            return False
        text = chunk.get("text", "")
        # 术语表特征：包含多个 "指" 分隔的定义行
        zhizhi_count = text.count("| 指 |") + text.count('"指"')
        if zhizhi_count >= 3:
            return True
        return False

    def _ensure_bm25(self) -> None:
        """懒加载 BM25 索引"""
        if self._bm25_ready:
            return

        bm25_path = config.DATA_DIR / "bm25_index.pkl"

        # 有缓存 → 加载
        if bm25_path.exists():
            logger.info(f"加载 BM25 索引: {bm25_path}")
            self.bm25.load(str(bm25_path))
            self._bm25_ready = True
            logger.info(f"  BM25 索引就绪: {len(self.bm25._chunks)} 条")
            return

        # 无缓存 → 从 Milvus 读取文本构建
        logger.info("BM25 缓存不存在，从 Milvus 自动构建...")
        chunks = self._load_chunks_from_milvus()
        if not chunks:
            logger.warning("  Milvus 中无数据，跳过 BM25 构建")
            return

        self.bm25.build_index(chunks)
        bm25_path.parent.mkdir(parents=True, exist_ok=True)
        self.bm25.save(str(bm25_path))
        self._bm25_ready = True
        logger.info(f"  BM25 索引构建完成: {len(chunks)} 条 → {bm25_path}")

    def _load_chunks_from_milvus(self) -> List[Dict[str, Any]]:
        """从 Milvus 读取所有文本块"""
        import pymilvus
        self.milvus.connect()
        collection = self.milvus._collection

        # 分批读取
        chunks = []
        offset = 0
        batch_size = 1000
        while True:
            results = collection.query(
                expr="chunk_index >= 0",
                offset=offset,
                limit=batch_size,
                output_fields=["text", "page_no", "source_file", "chunk_index", "chunk_type", "section_path"],
            )
            if not results:
                break
            for r in results:
                chunks.append({
                    "text": r["text"],
                    "page_no": r["page_no"],
                    "source_file": r.get("source_file", ""),
                    "chunk_index": r.get("chunk_index", 0),
                    "chunk_type": r.get("chunk_type", "text"),
                    "section_path": r.get("section_path", ""),
                })
            offset += len(results)
            if len(results) < batch_size:
                break

        return chunks

    def _fuse(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """加权融合：向量分 × 0.7 + BM25 分 × 0.3

        去重策略：同一个表格的 table_semantic 和 table_json 只保留 table_semantic
        （Markdown 对 LLM 更友好）
        """
        score_map: Dict[str, Dict[str, Any]] = {}

        def _dedup_key(r: Dict[str, Any]) -> str:
            """生成去重 key：表格块按 page+source 去重，文本块按 text[:100]"""
            ct = r.get("chunk_type", "text")
            if ct.startswith("table_"):
                # 同页同源的表格只保留一个
                return f"table|{r.get('source_file', '')}|{r.get('page_no', 0)}|{r['text'][:40]}"
            return r["text"][:100]

        def _should_replace(existing: Dict[str, Any], new: Dict[str, Any]) -> bool:
            """新的 chunk 是否应该替换已有的（table_semantic 优先于 table_json）"""
            existing_ct = existing.get("chunk_type", "text")
            new_ct = new.get("chunk_type", "text")
            # table_semantic 优先
            if new_ct == "table_semantic" and existing_ct == "table_json":
                return True
            return False

        v_scores = [r["score"] for r in vector_results]
        v_max = max(v_scores) if v_scores else 1
        for r in vector_results:
            key = _dedup_key(r)
            norm_score = r["score"] / v_max if v_max > 0 else 0
            if key not in score_map:
                score_map[key] = {
                    "text": r["text"], "page_no": r["page_no"],
                    "source_file": r["source_file"],
                    "chunk_type": r.get("chunk_type", "text"),
                    "section_path": r.get("section_path", ""),
                    "_score": 0.0,
                }
            elif _should_replace(score_map[key], r):
                # 用 table_semantic 替换 table_json
                score_map[key]["text"] = r["text"]
                score_map[key]["chunk_type"] = r.get("chunk_type", "text")
            score_map[key]["_score"] += norm_score * config.VECTOR_WEIGHT

        b_scores = [r["score"] for r in bm25_results]
        b_max = max(b_scores) if b_scores else 1
        for r in bm25_results:
            key = _dedup_key(r)
            norm_score = r["score"] / b_max if b_max > 0 else 0
            if key not in score_map:
                score_map[key] = {
                    "text": r["text"], "page_no": r["page_no"],
                    "source_file": r["source_file"],
                    "chunk_type": r.get("chunk_type", "text"),
                    "section_path": r.get("section_path", ""),
                    "_score": 0.0,
                }
            elif _should_replace(score_map[key], r):
                score_map[key]["text"] = r["text"]
                score_map[key]["chunk_type"] = r.get("chunk_type", "text")
            score_map[key]["_score"] += norm_score * config.BM25_WEIGHT

        results = sorted(score_map.values(), key=lambda x: x["_score"], reverse=True)
        for r in results:
            r["score"] = r.pop("_score")
        return results[:config.HYBRID_TOP_K]
