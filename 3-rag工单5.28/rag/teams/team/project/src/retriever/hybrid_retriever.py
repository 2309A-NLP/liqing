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
    ) -> List[Dict[str, Any]]:
        """混合检索 + Reranker，含全链路计时和分数日志

        Args:
            query: 查询文本
            top_k: 返回数量
            source_filter: 按 source_file 过滤（如 "招股说明书1-无水印"），None 表示不过滤

        表格查询优化：
        - 识别表格意图关键词（多少、比例、占比、收入等）
        - 表格块权重提升 20%
        """
        t_total = time.perf_counter()

        # 0. 确保 BM25 索引就绪
        self._ensure_bm25()

        # 1. 生成查询向量
        t0 = time.perf_counter()
        query_vector = self.embedder.embed(query)
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

        # 3. BM25 检索 Top-20
        t0 = time.perf_counter()
        bm25_results = self.bm25.search(query, top_k=config.BM25_TOP_K)
        # BM25 按 source_file 过滤
        if source_filter:
            bm25_results = [r for r in bm25_results if r.get("source_file") == source_filter]
        bm25_ms = (time.perf_counter() - t0) * 1000

        # 4. 加权融合（粗排）
        t0 = time.perf_counter()
        merged = self._fuse(vector_results, bm25_results)

        # 表格查询优化：提升表格块权重
        is_table_query = self._is_table_query(query)
        if is_table_query:
            for r in merged:
                chunk_type = r.get("chunk_type", "")
                if chunk_type.startswith("table_"):
                    r["score"] *= 1.2  # 表格块权重提升 20%

        # 数字匹配加权：如果查询中有数字，包含相同数字的 chunk 权重提升
        import re
        query_numbers = set(re.findall(r'[\d,]+\.?\d*', query))
        if query_numbers:
            for r in merged:
                text = r.get("text", "")
                matched = sum(1 for num in query_numbers if num in text)
                if matched > 0:
                    r["score"] *= (1 + matched * 0.3)  # 每匹配一个数字提升 30%

        fuse_ms = (time.perf_counter() - t0) * 1000

        # ── 粗排日志：融合后 Top-10 分数 + 文本摘要 ──
        logger.info(
            f"[粗排] query=\"{query[:50]}\" | "
            f"向量召回={len(vector_results)}条 BM25召回={len(bm25_results)}条 → 融合Top-{len(merged)}"
            f"{' [表格查询]' if is_table_query else ''}"
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

        # 5.5 定义表惩罚：术语/定义类 chunk 对大多数问题都是噪声
        for r in reranked:
            if self._is_definition_table(r):
                r["score"] *= 0.3  # 降权 70%
        # 重新排序
        reranked.sort(key=lambda x: x["score"], reverse=True)

        # 5.6 数字匹配加权：在 Reranker 之后，对包含查询中关键数字的 chunk 提升排名
        import re
        # 提取查询中的关键数字（如 1,670、25.04% 等）
        query_numbers = set(re.findall(r'[\d,]+\.?\d*%?', query))
        # 只保留有实际意义的数字（长度 > 2）
        query_numbers = {n for n in query_numbers if len(n) > 2}
        if query_numbers:
            for r in reranked:
                text = r.get("text", "")
                matched = sum(1 for num in query_numbers if num in text)
                if matched > 0:
                    r["score"] = min(1.0, r["score"] + matched * 0.01)  # 每匹配一个数字提升 0.01
            # 重新排序
            reranked.sort(key=lambda x: x["score"], reverse=True)

        # 5.7 表格查询优化：如果是表格类型问题，优先返回表格数据
        if is_table_query:
            # 找到所有表格 chunk
            table_chunks = [r for r in reranked if r.get("chunk_type", "").startswith("table_")]
            if table_chunks:
                best_table = max(table_chunks, key=lambda x: x["score"])
                # 强制把最高分的表格 chunk 插入到 #2 位置
                reranked.remove(best_table)
                reranked.insert(1, best_table)
                # 如果有第二个表格 chunk，也插入到 #3 位置
                remaining_tables = [r for r in reranked[2:] if r.get("chunk_type", "").startswith("table_")]
                if remaining_tables:
                    second_table = max(remaining_tables, key=lambda x: x["score"])
                    reranked.remove(second_table)
                    reranked.insert(2, second_table)

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
