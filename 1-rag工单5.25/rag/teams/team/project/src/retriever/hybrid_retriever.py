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
    ) -> List[Dict[str, Any]]:
        """混合检索 + Reranker，含全链路计时和分数日志"""
        t_total = time.perf_counter()

        # 0. 确保 BM25 索引就绪
        self._ensure_bm25()

        # 1. 生成查询向量
        t0 = time.perf_counter()
        query_vector = self.embedder.embed(query)
        embed_ms = (time.perf_counter() - t0) * 1000

        # 2. 向量检索 Top-20
        t0 = time.perf_counter()
        vector_results = self.milvus.search(
            query_embedding=query_vector,
            top_k=config.VECTOR_TOP_K,
        )
        vector_ms = (time.perf_counter() - t0) * 1000

        # 3. BM25 检索 Top-20
        t0 = time.perf_counter()
        bm25_results = self.bm25.search(query, top_k=config.BM25_TOP_K)
        bm25_ms = (time.perf_counter() - t0) * 1000

        # 4. 加权融合（粗排）
        t0 = time.perf_counter()
        merged = self._fuse(vector_results, bm25_results)
        fuse_ms = (time.perf_counter() - t0) * 1000

        # ── 粗排日志：融合后 Top-10 分数 + 文本摘要 ──
        logger.info(
            f"[粗排] query=\"{query[:50]}\" | "
            f"向量召回={len(vector_results)}条 BM25召回={len(bm25_results)}条 → 融合Top-{len(merged)}"
        )
        for i, r in enumerate(merged[:10]):
            logger.info(
                f"  粗排#{i+1}  score={r['score']:.4f}  page={r.get('page_no', '?')}  "
                f"text={_text_preview(r['text'])}"
            )

        # 5. Reranker 精排
        t0 = time.perf_counter()
        reranked = self.reranker.rerank(query, merged, top_n=config.RERANK_TOP_N)
        rerank_ms = (time.perf_counter() - t0) * 1000

        # ── 精排日志：Top-3 分数 + 文本摘要 ──
        logger.info(f"[精排] Reranker 输出 Top-{len(reranked)}：")
        for i, r in enumerate(reranked[:config.RERANK_TOP_N]):
            logger.info(
                f"  精排#{i+1}  score={r['score']:.4f}  page={r.get('page_no', '?')}  "
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
                output_fields=["text", "page_no", "source_file", "chunk_index"],
            )
            if not results:
                break
            for r in results:
                chunks.append({
                    "text": r["text"],
                    "page_no": r["page_no"],
                    "source_file": r.get("source_file", ""),
                    "chunk_index": r.get("chunk_index", 0),
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
        """加权融合：向量分 × 0.7 + BM25 分 × 0.3"""
        score_map: Dict[str, Dict[str, Any]] = {}

        v_scores = [r["score"] for r in vector_results]
        v_max = max(v_scores) if v_scores else 1
        for r in vector_results:
            key = r["text"][:100]
            norm_score = r["score"] / v_max if v_max > 0 else 0
            if key not in score_map:
                score_map[key] = {
                    "text": r["text"], "page_no": r["page_no"],
                    "source_file": r["source_file"], "_score": 0.0,
                }
            score_map[key]["_score"] += norm_score * config.VECTOR_WEIGHT

        b_scores = [r["score"] for r in bm25_results]
        b_max = max(b_scores) if b_scores else 1
        for r in bm25_results:
            key = r["text"][:100]
            norm_score = r["score"] / b_max if b_max > 0 else 0
            if key not in score_map:
                score_map[key] = {
                    "text": r["text"], "page_no": r["page_no"],
                    "source_file": r["source_file"], "_score": 0.0,
                }
            score_map[key]["_score"] += norm_score * config.BM25_WEIGHT

        results = sorted(score_map.values(), key=lambda x: x["_score"], reverse=True)
        for r in results:
            r["score"] = r.pop("_score")
        return results[:config.HYBRID_TOP_K]
