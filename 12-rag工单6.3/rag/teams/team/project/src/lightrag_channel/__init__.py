"""
LightRAG 通道 — 知识图谱 RAG
工单12：LightRAG 优化任务

使用 LightRAG 实现基于知识图谱的检索，与传统向量 RAG 通道并行。
"""

from .config import (
    MIMO_BASE_URL, MIMO_API_KEY, MIMO_MODEL,
    BGE_M3_PATH, EMBEDDING_DIM, LIGHTRAG_STORAGE_DIR,
    DEFAULT_QUERY_MODE, DEFAULT_TOP_K,
)
from .init_lightrag import create_lightrag_instance, llm_model_func
from .query import query_lightrag, query_lightrag_with_contexts
from .ingest import ingest_document, find_content_lists
from .prompts import FINANCIAL_ENTITY_TYPES_GUIDANCE

__all__ = [
    "create_lightrag_instance",
    "query_lightrag",
    "query_lightrag_with_contexts",
    "ingest_document",
    "find_content_lists",
    "llm_model_func",
    "FINANCIAL_ENTITY_TYPES_GUIDANCE",
    "MIMO_BASE_URL", "MIMO_API_KEY", "MIMO_MODEL",
    "BGE_M3_PATH", "EMBEDDING_DIM", "LIGHTRAG_STORAGE_DIR",
    "DEFAULT_QUERY_MODE", "DEFAULT_TOP_K",
]
