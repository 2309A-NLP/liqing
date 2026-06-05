# 检索核心模块：负责向量召回、BM25 召回、重排序和上下文拼装。
import logging
import os
import re
from functools import lru_cache
from threading import Lock
from typing import Dict, List

import numpy as np
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from rank_bm25 import BM25Okapi
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, utility

from .config import (
    DEFAULT_COLLECTION,
    DEFAULT_MEMORY_COLLECTION,
    get_memory_collection_name,
    get_milvus_collection_name,
    load_embedder,
    load_reranker,
)
from .models import RetrievedDoc, RetrievalStats

# 模块日志。
logger = logging.getLogger(__name__)
# reranker 在并发环境下加锁，避免 tokenizer 线程冲突。
_RERANK_LOCK = Lock()


# 这个函数负责获取 Milvus 里的集合对象，并把结果缓存起来。
# 这样后面反复检索同一个集合时，就不用每次都重新创建连接句柄。
# 它是后续所有检索函数共同依赖的基础入口。
@lru_cache(maxsize=2)
def get_collection(collection_name: str) -> Collection:
    if not utility.has_collection(collection_name):
        raise ValueError(f"Milvus collection 不存在: {collection_name}")
    collection = Collection(collection_name)
    try:
        collection.load()
    except Exception:
        if collection_name == get_memory_collection_name():
            raise
        raise
    return collection


# 这个函数生成集合签名，用来表示当前集合“有没有变化”。
# BM25 索引缓存会用到它，避免集合变化后还沿用旧索引。
def get_collection_signature(collection: Collection) -> str:
    try:
        return f"{collection.name}:{collection.num_entities}"
    except Exception:
        return f"{collection.name}:unknown"


# 这个函数负责从 Milvus 里读出所有文档，并转成 BM25 需要的语料。
# 它同时也会把原始记录包装成 RetrievedDoc，后面 BM25 召回时可以直接复用。
# 由于数据量可能很大，所以这里也做了缓存。
@lru_cache(maxsize=16)
def load_bm25_corpus(collection_name: str, signature: str):
    collection = get_collection(collection_name)
    query_result = collection.query(
        expr="id != ''",
        output_fields=["id", "source", "domain", "title", "role", "keywords", "content", "vector_text"],
        limit=16384,
    )
    corpus_texts = []
    docs: List[RetrievedDoc] = []
    for row in query_result or []:
        text = str(
            row.get("search_text")
            or row.get("vector_text")
            or row.get("summary")
            or row.get("content")
            or ""
        )
        corpus_texts.append(tokenize(text))
        docs.append(
            RetrievedDoc(
                id=str(row.get("id", "")),
                score=0.0,
                source=str(row.get("source", "")),
                domain=str(row.get("domain", "")),
                title=str(row.get("title", "")),
                role=str(row.get("role", "")),
                keywords=str(row.get("keywords", "")),
                content=str(row.get("content", "")),
                vector_text=str(row.get("vector_text", "")),
                retrieval_source="bm25",
            )
        )
    return corpus_texts, docs


# 这个函数把 BM25 语料构建成索引。
# 之后每次关键词检索都可以直接使用这个索引来打分。
# 索引本身比较重，所以这里也做缓存。
@lru_cache(maxsize=16)
def load_bm25_index(collection_name: str, signature: str) -> BM25Okapi:
    corpus_texts, _ = load_bm25_corpus(collection_name, signature)
    return BM25Okapi(corpus_texts)


CHINESE_STOPWORDS = {
    "什么", "哪些", "如何", "为什么", "可以", "是否", "怎么", "请问", "一下", "主要", "依据", "指标", "内容", "问题", "以及",
    "和", "与", "的", "是", "了", "在", "对", "有", "及", "把", "给", "请", "帮", "我", "你", "它", "这", "那",
    "请帮我", "请帮忙", "告诉我", "说明", "介绍", "有关", "相关"
}

QUERY_SYNONYMS = {
    "高血压": ["血压高", "血压升高", "hypertension", "收缩压升高"],
    "糖尿病": ["血糖", "diabetes"],
    "冠心病": ["心绞痛", "心脏病", "coronary"],
    "危险分层": ["分层", "风险分层", "风险评估", "风险等级"],
    "单纯收缩期高血压": ["收缩压升高", "单纯收缩性高血压", "isolated systolic hypertension"],
    "收缩期": ["收缩压", "收缩期血压"],
    "指标": ["标准", "参数", "特征", "依据"],
}


# 这个函数负责分词。
# 它把中文和英文内容拆成适合 BM25 检索的 token 列表。
# 同时会去掉一些常见停用词，减少无意义词对检索的干扰。
def tokenize(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text.lower()).strip()
    if not text:
        return []
    chunks = re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9]+", text)
    tokens: List[str] = []
    for chunk in chunks:
        if re.fullmatch(r"[\u4e00-\u9fff]+", chunk):
            if len(chunk) <= 2:
                tokens.append(chunk)
            else:
                tokens.append(chunk)
                tokens.extend(chunk[i:i+2] for i in range(len(chunk) - 1))
        else:
            tokens.append(chunk)
    return [t for t in tokens if t and t not in CHINESE_STOPWORDS]


# 这个函数会把用户问题扩展成更适合检索的查询文本。
# 它会加入同义词和分词结果，让召回更容易命中相关内容。
# 这一步相当于给检索器“补充搜索关键词”。
def build_query_text(question: str) -> str:
    question = question.strip()
    if not question:
        return question
    expanded = [question]
    for key, syns in QUERY_SYNONYMS.items():
        if key in question:
            expanded.extend(syns)
    tokens = tokenize(question)
    expanded.extend(tokens)
    deduped = []
    seen = set()
    for item in expanded:
        item = str(item).strip()
        if item and item not in seen:
            seen.add(item)
            deduped.append(item)
    return " | ".join(deduped)


# 这个函数负责把 top_k 控制在安全范围内。
# 这样可以防止前端传入过大或非法数值影响检索稳定性。
# 它属于检索参数的保护层。
def safe_top_k(value: int, minimum: int = 1, maximum: int = 50) -> int:
    try:
        return max(minimum, min(maximum, int(value)))
    except Exception:
        return minimum


# 这个函数负责把自然语言问题编码成向量。
# 向量会被送到 Milvus 里做语义相似度检索。
# 这是向量召回的起点。
def embed_query(question: str) -> List[float]:
    embedder = load_embedder()
    query_text = build_query_text(question)
    result = embedder.encode([query_text], batch_size=1, max_length=256)
    vec = np.asarray(result["dense_vecs"], dtype=np.float32)[0]
    return vec.tolist()


# 这个函数负责统计检索结果的来源数量。
# 它可以告诉我们向量召回、BM25、长期记忆各贡献了多少结果。
# 这些统计信息后面会回传给前端或写入日志。
def collect_retrieval_stats(*doc_lists: List[RetrievedDoc]) -> RetrievalStats:
    stats = RetrievalStats()
    for docs in doc_lists:
        for doc in docs:
            if doc.memory_type == "memory":
                stats.memory += 1
            elif doc.score > 0:
                stats.dense += 1
            else:
                stats.bm25 += 1
    stats.merged = stats.dense + stats.bm25 + stats.memory
    return stats


# 这个函数负责执行向量检索，也就是 dense retrieval。
# 它的输入是向量，输出是语义最接近的文档列表。
# 这部分负责“找语义相似内容”。
def search_dense(collection: Collection, query_vector: List[float], top_k: int, retrieval_source: str = "dense") -> List[RetrievedDoc]:
    output_fields = ["source", "domain", "title", "role", "keywords", "content", "vector_text"]
    results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"ef": 32}},
        limit=safe_top_k(top_k, 1, 50),
        output_fields=output_fields,
    )
    docs: List[RetrievedDoc] = []
    for hit in results[0]:
        entity = hit.entity
        docs.append(
            RetrievedDoc(
                id=str(hit.id),
                score=float(hit.score),
                source=str(entity.get("source", "")),
                domain=str(entity.get("domain", "")),
                title=str(entity.get("title", "")),
                role=str(entity.get("role", "")),
                keywords=str(entity.get("keywords", "")),
                content=str(entity.get("content", "")),
                vector_text=str(entity.get("vector_text", "")),
                retrieval_source=retrieval_source,
            )
        )
    return docs


# 这个函数负责 BM25 关键词召回。
# 它更擅长查找关键词匹配明显、术语明确的内容。
# 这部分负责“找字面匹配内容”。
def search_bm25(collection: Collection, question: str, bm25_top_k: int, retrieval_source: str = "bm25") -> List[RetrievedDoc]:
    try:
        signature = get_collection_signature(collection)
        corpus_texts, docs = load_bm25_corpus(collection.name, signature)
        if not corpus_texts:
            return []

        for doc in docs:
            doc.retrieval_source = retrieval_source

        query_tokens = tokenize(build_query_text(question))
        bm25 = load_bm25_index(collection.name, signature)
        scores = bm25.get_scores(query_tokens)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:safe_top_k(bm25_top_k, 1, 200)]
    except Exception:
        logger.exception("bm25 search failed")
        return []
    output: List[RetrievedDoc] = []
    for doc, score in ranked:
        doc.score = float(score)
        output.append(doc)
    return output


# 这个函数负责从长期记忆集合里做召回。
# 它用于把之前保存的重要记忆重新找回来，参与本轮回答。
# 这部分负责“找曾经记住的内容”。
def search_long_term_memory(collection_name: str, query_text: str, top_k: int) -> List[RetrievedDoc]:
    if not utility.has_collection(collection_name):
        return []
    try:
        collection = get_collection(collection_name)
    except Exception:
        return []
    query_vector = embed_query(query_text)
    docs = search_dense(collection, query_vector, top_k, retrieval_source="memory")
    for doc in docs:
        doc.memory_type = "memory"
    return docs


# 这个函数负责合并多个来源的候选文档。
# 它会去重并按分数排序，得到一份统一的候选集合。
# 这一步是把不同检索通道的结果汇总到一起。
def merge_candidates(*doc_lists: List[RetrievedDoc]) -> List[RetrievedDoc]:
    merged: Dict[str, RetrievedDoc] = {}
    for docs in doc_lists:
        for doc in docs:
            key = f"{doc.retrieval_source}:{doc.memory_type}:{doc.id}:{doc.session_id}:{doc.user_id}"
            if key not in merged or doc.score > merged[key].score:
                merged[key] = doc
    values = list(merged.values())
    values.sort(key=lambda d: (d.score, len((d.content or d.vector_text or ""))), reverse=True)
    return values


# 这个函数负责重排序，也就是 rerank。
# 它会重新评估“问题和文档”的匹配程度，把真正相关的排到前面。
# 这一步能进一步提升最终上下文质量。
def rerank(question: str, docs: List[RetrievedDoc], rerank_top_k: int) -> List[RetrievedDoc]:
    if not docs:
        return []
    reranker = load_reranker()
    pairs = [[question, d.vector_text or d.content or d.title] for d in docs]
    try:
        with _RERANK_LOCK:
            scores = reranker.compute_score(pairs)
    except Exception:
        logger.exception("rerank failed, fallback to original order")
        return docs[:safe_top_k(rerank_top_k, 1, 50)]
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    output: List[RetrievedDoc] = []
    for doc, score in ranked[:safe_top_k(rerank_top_k, 1, 50)]:
        doc.score = float(score)
        output.append(doc)
    return output


# 这个函数负责把检索到的多个文档拼成上下文文本。
# 它会控制每段长度和总长度，避免 prompt 过长。
# 最终会给大模型提供一个可直接阅读的资料块。
def build_context(docs: List[RetrievedDoc], max_chars_per_doc: int = 500, max_total_chars: int = 2600) -> str:
    blocks = []
    total_chars = 0
    for idx, doc in enumerate(docs, start=1):
        content = (doc.content or doc.vector_text or "").strip()
        if len(content) > max_chars_per_doc:
            content = content[:max_chars_per_doc] + "..."
        block = (
            f"[片段 {idx}]\n"
            f"类型: {doc.memory_type}\n"
            f"来源分类: {doc.retrieval_source}\n"
            f"ID: {doc.id}\n"
            f"会话: {doc.session_id}\n"
            f"用户: {doc.user_id}\n"
            f"来源: {doc.source}\n"
            f"领域: {doc.domain}\n"
            f"标题: {doc.title}\n"
            f"角色: {doc.role}\n"
            f"关键词: {doc.keywords}\n"
            f"内容: {content}\n"
        )
        if total_chars + len(block) > max_total_chars:
            break
        blocks.append(block)
        total_chars += len(block)
    return "\n".join(blocks)


# 这个函数负责把问题、角色和历史拼成记忆检索用的查询文本。
# 它让长期记忆召回时也能考虑当前角色和上下文。
# 这样记忆检索就不会只看单独一句问题。
def build_memory_query(question: str, role_name: str, history_text: str) -> str:
    return f"角色:{role_name}\n历史:{history_text}\n问题:{question}".strip()


# 这个函数负责构造最终喂给大模型的 prompt。
# 它会把角色设定、短期历史和检索内容一起组织起来。
# 这就是模型真正“看见”的输入内容。
def build_prompt(question: str, role_config, short_history: str, context: str) -> str:
    return (
        f"你当前扮演的角色是：{role_config.persona}\n"
        f"回答风格：{role_config.style}\n"
        f"约束规则：{role_config.rules}\n\n"
        f"短期对话记忆：\n{short_history or '无'}\n\n"
        f"检索到的长期记忆与知识：\n{context or '无'}\n\n"
        f"用户问题：{question}\n\n"
        "请结合角色设定、短期记忆和检索结果进行回答。"
        "如果资料不足，请明确说明资料不足，并给出建议。"
    )


# 这个函数负责生成 prompt 的预览文本。
# 它通常用于日志或前端调试，避免把整段 prompt 全部暴露出来。
# 这样既能排查问题，又不会让日志太长。
def build_prompt_preview(prompt: str, max_chars: int = 500) -> str:
    prompt = prompt.strip()
    return prompt if len(prompt) <= max_chars else prompt[:max_chars] + "..."
