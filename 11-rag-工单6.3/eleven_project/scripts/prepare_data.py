"""
从 Milvus 导出 chunk → LLM 生成 query → BM25 挖 hard negative → 输出训练数据
用法: python scripts/prepare_data.py
"""

import json
import os
import sys
import random
import requests
from typing import List, Dict, Tuple
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── 把 RAG 项目加到 sys.path ──
RAG_PROJECT = "/mnt/d/Desktop/rag-hermes/teams/team/project"
sys.path.insert(0, RAG_PROJECT)

from src.store.milvus_store import MilvusStore

# ── 配置 ──
MILVUS_COLLECTION = "ccf_chunks"
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")

OUTPUT_DIR = Path(__file__).parent.parent / "data"
TRAIN_RATIO = 0.9  # 训练集占比
QUERIES_PER_CHUNK = 2  # 每个 chunk 生成几个 query
NEG_TOP_K = 20  # BM25 检索 top-k 用于挖负例
MAX_CHUNKS = 8000  # 最多处理多少 chunk（太多太慢）
CONCURRENT_WORKERS = 10  # 并发请求数


def export_chunks_from_milvus() -> List[Dict]:
    """从 Milvus 导出所有 chunk 文本"""
    print("正在从 Milvus 导出 chunk ...")
    store = MilvusStore(collection_name=MILVUS_COLLECTION)
    store.connect()
    store._collection.flush()
    total = store._collection.num_entities
    print(f"Milvus 中共 {total} 条 chunk")

    # 分批查询（用 id 范围，Milvus offset 上限 16384）
    all_chunks = []
    batch_size = 2000

    # 先拿一条确定起始 id
    first = store._collection.query(
        expr="id >= 0",
        output_fields=["id"],
        limit=1,
    )
    if not first:
        store.close()
        return []

    current_id = first[0]["id"] - 1
    pbar = tqdm(total=total, desc="导出 chunk")
    while True:
        results = store._collection.query(
            expr=f"id > {current_id}",
            output_fields=["id", "text", "source_file", "chunk_index", "section_path"],
            limit=batch_size,
        )
        if not results:
            break
        for r in results:
            text = r.get("text", "").strip()
            if len(text) < 50:
                continue
            all_chunks.append({
                "id": r["id"],
                "text": text,
                "source_file": r.get("source_file", ""),
                "chunk_index": r.get("chunk_index", 0),
                "section_path": r.get("section_path", ""),
            })
            current_id = r["id"]
        pbar.update(len(results))
    pbar.close()

    store.close()
    print(f"导出有效 chunk: {len(all_chunks)} 条")
    return all_chunks


def generate_queries_for_chunk(text: str, source_file: str = "") -> List[str]:
    """用 LLM 对一个 chunk 生成 2-3 个查询问题"""
    prompt = f"""你是一个银行-保险领域文档分析专家。请根据以下文档片段，生成 {QUERIES_PER_CHUNK} 个用户可能会问的问题。
要求：
1. 问题要具体，能从文档片段中找到答案
2. 问题类型多样化（事实查询、概括查询、对比查询等）
3. 问题要自然，像真实用户会问的
4. 只输出问题，每行一个，不要编号

文档片段：
{text[:1500]}

文档来源：{source_file}"""

    try:
        base = DEEPSEEK_BASE_URL.rstrip("/")
        if not base.endswith("/v1"):
            base = base + "/v1"
        resp = requests.post(
            f"{base}/chat/completions",
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": DEEPSEEK_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 300,
            },
            timeout=30,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        queries = [q.strip() for q in content.strip().split("\n") if q.strip()]
        # 清理编号前缀
        cleaned = []
        for q in queries:
            q = q.lstrip("0123456789.、）) ")
            if len(q) >= 4:
                cleaned.append(q)
        return cleaned[:QUERIES_PER_CHUNK]
    except Exception as e:
        print(f"  LLM 生成失败: {e}")
        return []


def build_bm25_index(chunks: List[Dict]) -> Dict[str, List[str]]:
    """构建 BM25 索引用于挖负例"""
    import jieba
    from rank_bm25 import BM25Okapi

    print("构建 BM25 索引 ...")
    corpus = [list(jieba.cut(c["text"])) for c in chunks]
    bm25 = BM25Okapi(corpus)
    return bm25


def find_hard_negatives(
    query: str,
    positive_text: str,
    bm25,
    chunks: List[Dict],
    top_k: int = NEG_TOP_K,
) -> List[str]:
    """用 BM25 检索找难负例：top-k 里排除正例"""
    import jieba

    query_tokens = list(jieba.cut(query))
    scores = bm25.get_scores(query_tokens)

    # 按分数排序，取 top-k
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    negatives = []
    for idx in ranked_indices[:top_k]:
        candidate = chunks[idx]["text"]
        # 排除正例（文本高度重叠的）
        if candidate == positive_text:
            continue
        # 排除文本重叠超过 60% 的（可能是同一 chunk 的变体）
        overlap = len(set(candidate) & set(positive_text)) / max(len(set(positive_text)), 1)
        if overlap > 0.6:
            continue
        negatives.append(candidate)
        if len(negatives) >= 3:  # 每个 query 找 3 个负例
            break

    return negatives


def generate_training_data(chunks: List[Dict]) -> List[Dict]:
    """生成完整训练数据：query + pos + neg（并发版）"""
    # 抽样：chunk 太多时随机取 MAX_CHUNKS 个
    if len(chunks) > MAX_CHUNKS:
        print(f"chunk 数量 {len(chunks)} 超过上限 {MAX_CHUNKS}，随机抽样 ...")
        random.seed(42)
        chunks = random.sample(chunks, MAX_CHUNKS)

    print(f"实际处理 chunk: {len(chunks)} 条")

    # 构建 BM25 索引
    bm25 = build_bm25_index(chunks)

    # ── 第一步：并发生成 query ──
    print(f"并发生成 query（{CONCURRENT_WORKERS} 线程） ...")
    chunk_queries = []  # [(chunk, queries), ...]

    def _gen_one(chunk):
        queries = generate_queries_for_chunk(chunk["text"], chunk["source_file"])
        return chunk, queries

    with ThreadPoolExecutor(max_workers=CONCURRENT_WORKERS) as pool:
        futures = {pool.submit(_gen_one, c): c for c in chunks}
        for future in tqdm(as_completed(futures), total=len(chunks), desc="生成 query"):
            chunk, queries = future.result()
            if queries:
                chunk_queries.append((chunk, queries))

    print(f"成功生成 query 的 chunk: {len(chunk_queries)} 条")

    # ── 第二步：组装训练数据（BM25 挖负例） ──
    print("组装训练数据（BM25 挖负例） ...")
    training_data = []
    for chunk, queries in tqdm(chunk_queries, desc="挖负例"):
        for query in queries:
            negatives = find_hard_negatives(query, chunk["text"], bm25, chunks)
            if not negatives:
                continue
            training_data.append({
                "query": query,
                "pos": [chunk["text"]],
                "neg": negatives,
            })

    print(f"生成训练样本: {len(training_data)} 条")
    return training_data


def save_data(training_data: List[Dict], chunks: List[Dict]):
    """拆分并保存 train/test 数据 + 评估数据"""
    random.seed(42)
    random.shuffle(training_data)

    split_idx = int(len(training_data) * TRAIN_RATIO)
    train_data = training_data[:split_idx]
    test_data = training_data[split_idx:]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_path = OUTPUT_DIR / "train.jsonl"
    test_path = OUTPUT_DIR / "test.jsonl"

    for path, data in [(train_path, train_data), (test_path, test_data)]:
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"训练集: {train_path} ({len(train_data)} 条)")
    print(f"测试集: {test_path} ({len(test_data)} 条)")

    # ── 生成评估数据（evaluate.py 需要的格式） ──
    # corpus.jsonl: 所有 chunk 的 docid + text
    corpus_path = OUTPUT_DIR / "test_corpus.jsonl"
    with open(corpus_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps({
                "docid": str(chunk["id"]),
                "text": chunk["text"],
            }, ensure_ascii=False) + "\n")
    print(f"语料库: {corpus_path} ({len(chunks)} 条)")

    # test_queries.jsonl: 测试集 query + positive_passages
    # 从 test_data 中取 query 和对应的正例，关联到 corpus docid
    test_queries_path = OUTPUT_DIR / "test_queries.jsonl"
    # 建立 text → chunk id 映射
    text_to_id = {c["text"]: str(c["id"]) for c in chunks}
    with open(test_queries_path, "w", encoding="utf-8") as f:
        for item in test_data:
            pos_texts = item.get("pos", [])
            pos_passages = []
            for pt in pos_texts:
                docid = text_to_id.get(pt)
                if docid:
                    pos_passages.append({"docid": docid, "text": pt})
            if pos_passages:
                f.write(json.dumps({
                    "query": item["query"],
                    "positive_passages": pos_passages,
                }, ensure_ascii=False) + "\n")
    print(f"测试 query: {test_queries_path}")


def main():
    # 检查 API Key
    if not DEEPSEEK_API_KEY:
        print("错误: 请设置环境变量 DEEPSEEK_API_KEY")
        print("  export DEEPSEEK_API_KEY='your-key-here'")
        sys.exit(1)

    # 1. 导出 chunk
    chunks = export_chunks_from_milvus()
    if not chunks:
        print("错误: Milvus 中没有 chunk 数据")
        sys.exit(1)

    # 2. 生成训练数据
    training_data = generate_training_data(chunks)

    # 3. 保存
    if training_data:
        save_data(training_data, chunks)
        print("数据准备完成!")
    else:
        print("错误: 未能生成训练数据")
        sys.exit(1)


if __name__ == "__main__":
    main()
