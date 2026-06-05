"""
图片描述增量入库 — 将 image_description 写入 Milvus
用法:
  python3 scripts/ingest_images.py
"""

import json, sys
from pathlib import Path

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

from src.store.milvus_store import MilvusStore


def load_descriptions() -> list:
    with open(PROJECT / "data" / "image_descriptions" / "descriptions.json", encoding="utf-8") as f:
        return json.load(f)


def get_max_chunk_index(store: MilvusStore, source_file: str) -> int:
    """从 Milvus 查现有最大 chunk_index"""
    store.connect()
    coll = store._collection
    try:
        coll.load()
    except Exception:
        pass
    try:
        results = coll.query(
            expr=f'source_file == "{source_file}"',
            output_fields=["chunk_index"],
            limit=1,
            offset=0,
        )
    except Exception:
        return 0
    if results:
        return max(r["chunk_index"] for r in results) + 1
    return 0


def main():
    descriptions = load_descriptions()
    print(f"共 {len(descriptions)} 条图片描述")

    # 按 source_file 分组
    from collections import defaultdict
    by_source = defaultdict(list)
    for d in descriptions:
        by_source[d["source_file"]].append(d)

    from src.embedder.embed import Embedder
    embedder = Embedder()
    store = MilvusStore()

    total = 0
    for source, descs in by_source.items():
        # 获取当前文档的最大 chunk_index
        max_idx = get_max_chunk_index(store, source)
        print(f"\n{source}: 现有 {max_idx} 条 chunk，新增 {len(descs)} 条图片描述")

        # 构造 chunk 并生成向量
        chunks = []
        for i, d in enumerate(descs):
            chunks.append({
                "text": d["description"],
                "page_no": d["page_no"],
                "source_file": source,
                "chunk_index": max_idx + i,
                "chunk_type": "image_description",
                "section_path": d.get("image_caption", ""),
            })

        # 生成向量
        texts = [c["text"] for c in chunks]
        embeddings = embedder.embed_batch(texts)

        # 入库
        store.insert_chunks(chunks, embeddings)
        total += len(chunks)
        print(f"  已入库 {len(chunks)} 条")

    print(f"\n完成！共入库 {total} 条 image_description chunk")


if __name__ == "__main__":
    main()
