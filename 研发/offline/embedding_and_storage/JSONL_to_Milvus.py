# -*- coding: utf-8 -*-
"""将已清洗好的 JSONL 数据向量化并入库到 Milvus。

默认适配 RAG 知识库常见字段：
- id
- source
- domain
- title
- role
- keywords
- content
- vector_text
- embedding（入库时生成）

如果你的 JSONL 字段名不同，可以在 `FIELD_MAP` 里修改。
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
from FlagEmbedding import BGEM3FlagModel
from pymilvus import DataType, MilvusClient
from tqdm import tqdm

try:
    import torch
except ImportError:
    torch = None

# ===================== 默认配置 =====================
MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "pdf_chunks"
VECTOR_DIM = 1024
MODEL_PATH = r"D:\models\bge-m3"
BATCH_SIZE = 256
EMBED_BATCH_SIZE = 32
MAX_TEXT_LEN = 65000
MAX_TITLE_LEN = 200
MAX_SOURCE_LEN = 128
MAX_DOMAIN_LEN = 32
MAX_ROLE_LEN = 64
MAX_KEYWORDS_LEN = 512

# 直接运行时的默认输入输出路径（可按需修改）
DEFAULT_INPUT_PATH = r"/研发/knowledge/medical/medical_10000_chunks.jsonl"
DEFAULT_OUTPUT_PATH = r"/研发/knowledge/medical/标准化备份.jsonl"
DEFAULT_ERROR_PATH = r"/研发\knowledge\medical\入库失败记录.jsonl"
DEFAULT_DEVICE = "cuda:0"

# 你清洗后的 JSONL 字段映射
FIELD_MAP = {
    "id": "id",
    "source": "source",
    "domain": "domain",
    "title": "title",
    "role": "role",
    "keywords": "keywords",
    "content": "content",
    "vector_text": "vector_text",
}


def normalize_text(text: str) -> str:
    text = str(text or "").replace("\u3000", " ")
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def shorten_title(title: str, content: str) -> str:
    title = normalize_text(title)
    content = normalize_text(content)

    if not title:
        # 用内容前半段自动生成更适合检索的标题摘要
        title = content[:120]
        for sep in ["。", "！", "？", "；", "\n"]:
            pos = title.find(sep)
            if 20 <= pos <= 120:
                title = title[:pos + 1]
                break

    if len(title) > MAX_TITLE_LEN:
        title = title[:MAX_TITLE_LEN].rstrip()
    return title


def summarize_title(content: str) -> str:
    content = normalize_text(content)
    if not content:
        return ""
    title = content[:120]
    for sep in ["。", "！", "？", "；", "\n"]:
        pos = title.find(sep)
        if 20 <= pos <= 120:
            title = title[:pos + 1]
            break
    return title[:MAX_TITLE_LEN].rstrip()


def build_record(item: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """把任意清洗后的 JSONL 记录整理成统一入库结构。"""
    def pick(key: str, default: str = "") -> str:
        src_key = FIELD_MAP.get(key, key)
        return str(item.get(src_key, default) or default).strip()

    record_id = pick("id") or f"doc_{idx:06d}"
    source = pick("source") or "jsonl_source"
    domain = pick("domain") or "通用"
    role = pick("role") or "unknown"
    content = normalize_text(pick("content"))[:MAX_TEXT_LEN]
    vector_text = normalize_text(pick("vector_text") or content)[:MAX_TEXT_LEN]

    if not content:
        content = vector_text
    if not vector_text:
        vector_text = content

    raw_title = pick("title")
    title = shorten_title(raw_title, content)
    if not title:
        title = summarize_title(content)
    keywords = normalize_text(pick("keywords"))[:MAX_KEYWORDS_LEN]

    source = source[:MAX_SOURCE_LEN]
    domain = domain[:MAX_DOMAIN_LEN]
    role = role[:MAX_ROLE_LEN]
    keywords = keywords[:MAX_KEYWORDS_LEN]

    return {
        "id": record_id,
        "source": source,
        "domain": domain,
        "title": title,
        "role": role,
        "keywords": keywords,
        "content": content,
        "vector_text": vector_text,
    }


def ensure_collection(client: MilvusClient, collection_name: str, vector_dim: int):
    if client.has_collection(collection_name):
        return

    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=128)
    schema.add_field(field_name="domain", datatype=DataType.VARCHAR, max_length=32)
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="role", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="keywords", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="vector_text", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=vector_dim)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200},
    )

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
    )


def get_device(prefer_cuda: bool = True) -> str:
    if prefer_cuda and torch is not None and torch.cuda.is_available():
        return DEFAULT_DEVICE
    return "cpu"


def main(input_path: str = DEFAULT_INPUT_PATH, output_path: str = DEFAULT_OUTPUT_PATH,
         collection_name: str = COLLECTION_NAME, milvus_uri: str = MILVUS_URI,
         model_path: str = MODEL_PATH, vector_dim: int = VECTOR_DIM,
         device: str | None = None):
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    run_device = device or get_device(prefer_cuda=True)
    use_fp16 = run_device.startswith("cuda")

    print(f"加载 BGE-m3 模型中... 设备: {run_device}")
    model = BGEM3FlagModel(model_path, use_fp16=use_fp16, device=run_device)

    print("读取 JSONL 数据中...")
    raw_data = load_jsonl(str(input_path))
    print(f"✅ 读取到 {len(raw_data)} 条记录")

    print("整理入库字段中...")
    records = [build_record(item, idx) for idx, item in enumerate(raw_data)]
    records = [r for r in records if r["content"] or r["vector_text"]]
    print(f"✅ 有效记录 {len(records)} 条")

    # 再次保险截断，防止 Milvus varchar 超长报错
    for r in records:
        r["id"] = str(r["id"])[:64]
        r["source"] = str(r["source"])[:MAX_SOURCE_LEN]
        r["domain"] = str(r["domain"])[:MAX_DOMAIN_LEN]
        r["title"] = normalize_text(str(r["title"]))[:MAX_TITLE_LEN]
        if len(r["title"]) > 80:
            r["title"] = r["title"][:80].rstrip()
        r["role"] = str(r["role"])[:MAX_ROLE_LEN]
        r["keywords"] = str(r["keywords"])[:MAX_KEYWORDS_LEN]
        r["content"] = str(r["content"])[:MAX_TEXT_LEN]
        r["vector_text"] = str(r["vector_text"])[:MAX_TEXT_LEN]

    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"✅ 标准化备份已保存到: {out_path}")

    print("生成向量中...")
    texts = [r["vector_text"] for r in records]
    embeddings = []
    for i in tqdm(range(0, len(texts), EMBED_BATCH_SIZE), desc="Embedding"):
        batch = texts[i:i + EMBED_BATCH_SIZE]
        batch_vec = model.encode(batch, batch_size=EMBED_BATCH_SIZE)["dense_vecs"].astype(np.float32)
        embeddings.append(batch_vec)
    embeddings = np.vstack(embeddings) if embeddings else np.empty((0, vector_dim), dtype=np.float32)

    client = MilvusClient(uri=milvus_uri)
    ensure_collection(client, collection_name, vector_dim)

    print(f"开始入库到 `{collection_name}` ...")
    for i in tqdm(range(0, len(records), BATCH_SIZE), desc="Insert"):
        batch_records = records[i:i + BATCH_SIZE]
        batch_embeddings = embeddings[i:i + BATCH_SIZE]
        data = [{**r, "embedding": emb.tolist()} for r, emb in zip(batch_records, batch_embeddings)]
        if data:
            try:
                client.insert(collection_name=collection_name, data=data)
            except Exception as e:
                print(f"⚠️ 批次 {i // BATCH_SIZE} 插入失败，开始逐条排查：{e}")
                bad_rows = []
                for row_idx, row in enumerate(data):
                    try:
                        client.insert(collection_name=collection_name, data=[row])
                    except Exception as row_err:
                        row = dict(row)
                        row["error"] = str(row_err)
                        bad_rows.append(row)
                        print(
                            f"❌ 跳过失败记录 batch={i // BATCH_SIZE}, row={row_idx}, id={row.get('id')}, "
                            f"title_len={len(str(row.get('title', '')))}, error={row_err}"
                        )
                if bad_rows:
                    error_path = Path(DEFAULT_ERROR_PATH)
                    error_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(error_path, "a", encoding="utf-8") as f:
                        for row in bad_rows:
                            f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    print(f"⚠️ 失败记录已追加保存到: {error_path}")

    total = client.get_collection_stats(collection_name)["row_count"]
    print("\n🎉 入库完成！")
    print(f"✅ 当前集合 `{collection_name}` 总数据量：{total} 条")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JSONL 清洗数据入库 Milvus")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="清洗后的 JSONL 文件路径")
    parser.add_argument("--collection", default=COLLECTION_NAME, help="Milvus 集合名")
    parser.add_argument("--milvus-uri", default=MILVUS_URI, help="Milvus URI")
    parser.add_argument("--model-path", default=MODEL_PATH, help="BGE-m3 模型路径")
    parser.add_argument("--vector-dim", type=int, default=VECTOR_DIM, help="向量维度")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="可选：保存标准化后的 JSONL 备份路径")
    parser.add_argument("--device", default="", help="运行设备，例如 cuda:0 或 cpu；留空则自动选择")
    args = parser.parse_args()

    chosen_device = args.device.strip() or None
    main(
        input_path=args.input,
        output_path=args.output,
        collection_name=args.collection,
        milvus_uri=args.milvus_uri,
        model_path=args.model_path,
        vector_dim=args.vector_dim,
        device=chosen_device,
    )
