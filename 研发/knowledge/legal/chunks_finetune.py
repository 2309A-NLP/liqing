# -*- coding: utf-8 -*-
"""法律 PDF 清洗、合理分块、向量化并入库 Milvus。

目标：
1. 面向 RAG 知识库，尽量保留条文/章节语义边界；
2. 清洗 PDF 噪声、修复断行、去重；
3. 保持现有字段不变：id/source/domain/title/role/keywords/content/vector_text/embedding；
4. 入库到 `pdf_chunks` 集合，若不存在则自动创建。
"""

import json
import re
from pathlib import Path

import numpy as np
from FlagEmbedding import BGEM3FlagModel
from pymilvus import DataType, MilvusClient
from tqdm import tqdm

# ===================== 路径配置 =====================
INPUT_PATH = r"/研发/knowledge/legal/finetune_dataset_20000.jsonl"
OUTPUT_CLEAN = r"D:\Desktop\RAG最新\PDF清洗版存储\法律数据\finetune_dataset_20000_chunks.jsonl"
MODEL_PATH = r"D:\models\bge-m3"

# ===================== Milvus 配置 =====================
MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "pdf_chunks"
VECTOR_DIM = 1024
BATCH_SIZE = 256
EMBED_BATCH_SIZE = 32
MAX_CHUNK_LEN = 900
MIN_CHUNK_LEN = 80
CHUNK_OVERLAP = 120

# ===================== 模型加载 =====================
print("加载 BGE-m3 向量模型...")
model = BGEM3FlagModel(MODEL_PATH, use_fp16=False, device="cuda:0")

# ===================== 清洗与分块工具 =====================
SECTION_PATTERNS = [
    r"^第[一二三四五六七八九十百千0-9]+编",
    r"^第[一二三四五六七八九十百千0-9]+章",
    r"^第[一二三四五六七八九十百千0-9]+节",
    r"^第[一二三四五六七八九十百千0-9]+条",
    r"^\d+\.\s*",
]
NOISE_PATTERNS = [
    r"^第\s*\d+\s*页$",
    r"^\d+\s*/\s*\d+$",
    r"^版权所有.*$",
    r"^目录$",
    r"^前言.*$",
    r"^http[s]?://.*$",
]

LAW_KEYWORDS = [
    "民事责任", "合同", "物权", "侵权", "婚姻", "继承", "劳动", "行政", "诉讼", "仲裁",
    "自然人", "法人", "监护", "债权", "债务", "赔偿", "违约", "权利", "义务", "程序",
]


def normalize_text(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
    text = re.sub(r"-\n(?=[a-zA-Z\u4e00-\u9fff])", "", text)  # 修复断词
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def remove_noise_lines(lines):
    result = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if any(re.match(pat, line) for pat in NOISE_PATTERNS):
            continue
        result.append(line)
    return result


def split_by_sections(lines):
    """先按章节/条文边界切分，再做长度控制。"""
    sections = []
    current = []
    for line in lines:
        if any(re.match(pat, line) for pat in SECTION_PATTERNS) and current:
            sections.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append("\n".join(current).strip())
    return [s for s in sections if s]


def sliding_chunks(text: str, max_len=MAX_CHUNK_LEN, overlap=CHUNK_OVERLAP):
    """面向 RAG 的重叠分块，避免切断法律语义。"""
    text = re.sub(r"\n{2,}", "\n", text).strip()
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_len, n)
        if end < n:
            window = text[start:end]
            cut = max(window.rfind("。"), window.rfind("；"), window.rfind("！"), window.rfind("？"))
            if cut > max_len * 0.4:
                end = start + cut + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def extract_keywords(text: str):
    kws = [kw for kw in LAW_KEYWORDS if kw in text]
    return list(dict.fromkeys(kws))


def infer_role(text: str):
    if re.search(r"^第.+条", text):
        return "legal_article"
    if "合同" in text:
        return "contract"
    if "婚姻" in text or "离婚" in text:
        return "family_law"
    if "继承" in text:
        return "inheritance"
    return "civil_law"


def infer_title(text: str):
    first_line = text.splitlines()[0].strip() if text.splitlines() else ""
    return first_line[:120]


# ===================== 读取与清洗 =====================
print("开始清洗法律数据，适配 RAG 知识库...")
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    raw_data = [json.loads(line) for line in f if line.strip()]

print(f"✅ 读取完成：共 {len(raw_data)} 条原始记录")
chunks = []
seen = set()

for idx, item in enumerate(raw_data):
    anchor = str(item.get("anchor", "")).strip()
    positive = str(item.get("positive", "")).strip()
    negative = str(item.get("negative", "")).strip()

    parts = []
    if anchor:
        parts.append(f"问题：{anchor}")
    if positive:
        parts.append(f"正向法条：{positive}")
    if negative:
        parts.append(f"对比法条：{negative}")

    merged = normalize_text("\n".join(parts))
    if not merged:
        continue

    lines = remove_noise_lines(merged.splitlines())
    sections = split_by_sections(lines)
    if not sections:
        sections = ["\n".join(lines)]

    for sec_idx, section in enumerate(sections):
        for chunk_idx, piece in enumerate(sliding_chunks(section)):
            piece = piece.strip()
            if len(piece) < MIN_CHUNK_LEN:
                continue
            dedup_key = re.sub(r"\s+", "", piece)[:200]
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            title = anchor[:100] if anchor else infer_title(piece)
            chunks.append({
                "id": f"law_{idx:06d}_{sec_idx:02d}_{chunk_idx:02d}",
                "source": "法律问答数据集",
                "domain": "法律",
                "title": title,
                "role": infer_role(piece),
                "keywords": ",".join(extract_keywords(piece) + ([anchor[:50]] if anchor else [])),
                "content": piece[:65000],
                "vector_text": piece[:65000],
            })

print(f"✅ 清洗分块完成：{len(chunks)} 个知识块")

# 保存清洗后的备份文件
Path(OUTPUT_CLEAN).parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_CLEAN, "w", encoding="utf-8") as f:
    for item in chunks:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

# ===================== 向量化 =====================
print("生成向量中...")
texts = [c["vector_text"] for c in chunks]
embeddings = []
for i in tqdm(range(0, len(texts), EMBED_BATCH_SIZE), desc="Embedding"):
    batch_texts = texts[i:i + EMBED_BATCH_SIZE]
    batch_vecs = model.encode(batch_texts, batch_size=EMBED_BATCH_SIZE)["dense_vecs"].astype(np.float32)
    embeddings.append(batch_vecs)
embeddings = np.vstack(embeddings) if embeddings else np.empty((0, VECTOR_DIM), dtype=np.float32)

# ===================== Milvus 入库 =====================
client = MilvusClient(uri=MILVUS_URI)

if not client.has_collection(COLLECTION_NAME):
    print(f"集合 `{COLLECTION_NAME}` 不存在，开始创建...")
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=128)
    schema.add_field(field_name="domain", datatype=DataType.VARCHAR, max_length=32)
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="role", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="keywords", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="vector_text", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200},
    )
    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )
    print(f"✅ 已创建 `{COLLECTION_NAME}` 集合")

print(f"开始入库 {len(chunks)} 条数据到 `{COLLECTION_NAME}` ...")
for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Insert"):
    batch_chunks = chunks[i:i + BATCH_SIZE]
    batch_embeddings = embeddings[i:i + BATCH_SIZE]
    data = [{**c, "embedding": emb.tolist()} for c, emb in zip(batch_chunks, batch_embeddings)]
    client.insert(collection_name=COLLECTION_NAME, data=data)

# ===================== 完成 =====================
total = client.get_collection_stats(COLLECTION_NAME)["row_count"]
print("\n🎉 全部完成！")
print(f"✅ 法律数据已清洗、合理分块、向量化并入库成功")
print(f"✅ 当前集合 `{COLLECTION_NAME}` 总数据量：{total} 条")
print(f"✅ 备份文件：{OUTPUT_CLEAN}")

