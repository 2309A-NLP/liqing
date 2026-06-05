# -*- coding: utf-8 -*-
"""医疗对话数据清洗、知识库分块、向量化并入库 Milvus。

目标：
1. 面向 RAG 知识库，保留问诊语义和回答上下文；
2. 去除空白、重复、过短文本，生成更稳定的检索块；
3. 保持字段不变：id/source/domain/title/role/keywords/content/vector_text/embedding；
4. 自动创建 `pdf_chunks` 集合（如不存在）。
"""

import json
import re
from pathlib import Path

import numpy as np
from FlagEmbedding import BGEM3FlagModel
from pymilvus import DataType, MilvusClient
from tqdm import tqdm

# ===================== 路径配置 =====================
INPUT_PATH = r"D:\Desktop\RAG最新\医疗数据\sampled_10000_medical.jsonl"
OUTPUT_CLEAN = r"D:\Desktop\RAG最新\医疗数据\medical_10000_chunks.jsonl"
MODEL_PATH = r"D:\models\bge-m3"

# ===================== Milvus 配置 =====================
MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "pdf_chunks"
VECTOR_DIM = 1024
BATCH_SIZE = 256
EMBED_BATCH_SIZE = 32
MIN_CHUNK_LEN = 40
MAX_CONTENT_LEN = 1200

print("加载 BGE-m3 模型中...")
model = BGEM3FlagModel(MODEL_PATH, use_fp16=False, device="cuda:0")


def normalize_text(text: str) -> str:
    text = str(text or "").replace("\u3000", " ")
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def collapse_dialogue(question: str, answer: str) -> str:
    question = normalize_text(question)
    answer = normalize_text(answer)
    if question and answer:
        return f"患者提问：{question}\n医生回复：{answer}"
    if question:
        return f"患者提问：{question}"
    if answer:
        return f"医生回复：{answer}"
    return ""


def split_qa_text(text: str):
    text = normalize_text(text)
    if not text:
        return []

    # 对问答进行轻量切分，但保留上下文完整性
    parts = re.split(r"(?<=[。！？；\n])", text)
    parts = [p.strip() for p in parts if p.strip()]
    chunks = []
    current = ""
    for part in parts:
        if len(current) + len(part) <= MAX_CONTENT_LEN:
            current += part
        else:
            if current:
                chunks.append(current.strip())
            current = part
    if current:
        chunks.append(current.strip())
    return chunks


def extract_keywords(text: str, question: str):
    kws = []
    for token in re.split(r"[，,。！？；：\s]+", f"{question} {text}"):
        token = token.strip()
        if len(token) >= 2 and token not in kws:
            kws.append(token)
    return kws[:8]


def infer_role(text: str) -> str:
    if any(x in text for x in ["疾病", "症状", "诊断", "治疗", "用药", "检查"]):
        return "medical_qa"
    return "doctor-patient"


print("开始清洗分块，适配 RAG 知识库...")
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    raw_data = [json.loads(line) for line in f if line.strip()]

print(f"✅ 读取到 {len(raw_data)} 条医疗数据，开始格式化...")
chunks = []
seen = set()

for idx, item in enumerate(raw_data):
    q = str(item.get("patient_question", "")).strip()
    a = str(item.get("doctor_answer", "")).strip()
    merged = collapse_dialogue(q, a)
    if len(merged) < MIN_CHUNK_LEN:
        continue

    for sub_idx, piece in enumerate(split_qa_text(merged)):
        piece = piece.strip()
        if len(piece) < MIN_CHUNK_LEN:
            continue
        dedup_key = re.sub(r"\s+", "", piece)[:220]
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        chunks.append({
            "id": f"med_{idx:06d}_{sub_idx:02d}",
            "source": "medical_dialogue",
            "domain": "医疗",
            "title": (q[:100] if q else piece[:100]),
            "role": infer_role(piece),
            "keywords": ",".join(extract_keywords(piece, q)),
            "content": piece[:65000],
            "vector_text": piece[:65000],
        })

print(f"✅ 清洗分块完成：{len(chunks)} 个知识块")

Path(OUTPUT_CLEAN).parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_CLEAN, "w", encoding="utf-8") as f:
    for c in chunks:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")

print("生成向量中...")
texts = [c["vector_text"] for c in chunks]
embeddings = []
for i in tqdm(range(0, len(texts), EMBED_BATCH_SIZE), desc="Embedding"):
    batch_texts = texts[i:i + EMBED_BATCH_SIZE]
    batch_vecs = model.encode(batch_texts, batch_size=EMBED_BATCH_SIZE)["dense_vecs"].astype(np.float32)
    embeddings.append(batch_vecs)
embeddings = np.vstack(embeddings) if embeddings else np.empty((0, VECTOR_DIM), dtype=np.float32)

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
    client.create_collection(collection_name=COLLECTION_NAME, schema=schema, index_params=index_params)
    print(f"✅ 已创建 `{COLLECTION_NAME}` 集合")

print(f"开始入库 {len(chunks)} 条数据到 `{COLLECTION_NAME}` ...")
for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Insert"):
    batch_chunks = chunks[i:i + BATCH_SIZE]
    batch_embeddings = embeddings[i:i + BATCH_SIZE]
    data = [{**c, "embedding": emb.tolist()} for c, emb in zip(batch_chunks, batch_embeddings)]
    client.insert(collection_name=COLLECTION_NAME, data=data)

total = client.get_collection_stats(COLLECTION_NAME)["row_count"]
print("\n🎉 全部完成！")
print(f"✅ 医疗数据已清洗、合理分块、向量化并入库成功")
print(f"✅ 当前集合 `{COLLECTION_NAME}` 总数据量：{total} 条")
print(f"✅ 备份文件：{OUTPUT_CLEAN}")
