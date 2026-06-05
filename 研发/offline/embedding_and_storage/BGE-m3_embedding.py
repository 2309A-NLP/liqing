# -*- coding: utf-8 -*-
# 读取标准 JSONL、拼接检索文本，并用 BGE-M3 生成向量。
import json
from pathlib import Path
from tqdm import tqdm
from FlagEmbedding import BGEM3FlagModel

# 输入的清洗后数据文件。
INPUT_PATH = r"/研发\knowledge\hypertension_clean.jsonl"
# 输出的带向量文件。
OUTPUT_PATH = r"/研发\knowledge\hypertension_clean_with_vectors.jsonl"
# 本地 BGE-M3 模型路径。
MODEL_PATH = r"D:\模型models\bge-m3"

# 初始化向量模型。
model = BGEM3FlagModel(
    MODEL_PATH,
    use_fp16=True
)

def _norm_text(value):
    # 将单值、列表、空值统一成可拼接的字符串。
    if value is None:
        return ""
    if isinstance(value, list):
        return "；".join(str(v).strip() for v in value if str(v).strip())
    return str(value).strip()


# 构造更适合检索和向量化的文本。
def build_text(item):
    title = _norm_text(item.get("title", ""))
    content = _norm_text(item.get("content", ""))
    keywords = _norm_text(item.get("keywords", []))
    source = _norm_text(item.get("source", ""))
    domain = _norm_text(item.get("domain", ""))
    role = _norm_text(item.get("role", ""))
    section = _norm_text(item.get("section", item.get("chapter", "")))
    subtitle = _norm_text(item.get("subtitle", ""))
    summary = _norm_text(item.get("summary", ""))
    tags = _norm_text(item.get("tags", []))

    parts = [
        f"领域：{domain}",
        f"来源：{source}",
    ]
    if role:
        parts.append(f"角色：{role}")
    if section:
        parts.append(f"章节：{section}")
    if title:
        parts.append(f"标题：{title}")
    if subtitle:
        parts.append(f"副标题：{subtitle}")
    if keywords:
        parts.append(f"关键词：{keywords}")
    if tags:
        parts.append(f"标签：{tags}")
    if summary:
        parts.append(f"摘要：{summary}")
    if content:
        parts.append(f"正文：{content}")

    return "\n".join(parts).strip()

def read_jsonl(path):
    # 逐行读取 JSONL，避免一次性处理失败。
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# 逐行写回 JSONL。
def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# 主流程：读取、拼接文本、生成向量、写回磁盘。
def main():
    # 读取全部样本。
    items = list(read_jsonl(INPUT_PATH))
    # 为每条样本构造用于检索的文本。
    texts = [build_text(item) for item in items]

    # 使用 BGE-M3 生成 dense 向量。
    results = model.encode(
        texts,
        batch_size=16,
        max_length=8192
    )

    # 取出 dense 向量。
    dense_vecs = results["dense_vecs"]

    # 将向量和增强文本写回每条记录。
    output_rows = []
    for item, vec, text in tqdm(zip(items, dense_vecs, texts), total=len(items)):
        item["vector"] = vec.tolist() if hasattr(vec, "tolist") else vec
        item["vector_text"] = text
        item["search_text"] = text
        output_rows.append(item)

    # 保存到新的 JSONL 文件。
    write_jsonl(OUTPUT_PATH, output_rows)
    print(f"完成，输出文件：{OUTPUT_PATH}")

if __name__ == "__main__":
    main()
