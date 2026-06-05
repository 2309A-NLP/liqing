from pymilvus import MilvusClient

client = MilvusClient("http://localhost:19530")
COLLECTION = "pdf_chunks"

# 查前5条刚存的医疗数据
res = client.query(
    collection_name=COLLECTION,
    filter='domain == "医疗"',  # 只查刚存的医疗数据
    output_fields=["id", "title", "content", "source", "vector_text", "embedding"],  # 查看关键字段
    limit=5
)

print("刚存的医疗数据示例：")
for idx, item in enumerate(res):
    print(f"\n--- 第{idx + 1}条 ---")
    print(f"ID: {item['id']}")
    print(f"标题（患者问题）: {item['title']}")
    print(f"内容: {item['content'][:100]}...")  # 只显示前100字，避免过长
    print(f"来源: {item['source']}")
    print(f"向量文本: {item['vector_text']}")
    print(f"向量: {item['embedding']}")
