"""从pdf_chunks中删除9个PDF的数据，保留招股说明书"""
from pymilvus import connections, Collection

connections.connect(host="localhost", port=19530)
col = Collection("pdf_chunks")
col.load()

print(f"pdf_chunks 删除前总数: {col.num_entities}")

pdf_files = [
    "中信证券", "中国人寿保险", "中国平安保险", "中国邮政储蓄银行",
    "国泰君安证券", "太平洋保险", "平安银行", "招商证券", "招商银行",
]

for name in pdf_files:
    expr = f'source_file == "{name}"'
    col.delete(expr)
    print(f"  已删除: {name}")

col.flush()
print(f"pdf_chunks 剩余: {col.num_entities}")
