"""检查检索质量"""
import sys, os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
sys.path.insert(0, ".")

from src.embedder.embed import Embedder
from src.store.milvus_store import MilvusStore
from src.store.keyword_store import BM25Index

# 1. 向量搜索
embedder = Embedder()
milvus = MilvusStore()
milvus.connect()
print(f"Milvus: {milvus.count()} docs")

q = "公司注册资本是多少"
qv = embedder.embed(q)
hits = milvus.search(qv, top_k=5)
print(f"\n--- 向量搜索 Top-5 ---")
for i, h in enumerate(hits):
    t = h["text"][:80].replace("\n", " ")
    print(f"  #{i+1} score={h['score']:.4f} page={h['page_no']} type={h.get('chunk_type','')} text={t}")

# 2. BM25 搜索
bm25 = BM25Index()
bm25.load(r"D:\Desktop\rag-hermes\teams\team\project\data\bm25_index.pkl")
bhits = bm25.search(q, top_k=5)
print(f"\n--- BM25 搜索 Top-5 ---")
for i, h in enumerate(bhits):
    t = h["text"][:80].replace("\n", " ")
    print(f"  #{i+1} score={h['score']:.4f} page={h['page_no']} type={h.get('chunk_type','')} text={t}")
