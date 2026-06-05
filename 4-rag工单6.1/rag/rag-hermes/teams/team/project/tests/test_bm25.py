import sys;

sys.path.insert(0, '.')
from src.store.keyword_store import BM25Index

bm25 = BM25Index()
bm25.load(r'D:\Desktop\rag-hermes\teams\team\project\data\bm25_index.pkl')
print(f"BM25 loaded: {len(bm25._chunks)} chunks")
# check first chunk has chunk_type
c = bm25._chunks[0]
print(f"First chunk keys: {list(c.keys())}")
print(f"chunk_type: {c.get('chunk_type', 'MISSING')}")
# search test
results = bm25.search('公司注册资本', top_k=3)
for r in results:
    ct = r.get('chunk_type', 'N/A')
    sp = r.get('section_path', '')[:40]
    txt = r['text'][:60]
    print(f"  type={ct}  section={sp}  text={txt}")
