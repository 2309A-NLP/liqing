# 这一层和 dense 一样，都是给上层提供更简单的调用入口。
# 这里只是把关键词检索封装成一个独立函数。
from typing import List

from pymilvus import Collection

from .models import RetrievedDoc
from .retrieval import search_bm25 as _search_bm25


# 对外提供 bm25 召回接口。
def search_bm25(collection: Collection, query_text: str, bm25_top_k: int) -> List[RetrievedDoc]:
    # 具体的 BM25 计算逻辑交给底层实现。
    return _search_bm25(collection, query_text, bm25_top_k)



