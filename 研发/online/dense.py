# 这一层只是把“文本问题 -> 向量检索”这件事包装得更好用一些。
# 上层只要传文本，不需要关心向量怎么生成。
from typing import List

from pymilvus import Collection

from .models import RetrievedDoc
from .retrieval import embed_query, search_dense as _search_dense


# 对外提供 dense 召回接口。
def search_dense(collection: Collection, query_text: str, top_k: int) -> List[RetrievedDoc]:
    # 第一步：把用户问题转成向量。
    query_vector = embed_query(query_text)
    # 第二步：交给底层检索实现去搜相似文档。
    return _search_dense(collection, query_vector, top_k)
