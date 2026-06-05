"""
BM25 关键词索引
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
"""

import pickle
import jieba
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi


class BM25Index:
    """BM25 关键词检索索引"""

    def __init__(self):
        self._index: BM25Okapi | None = None
        self._chunks: List[Dict[str, Any]] = []
        self._tokenized_corpus: List[List[str]] = []

    def _tokenize(self, text: str) -> List[str]:
        """中文分词"""
        return list(jieba.cut(text))

    def build_index(self, chunks: List[Dict[str, Any]]) -> None:
        """从分块列表构建 BM25 索引"""
        self._chunks = chunks
        self._tokenized_corpus = [self._tokenize(c["text"]) for c in chunks]
        self._index = BM25Okapi(self._tokenized_corpus)

    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """BM25 检索

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            [{"text": str, "page_no": int, "source_file": str, "score": float}, ...]
        """
        if self._index is None:
            return []

        tokenized_query = self._tokenize(query)
        scores = self._index.get_scores(tokenized_query)

        # 排序取 top_k
        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1], reverse=True)
        top = indexed[:top_k]

        results = []
        for idx, score in top:
            if score <= 0:
                continue
            chunk = self._chunks[idx]
            results.append({
                "text": chunk["text"],
                "page_no": chunk["page_no"],
                "source_file": chunk["source_file"],
                "score": float(score),
            })
        return results

    def save(self, path: str) -> None:
        """持久化到磁盘"""
        data = {
            "chunks": self._chunks,
            "tokenized_corpus": self._tokenized_corpus,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str) -> None:
        """从磁盘加载"""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._chunks = data["chunks"]
        self._tokenized_corpus = data["tokenized_corpus"]
        self._index = BM25Okapi(self._tokenized_corpus)
